"""
hdgnn_source.py — HD-GNN (2024 FSE) Full GNN Architecture

Paper-faithful implementation of the hierarchical graph neural network:
  1. Entity Reference Graph: entities as nodes, syntactic relations as directed edges
  2. Code Change Relation Graph: hunks as nodes, fully-connected (classification target)

4 layers per iteration:
  1. Entity Embedding: message passing over entity graph
  2. Relation Embedding: edge update with triangle context
  3. Untangling Relation: aggregate entity/relation info per change pair
  4. Classifying Layer: binary softmax → P(same_concern)

Uses raw PyTorch (no PyG/DGL).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..source_resolver import FileSource, reconstruct_pre_post_from_diff
from ..entity_extractor import EntityGraph, EntityRef, Entity, extract_entities
from .hdgnn_faithful import (
    extract_code_changes_faithful, cluster_from_scores, _map_to_lines,
    _HUNK_RE, _IDENT_RE, _STOP_WORDS,
)
from ..ast_parser import is_c_file

# Entity kind → index mapping
ENTITY_KIND_MAP = {'function': 0, 'variable': 1, 'struct': 2, 'macro': 3, 'typedef': 4}
# Reference kind → index mapping
REF_KIND_MAP = {'call': 0, 'use': 1, 'type_use': 2}

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph_from_entry(entry: Dict, is_synthetic: bool = False) -> Optional[Dict]:
    """
    Build graph tensors from a dataset entry.

    Returns dict with:
      entity_types: LongTensor [n_entities]
      edge_index: LongTensor [2, n_edges]  (directed, entity graph)
      edge_types: LongTensor [n_edges]
      change_entity_mask: BoolTensor [n_changes, n_entities]
      n_changes: int
      pair_indices: LongTensor [2, n_pairs]  (all i<j pairs)
      pair_labels: FloatTensor [n_pairs]  (1=same, 0=different; None at test)
      hunk_groups: list (ground truth hunk-level groups, or None)
    """
    if is_synthetic:
        diff_text = entry['tangled_diff']
    else:
        diff_text = entry['source']['diff']

    file_sources = reconstruct_pre_post_from_diff(diff_text)
    entity_graph = extract_entities(file_sources)
    changes = extract_code_changes_faithful(diff_text, file_sources, entity_graph)

    if len(changes) < 2:
        return None

    # --- Build entity node list ---
    # Collect all entities referenced by any change
    all_entity_names = set()
    for c in changes:
        all_entity_names.update(c['entities'])
        all_entity_names.update(c['ast_entities'])

    # Also include entities from the graph that connect to change entities
    graph_entity_names = set(entity_graph.entities.keys())
    relevant_entities = all_entity_names | (all_entity_names & graph_entity_names)
    # Add neighbors in entity graph
    for ref in entity_graph.references:
        if ref.src in all_entity_names or ref.dst in all_entity_names:
            relevant_entities.add(ref.src)
            relevant_entities.add(ref.dst)

    entity_list = sorted(relevant_entities)
    entity_to_idx = {name: i for i, name in enumerate(entity_list)}
    n_entities = len(entity_list)

    # Cap entities to prevent OOM on very large graphs
    MAX_ENTITIES = 200
    if n_entities > MAX_ENTITIES:
        # Keep entities that appear in changes (most relevant)
        change_entities = set()
        for c in changes:
            change_entities.update(e for e in c['entities'] if e in entity_to_idx)
        # Prioritize change entities, fill rest from graph
        priority = sorted(change_entities)[:MAX_ENTITIES]
        remaining = [e for e in entity_list if e not in change_entities][:MAX_ENTITIES - len(priority)]
        entity_list = priority + remaining
        entity_to_idx = {name: i for i, name in enumerate(entity_list)}
        n_entities = len(entity_list)

    if n_entities == 0:
        return None

    # Entity types
    entity_types = []
    for name in entity_list:
        if name in entity_graph.entities:
            kind = entity_graph.entities[name].kind
            entity_types.append(ENTITY_KIND_MAP.get(kind, 0))
        else:
            entity_types.append(1)  # default: variable
    entity_types_t = torch.tensor(entity_types, dtype=torch.long)

    # --- Build edges (entity reference graph) ---
    src_list, dst_list, edge_type_list = [], [], []
    for ref in entity_graph.references:
        if ref.src in entity_to_idx and ref.dst in entity_to_idx:
            src_list.append(entity_to_idx[ref.src])
            dst_list.append(entity_to_idx[ref.dst])
            edge_type_list.append(REF_KIND_MAP.get(ref.kind, 0))

    # Add co-occurrence edges for entities in same change (capped to avoid O(n^2))
    MAX_COOCCUR_PER_HUNK = 20
    for c in changes:
        c_ents = [entity_to_idx[e] for e in c['entities'] if e in entity_to_idx]
        if len(c_ents) > MAX_COOCCUR_PER_HUNK:
            c_ents = c_ents[:MAX_COOCCUR_PER_HUNK]
        for i_idx in range(len(c_ents)):
            for j_idx in range(i_idx + 1, len(c_ents)):
                src_list.append(c_ents[i_idx])
                dst_list.append(c_ents[j_idx])
                edge_type_list.append(1)  # 'use' type for co-occurrence

    n_edges = len(src_list)
    if n_edges == 0:
        # Add dummy self-edge to avoid empty graph
        src_list, dst_list, edge_type_list = [0], [0], [0]
        n_edges = 1

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_types = torch.tensor(edge_type_list, dtype=torch.long)

    # --- Change-entity membership mask ---
    n_changes = len(changes)
    change_entity_mask = torch.zeros(n_changes, n_entities, dtype=torch.bool)
    for ci, c in enumerate(changes):
        for e in c['entities']:
            if e in entity_to_idx:
                change_entity_mask[ci, entity_to_idx[e]] = True

    # --- All pairs (i < j) ---
    pair_src, pair_dst = [], []
    for i in range(n_changes):
        for j in range(i + 1, n_changes):
            pair_src.append(i)
            pair_dst.append(j)

    n_pairs = len(pair_src)
    if n_pairs == 0:
        return None

    pair_indices = torch.tensor([pair_src, pair_dst], dtype=torch.long)

    # --- Ground truth labels (hunk-level) ---
    hunk_groups = _get_hunk_groups_for_entry(entry, diff_text, n_changes, is_synthetic)
    if hunk_groups is not None:
        pair_labels = torch.zeros(n_pairs, dtype=torch.float)
        for pi in range(n_pairs):
            ci, cj = pair_src[pi], pair_dst[pi]
            pair_labels[pi] = 1.0 if hunk_groups[ci] == hunk_groups[cj] else 0.0
    else:
        pair_labels = None

    return {
        'entity_types': entity_types_t,
        'edge_index': edge_index,
        'edge_types': edge_types,
        'change_entity_mask': change_entity_mask,
        'n_changes': n_changes,
        'n_entities': n_entities,
        'pair_indices': pair_indices,
        'pair_labels': pair_labels,
        'hunk_groups': hunk_groups,
    }


def _get_hunk_groups_for_entry(entry: Dict, diff_text: str, n_hunks: int,
                                is_synthetic: bool) -> Optional[List[int]]:
    """Extract hunk-level ground truth group assignments."""
    if is_synthetic:
        # Synthetic format: ground_truth.assignments is line-level
        assignments = entry['ground_truth']['assignments']
        if not assignments:
            return None
        return _hunk_groups_from_diff_and_assignments(diff_text, assignments, n_hunks)
    else:
        # Standard format: ground_truth.files[].lines[].group_id
        gt = entry.get('ground_truth', {})
        fk = 'files' if 'files' in gt else 'file_assignments'
        if fk not in gt:
            return None

        gt_lines = []
        for f in gt[fk]:
            for la in f['lines']:
                if la['type'] in ('add', 'del'):
                    gt_lines.append(la.get('group_id', 0))

        if not gt_lines:
            return None

        return _hunk_groups_from_gt_lines(diff_text, gt_lines, n_hunks)


def _hunk_groups_from_gt_lines(diff_text: str, gt_lines: List[int],
                                n_hunks: int) -> Optional[List[int]]:
    """Map line-level ground truth to hunk-level via majority vote."""
    hunk_groups = []
    current_hunk_groups = []
    line_idx = 0
    in_hunk = False

    for line in diff_text.splitlines():
        if line.startswith('diff --git') or line.startswith('---') or line.startswith('+++'):
            continue

        if _HUNK_RE.match(line):
            if in_hunk and current_hunk_groups:
                most_common = Counter(current_hunk_groups).most_common(1)[0][0]
                hunk_groups.append(most_common)
            current_hunk_groups = []
            in_hunk = True
            continue

        if in_hunk:
            if (line.startswith('+') and not line.startswith('+++')) or \
               (line.startswith('-') and not line.startswith('---')):
                if line_idx < len(gt_lines):
                    current_hunk_groups.append(gt_lines[line_idx])
                    line_idx += 1

    if current_hunk_groups:
        most_common = Counter(current_hunk_groups).most_common(1)[0][0]
        hunk_groups.append(most_common)

    # Align to n_hunks
    if len(hunk_groups) != n_hunks:
        if len(hunk_groups) > n_hunks:
            hunk_groups = hunk_groups[:n_hunks]
        else:
            hunk_groups.extend([0] * (n_hunks - len(hunk_groups)))

    return hunk_groups


def _hunk_groups_from_diff_and_assignments(diff_text: str, assignments: List,
                                            n_hunks: int) -> Optional[List[int]]:
    """Map synthetic assignments to hunk-level groups."""
    # assignments is a list of dicts with 'group_id'
    gt_lines = []
    for a in assignments:
        if isinstance(a, dict):
            gt_lines.append(a.get('group_id', 0))
        else:
            gt_lines.append(int(a))

    return _hunk_groups_from_gt_lines(diff_text, gt_lines, n_hunks)


# ---------------------------------------------------------------------------
# GNN Model Components
# ---------------------------------------------------------------------------

class EntityEmbeddingLayer(nn.Module):
    """
    Entity embedding update (Paper Section 4, Layer 1):
      e_i^t = MLP2(Sum_j(MLP1([e_i^{t-1}, e_j^{t-1}, r_ij^{t-1}])))
    Message passing on the entity reference graph.
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, entity_emb: torch.Tensor, rel_emb: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entity_emb: [n_entities, embed_dim]
            rel_emb: [n_edges, embed_dim]
            edge_index: [2, n_edges] (src → dst)
        Returns:
            updated entity_emb: [n_entities, embed_dim]
        """
        src, dst = edge_index[0], edge_index[1]
        n_entities = entity_emb.size(0)

        # Messages: for each edge (src→dst), dst receives message from src
        msg_input = torch.cat([
            entity_emb[dst],   # receiver
            entity_emb[src],   # sender
            rel_emb,           # edge embedding
        ], dim=-1)  # [n_edges, 3*embed_dim]

        messages = self.mlp1(msg_input)  # [n_edges, hidden_dim]

        # Scatter-add messages to destination nodes, then normalize by degree
        aggregated = torch.zeros(n_entities, messages.size(-1),
                                 device=entity_emb.device)
        aggregated.index_add_(0, dst, messages)
        # Normalize by in-degree
        degree = torch.zeros(n_entities, 1, device=entity_emb.device)
        degree.index_add_(0, dst, torch.ones(len(dst), 1, device=entity_emb.device))
        degree = degree.clamp(min=1.0)
        aggregated = aggregated / degree

        # Apply MLP2
        updated = self.norm(self.mlp2(aggregated))  # [n_entities, embed_dim]
        return updated


class RelationEmbeddingLayer(nn.Module):
    """
    Relation embedding update (Paper Section 4, Layer 2):
      r_ij^t = r_ij^{t-1} + MLP2(Sum_k(MLP1([r_ij, e_i, e_j, r_jk, r_ki])))
    Updates edge embeddings using triangle context.
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(5 * embed_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, entity_emb: torch.Tensor, rel_emb: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entity_emb: [n_entities, embed_dim]
            rel_emb: [n_edges, embed_dim]
            edge_index: [2, n_edges]
        Returns:
            updated rel_emb: [n_edges, embed_dim]
        """
        src, dst = edge_index[0], edge_index[1]
        n_edges = rel_emb.size(0)
        n_entities = entity_emb.size(0)
        embed_dim = rel_emb.size(-1)

        # For each edge (i→j), use triangle context approximation
        # outgoing[node] = mean of rel_emb for edges starting from node
        # incoming[node] = mean of rel_emb for edges ending at node
        out_sum = torch.zeros(n_entities, embed_dim, device=rel_emb.device)
        out_count = torch.zeros(n_entities, 1, device=rel_emb.device)
        out_sum.index_add_(0, src, rel_emb)
        out_count.index_add_(0, src, torch.ones(n_edges, 1, device=rel_emb.device))
        out_mean = out_sum / (out_count + 1e-8)

        in_sum = torch.zeros(n_entities, embed_dim, device=rel_emb.device)
        in_count = torch.zeros(n_entities, 1, device=rel_emb.device)
        in_sum.index_add_(0, dst, rel_emb)
        in_count.index_add_(0, dst, torch.ones(n_edges, 1, device=rel_emb.device))
        in_mean = in_sum / (in_count + 1e-8)

        # For edge (i→j): [r_ij, e_i, e_j, out_mean[j](≈r_jk), in_mean[i](≈r_ki)]
        context = torch.cat([
            rel_emb,
            entity_emb[src],
            entity_emb[dst],
            out_mean[dst],
            in_mean[src],
        ], dim=-1)  # [n_edges, 5*embed_dim]

        update = self.mlp2(self.mlp1(context))  # [n_edges, embed_dim]

        # Residual connection + norm
        return self.norm(rel_emb + update)


class UntanglingRelationLayer(nn.Module):
    """
    Untangling Relation Embedding (Paper Section 4, Layer 3):
    For each change pair (i,j), aggregate entity/relation info from both changes.
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        # Input: concat of two change representations (each from entity pool + relation pool)
        self.mlp = nn.Sequential(
            nn.Linear(4 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, entity_emb: torch.Tensor, rel_emb: torch.Tensor,
                edge_index: torch.Tensor,
                change_entity_mask: torch.Tensor,
                pair_indices: torch.Tensor,
                prev_untangle_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            entity_emb: [n_entities, embed_dim]
            rel_emb: [n_edges, embed_dim]
            edge_index: [2, n_edges]
            change_entity_mask: [n_changes, n_entities]
            pair_indices: [2, n_pairs]
            prev_untangle_emb: [n_pairs, embed_dim] or None
        Returns:
            untangle_emb: [n_pairs, embed_dim]
        """
        n_changes = change_entity_mask.size(0)
        embed_dim = entity_emb.size(-1)
        src_edges, dst_edges = edge_index[0], edge_index[1]

        # Pool entity embeddings per change (mean of entities in change)
        # change_entity_mask: [n_changes, n_entities], entity_emb: [n_entities, embed_dim]
        mask_float = change_entity_mask.float()  # [n_changes, n_entities]
        mask_count = mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)  # [n_changes, 1]
        change_entity_pool = (mask_float @ entity_emb) / mask_count  # [n_changes, embed_dim]

        # Pool relation embeddings per change (mean)
        edge_src_membership = change_entity_mask[:, src_edges]  # [n_changes, n_edges]
        edge_dst_membership = change_entity_mask[:, dst_edges]  # [n_changes, n_edges]
        edge_membership = (edge_src_membership | edge_dst_membership).float()
        edge_count = edge_membership.sum(dim=1, keepdim=True).clamp(min=1.0)
        change_rel_pool = (edge_membership @ rel_emb) / edge_count  # [n_changes, embed_dim]

        # For each pair (i, j): concat [entity_pool_i, rel_pool_i, entity_pool_j, rel_pool_j]
        pi, pj = pair_indices[0], pair_indices[1]
        pair_input = torch.cat([
            change_entity_pool[pi],
            change_rel_pool[pi],
            change_entity_pool[pj],
            change_rel_pool[pj],
        ], dim=-1)  # [n_pairs, 4*embed_dim]

        update = self.mlp(pair_input)  # [n_pairs, embed_dim]

        # Residual + norm
        if prev_untangle_emb is not None:
            return self.norm(prev_untangle_emb + update)
        return self.norm(update)


class ClassifyingLayer(nn.Module):
    """
    Classifying Layer (Paper Section 4, Layer 4):
    Linear(embed_dim, 2) + softmax → P(same_concern, different_concern)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 2)

    def forward(self, untangle_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            untangle_emb: [n_pairs, embed_dim]
        Returns:
            probs: [n_pairs, 2] (col 0 = P(same), col 1 = P(different))
        """
        logits = self.linear(untangle_emb)
        return F.softmax(logits, dim=-1)


class HDGNNModel(nn.Module):
    """
    Full HD-GNN model (Paper Section 4).

    Two-level hierarchical GNN:
      - Lower level: Entity Reference Graph (entities + syntactic relations)
      - Upper level: Code Change Relation Graph (change pairs, binary classification)

    Iterates n_iters times through the 4 layers.
    """

    def __init__(self, n_entity_types: int = 5, n_rel_types: int = 3,
                 embed_dim: int = 64, hidden_dim: int = 128, n_iters: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_iters = n_iters

        # Learnable type embeddings
        self.entity_type_emb = nn.Embedding(n_entity_types, embed_dim)
        self.rel_type_emb = nn.Embedding(n_rel_types, embed_dim)

        # Shared layers across iterations (weight sharing as per paper)
        self.entity_layer = EntityEmbeddingLayer(embed_dim, hidden_dim)
        self.relation_layer = RelationEmbeddingLayer(embed_dim, hidden_dim)
        self.untangling_layer = UntanglingRelationLayer(embed_dim, hidden_dim)
        self.classifying_layer = ClassifyingLayer(embed_dim)

    def forward(self, graph_data: Dict) -> torch.Tensor:
        """
        Args:
            graph_data: dict from build_graph_from_entry (tensors moved to device)
        Returns:
            pair_probs: [n_pairs, 2] — P(same), P(different)
        """
        entity_types = graph_data['entity_types']
        edge_index = graph_data['edge_index']
        edge_types = graph_data['edge_types']
        change_entity_mask = graph_data['change_entity_mask']
        pair_indices = graph_data['pair_indices']

        # Initialize embeddings from type
        entity_emb = self.entity_type_emb(entity_types)   # [n_entities, embed_dim]
        rel_emb = self.rel_type_emb(edge_types)           # [n_edges, embed_dim]
        untangle_emb = None

        # Iterate through GNN layers
        for _ in range(self.n_iters):
            # Layer 1: Entity embedding update
            entity_emb = entity_emb + self.entity_layer(entity_emb, rel_emb, edge_index)

            # Layer 2: Relation embedding update
            rel_emb = self.relation_layer(entity_emb, rel_emb, edge_index)

            # Layer 3: Untangling relation embedding
            untangle_emb = self.untangling_layer(
                entity_emb, rel_emb, edge_index,
                change_entity_mask, pair_indices, untangle_emb
            )

        # Layer 4: Classification
        pair_probs = self.classifying_layer(untangle_emb)
        return pair_probs


# ---------------------------------------------------------------------------
# Prediction interface
# ---------------------------------------------------------------------------

def predict_hdgnn_source(entry: Dict, model: HDGNNModel,
                         device: torch.device = None) -> List[Optional[int]]:
    """
    Run trained HD-GNN model on an entry to produce line-level predictions.

    Args:
        entry: dataset entry (standard format with source.diff)
        model: trained HDGNNModel
        device: torch device
    Returns:
        List of group IDs per changed line
    """
    if device is None:
        device = next(model.parameters()).device

    diff_text = entry['source']['diff']
    num_groups = entry['ground_truth']['num_groups']

    # Build graph
    graph_data = build_graph_from_entry(entry, is_synthetic=False)
    if graph_data is None:
        n_lines = sum(1 for f in entry['ground_truth'].get('files', entry['ground_truth'].get('file_assignments', []))
                      for la in f['lines'] if la['type'] in ('add', 'del'))
        return [0] * n_lines

    # Move to device
    gd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
           for k, v in graph_data.items()}

    # Forward pass
    model.eval()
    with torch.no_grad():
        pair_probs = model(gd)  # [n_pairs, 2]

    # Extract P(same_concern) scores
    same_scores = pair_probs[:, 0].cpu().numpy()  # [n_pairs]

    # Build score matrix
    n_changes = graph_data['n_changes']
    score_matrix = np.zeros((n_changes, n_changes))
    pair_indices = graph_data['pair_indices'].numpy()
    for pi in range(pair_indices.shape[1]):
        ci, cj = pair_indices[0, pi], pair_indices[1, pi]
        score_matrix[ci, cj] = same_scores[pi]
        score_matrix[cj, ci] = same_scores[pi]

    # Cluster
    if n_changes <= num_groups:
        change_clusters = list(range(n_changes))
    else:
        change_clusters = cluster_from_scores(score_matrix, num_groups)

    # Map to lines
    return _map_to_lines(diff_text, change_clusters)