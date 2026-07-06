"""
flexeme_faithful.py — Flexeme (2020 FSE) faithful reimplementation

 Section 3
 1. Build delta-PDG AST/PDG
 2. ExtractChangeNode 1-hop Graph
 3. WL subtree kernel (k=10 iterations) ComputeGraphSimilarity
 4. Complete-linkage Cluster (distance_threshold=0.5)
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.spatial.distance
import scipy.cluster.hierarchy

from ..delta_pdg import DeltaPDG, DeltaNode, DeltaEdge, build_delta_pdg, build_delta_pdg_full
from ..source_resolver import FileSource, reconstruct_pre_post_from_diff, resolve_entry


# ---------------------------------------------------------------------------
# 1-hop GraphExtract ( Section 3.3)
# ---------------------------------------------------------------------------

def _build_adjacency(delta: DeltaPDG):
    """Build adjacency list and edge kind mapping from delta-PDG (undirected)."""
    n = len(delta.nodes)
    adj = defaultdict(set)
    edge_kinds = defaultdict(set)

    for edge in delta.edges:
        adj[edge.src].add(edge.dst)
        adj[edge.dst].add(edge.src)
        edge_kinds[(edge.src, edge.dst)].add(edge.kind)
        edge_kinds[(edge.dst, edge.src)].add(edge.kind)
        if hasattr(edge, 'flow_names') and edge.flow_names:
            for name in edge.flow_names:
                edge_kinds[(edge.src, edge.dst)].add(f"nf:{name}")
                edge_kinds[(edge.dst, edge.src)].add(f"nf:{name}")

    return adj, edge_kinds


def _build_directed_adjacency(delta: DeltaPDG):
    """Build DIRECTED adjacency (outgoing only) — matches paper's nx.single_source_shortest_path_length."""
    adj = defaultdict(set)
    edge_kinds = defaultdict(set)

    for edge in delta.edges:
        adj[edge.src].add(edge.dst)  # only outgoing
        edge_kinds[(edge.src, edge.dst)].add(edge.kind)
        if hasattr(edge, 'flow_names') and edge.flow_names:
            for name in edge.flow_names:
                edge_kinds[(edge.src, edge.dst)].add(f"nf:{name}")

    return adj, edge_kinds


def _extract_subgraph(seed: int, adj: Dict, edge_kinds: Dict,
                      delta: DeltaPDG, khop: int = 1):
    """
    Extract k-hop subgraph around a seed node.

    Matches paper: BFS via directed (outgoing) edges to select nodes,
    then build INDUCED subgraph (all edges between selected nodes).

    Returns:
        nodes: list of node indices in the subgraph
        sub_adj: adjacency dict within subgraph (local indices)
        node_labels: label for each node (local index)
        edge_labels: label for each edge (local index pair)
    """
    # BFS to find k-hop neighborhood (outgoing edges only, matching
    # nx.single_source_shortest_path_length on DiGraph)
    visited = {seed}
    frontier = {seed}
    for _ in range(khop):
        next_frontier = set()
        for node in frontier:
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier

    # Build subgraph with local indices
    nodes = sorted(visited)
    global_to_local = {g: l for l, g in enumerate(nodes)}

    # Build INDUCED subgraph: include ALL edges between selected nodes
    # (matching paper's delta.subgraph(nodes) which preserves all edges)
    sub_adj = defaultdict(set)
    edge_labels = {}
    for edge in delta.edges:
        if edge.src in visited and edge.dst in visited:
            l_src = global_to_local[edge.src]
            l_dst = global_to_local[edge.dst]
            sub_adj[l_src].add(l_dst)
            # Encode edge type as integer matching paper's DOT keys:
            # 0=control, 1=data, 2=call, 3=name
            if edge.kind == 'data':
                etype = 1
            elif edge.kind == 'call':
                etype = 2
            elif edge.kind == 'name_flow':
                etype = 3
            else:
                etype = 0  # control
            edge_labels[(l_src, l_dst)] = etype

    # Node labels: ALL nodes get '0' — this matches the paper's actual behavior
    # due to a string/int comparison bug in graph_to_grakel:
    # ('1' if any([k == 1 for u, v, k in g.edges(...)]) else '0')
    # where k is a STRING (from pydot) so k == 1 (int) always fails
    node_labels = {}
    for local_idx in range(len(nodes)):
        node_labels[local_idx] = '0'

    return nodes, sub_adj, node_labels, edge_labels


# ---------------------------------------------------------------------------
# WL Subtree Kernel ( GraKeL, n_iter=10)
# ---------------------------------------------------------------------------

def _wl_feature_vector(sub_adj: Dict, node_labels: Dict, edge_labels: Dict,
                       n_nodes: int, k: int = 10) -> Counter:
    """
    Compute WL subtree feature vector for a single subgraph.

    Matches GraKeL's WL subtree kernel exactly:
    - Edge labels are NOT used in WL relabeling (only graph topology)
    - credential = str(label_of_v) + "," + str(sorted([label_of_n for n in neighbors]))
    - Feature histogram counts label occurrences across all iterations

    Returns histogram (Counter) across all iterations.
    """
    features = Counter()
    current = dict(node_labels)  # local_idx -> label string

    # Iteration 0: count initial labels
    for label in current.values():
        features[f"h0:{label}"] += 1

    for iteration in range(1, k + 1):
        new_labels = {}
        for node_idx in range(n_nodes):
            neighbors = sorted(sub_adj.get(node_idx, set()))
            # GraKeL WL: only use neighbor NODE labels, NO edge labels
            neighbor_labels = sorted([str(current.get(nb, '?')) for nb in neighbors])
            # credential = str(self_label) + "," + str(sorted_neighbor_labels)
            # Match GraKeL's exact format for consistent hashing
            combined = str(current.get(node_idx, '?')) + "," + str(neighbor_labels)
            new_label = hashlib.md5(combined.encode()).hexdigest()[:12]
            new_labels[node_idx] = new_label
            features[f"h{iteration}:{new_label}"] += 1
        current = new_labels

    return features


def wl_subtree_kernel_on_subgraphs(
    delta: DeltaPDG,
    change_node_ids: List[int],
    adj: Dict,
    edge_kinds: Dict,
    k: int = 10,
    khop: int = 1,
) -> List[List[float]]:
    """
    Compute WL subtree kernel similarity matrix between 1-hop subgraphs
    of changed nodes. Matches the paper's algorithm:

    1. For each changed node, extract 1-hop subgraph
    2. Compute WL feature vector per subgraph (k iterations)
    3. Compute normalized dot product (cosine) between all pairs

    Args:
        delta: Full delta-PDG
        change_node_ids: Indices of changed nodes to cluster
        adj: Precomputed adjacency
        edge_kinds: Precomputed edge kinds
        k: WL iterations (paper uses 10)
        khop: Neighborhood radius (paper uses 1)

    Returns:
        n x n similarity matrix where n = len(change_node_ids)
    """
    n = len(change_node_ids)
    if n == 0:
        return []

    # Extract subgraphs and compute feature vectors
    feature_vectors = []
    for seed in change_node_ids:
        nodes, sub_adj, node_labels, edge_labels = _extract_subgraph(
            seed, adj, edge_kinds, delta, khop
        )
        n_nodes = len(nodes)
        fv = _wl_feature_vector(sub_adj, node_labels, edge_labels, n_nodes, k)
        feature_vectors.append(fv)

    # Compute normalized kernel matrix (cosine similarity)
    norms = []
    for fv in feature_vectors:
        norm = math.sqrt(sum(v * v for v in fv.values()))
        norms.append(norm if norm > 0 else 1.0)

    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sim_matrix[i][i] = 1.0
        for j in range(i + 1, n):
            # Dot product
            smaller, larger = (feature_vectors[i], feature_vectors[j]) \
                if len(feature_vectors[i]) <= len(feature_vectors[j]) \
                else (feature_vectors[j], feature_vectors[i])
            dot = sum(v * larger.get(key, 0) for key, v in smaller.items())
            sim = dot / (norms[i] * norms[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    return sim_matrix


# ---------------------------------------------------------------------------
# Complete-linkage Cluster (: distance_threshold=0.5, linkage='complete')
# ---------------------------------------------------------------------------

def agglomerative_cluster(sim_matrix: List[List[float]],
                          threshold: float = 0.5) -> List[int]:
    """
    Complete-linkage agglomerative clustering (matches original paper).

    Uses scipy for efficient hierarchical clustering.

    Args:
        sim_matrix: n x n similarity matrix (0-1, higher = more similar)
        threshold: distance threshold (paper default = 0.5)
    """
    n = len(sim_matrix)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Convert similarity to distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = 1.0 - sim_matrix[i][j]
    # Clamp to avoid floating-point negative distances
    np.clip(dist_matrix, 0, None, out=dist_matrix)
    np.fill_diagonal(dist_matrix, 0)

    # Handle edge case: 2 nodes
    if n == 2:
        return [0, 0] if dist_matrix[0][1] <= threshold else [0, 1]

    # Condensed distance vector for scipy
    condensed = scipy.spatial.distance.squareform(dist_matrix)

    # Complete linkage (same as paper)
    linkage_matrix = scipy.cluster.hierarchy.linkage(condensed, method='complete')

    # Cut dendrogram at threshold
    labels = scipy.cluster.hierarchy.fcluster(
        linkage_matrix, t=threshold, criterion='distance'
    )

    # Convert to 0-based
    return [int(l - 1) for l in labels]


# ---------------------------------------------------------------------------
# PredictFunction
# ---------------------------------------------------------------------------

# ConfigureLinux bare repo Path
_LINUX_REPO_PATH = None


def set_repo_path(path):
    """Set Linux bare repo Path full-project PDG"""
    global _LINUX_REPO_PATH
    from pathlib import Path
    _LINUX_REPO_PATH = Path(path) if path else None


def predict_flexeme_faithful(entry: Dict) -> List[Optional[int]]:
    """
 Flexeme faithful Predict

 (Barnett et al. 2020 FSE "Flexeme"):
 1. Build delta-PDG (multiversion PDG)
 2. ExtractChangeNode 1-hop Graph
    3. WL subtree kernel (k=10, normalized cosine similarity)
 4. Complete-linkage Cluster, distance_threshold=0.5

 _LINUX_REPO_PATH FileBuild full-project PDG
    """
    diff_text = entry['source']['diff']

    # CheckChange diff
    n_changed_lines = sum(1 for line in diff_text.splitlines()
                          if line.startswith('+') and not line.startswith('+++')
                          or line.startswith('-') and not line.startswith('---'))
    if n_changed_lines > 300:
        # diff FunctionGroupSkip PDG Build
        return _fallback_from_diff(diff_text, n_changed_lines)

    # full-project PDG
    if _LINUX_REPO_PATH and _LINUX_REPO_PATH.exists():
        resolved = resolve_entry(entry, repo_path=_LINUX_REPO_PATH)
        if resolved.resolved_from_git:
            delta = build_delta_pdg_full(diff_text, resolved.files)
        else:
            file_sources = reconstruct_pre_post_from_diff(diff_text)
            delta = build_delta_pdg(diff_text, file_sources)
    else:
        file_sources = reconstruct_pre_post_from_diff(diff_text)
        delta = build_delta_pdg(diff_text, file_sources)

    # ChangeNodeClusterNode WL
    change_node_ids = [i for i, n in enumerate(delta.nodes) if n.change_type != 'ctx']
    n_change = len(change_node_ids)

    if n_change == 0:
        n_lines = sum(1 for f in entry['ground_truth']['files']
                      for la in f['lines'] if la['type'] in ('add', 'del'))
        return [0] * n_lines

    # case 2000 Node 300 ChangeNode
    if len(delta.nodes) > 2000 or n_change > 300:
        return _fallback_predict_changes_only(delta, change_node_ids)

    # Build adjacency structure once
    adj, edge_kinds = _build_adjacency(delta)

    # WL subtree kernel on 1-hop subgraphs (: GraKeL, n_iter=10)
    sim_matrix = wl_subtree_kernel_on_subgraphs(
        delta, change_node_ids, adj, edge_kinds, k=10, khop=1
    )

    # Complete-linkage clustering, distance_threshold=0.5 (Parameter)
    cluster_ids = agglomerative_cluster(sim_matrix, threshold=0.5)

    return cluster_ids


def _fallback_predict_changes_only(delta: DeltaPDG, change_node_ids: List[int]) -> List[int]:
    """ case FunctionGroupChangeNode"""
    group_map = {}
    gid = 0
    result = []
    for nid in change_node_ids:
        node = delta.nodes[nid]
        key = node.function or '__unknown__'
        if key not in group_map:
            group_map[key] = gid
            gid += 1
        result.append(group_map[key])
    return result


_HUNK_RE_FALLBACK = re.compile(r'^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@\s*(.*)')


def _fallback_from_diff(diff_text: str, n_lines: int) -> List[int]:
    """ diff hunk header FunctionGroupChange"""
    group_map = {}
    gid = 0
    result = []
    current_func = '__unknown__'

    for line in diff_text.splitlines():
        m = _HUNK_RE_FALLBACK.match(line)
        if m:
            func_ctx = m.group(1).strip()
            current_func = func_ctx if func_ctx else '__unknown__'
            continue
        if line.startswith('diff --git'):
            current_func = '__unknown__'
            continue
        if (line.startswith('+') and not line.startswith('+++')) or \
           (line.startswith('-') and not line.startswith('---')):
            if current_func not in group_map:
                group_map[current_func] = gid
                gid += 1
            result.append(group_map[current_func])

    return result


def _fallback_predict(delta: DeltaPDG) -> List[int]:
    """ case FunctionGroup"""
    group_map = {}
    gid = 0
    result = []
    for node in delta.nodes:
        key = node.function or '__unknown__'
        if key not in group_map:
            group_map[key] = gid
            gid += 1
        result.append(group_map[key])
    return result


# ---------------------------------------------------------------------------
# DOT-based prediction (Build Roslyn PDG)
# ---------------------------------------------------------------------------

from pathlib import Path

_CORPORA_ROOT = None


def set_corpora_path(path):
    """Set corpora_clean DirectoryPath"""
    global _CORPORA_ROOT
    _CORPORA_ROOT = Path(path) if path else None


def _parse_dot_file(dot_path):
    """
 Parse merged.dot File。

 Return:
        nodes: dict of node_id -> {label, span, cluster, color, community, file}
        edges: list of (src, dst, key)
    """
    content = Path(dot_path).read_text(encoding='utf-8', errors='replace')

    nodes = {}
    edges = []

    node_re = re.compile(r'^\s*(\w+)\s*\[(.+)\]\s*;?\s*$')
    edge_re = re.compile(r'^\s*(\w+)\s*->\s*(\w+)\s*\[(.+)\]\s*;?\s*$')
    attr_re = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|[^,\]]+)')

    for line in content.splitlines():
        line = line.strip()
        if line.startswith('digraph') or line == '{' or line == '}':
            continue

        em = edge_re.match(line)
        if em:
            src, dst, attrs_str = em.group(1), em.group(2), em.group(3)
            attrs = {}
            for m in attr_re.finditer(attrs_str):
                k, v = m.group(1), m.group(2).strip('"')
                attrs[k] = v
            key = int(attrs.get('key', 0))
            edges.append((src, dst, key))
            continue

        nm = node_re.match(line)
        if nm:
            nid, attrs_str = nm.group(1), nm.group(2)
            attrs = {}
            for m in attr_re.finditer(attrs_str):
                k, v = m.group(1), m.group(2).strip('"')
                attrs[k] = v
            nodes[nid] = attrs

    return nodes, edges


def _dot_to_delta_structures(nodes, edges):
    """
 DOT ParseResults WL kernel

 Return:
        delta: DeltaPDG
        change_node_ids: list of changed node indices
        community_map: {node_idx: community_id}
    """
    delta = DeltaPDG()
    node_id_map = {}  # dot_id -> index
    change_node_ids = []
    community_map = {}

    for dot_id, attrs in nodes.items():
        nid = len(delta.nodes)
        node_id_map[dot_id] = nid

        color = attrs.get('color', '')
        community = attrs.get('community')
        change_type = 'ctx'
        if color in ('green', 'darkgreen'):
            change_type = 'add'
        elif color in ('red', 'darkred'):
            change_type = 'del'
        elif color == 'orange':
            change_type = 'ctx'  # anchor node

        node = DeltaNode(
            node_id=nid,
            line=0,
            change_type=change_type,
            function=attrs.get('cluster'),
            content=attrs.get('label', ''),
            identifiers=set(),
            label=f"{change_type}:{attrs.get('cluster', '')}",
        )
        delta.nodes.append(node)

        if change_type != 'ctx' and community is not None:
            change_node_ids.append(nid)
            community_map[nid] = int(community)

    for src_dot, dst_dot, key in edges:
        if src_dot in node_id_map and dst_dot in node_id_map:
            src_id = node_id_map[src_dot]
            dst_id = node_id_map[dst_dot]
            # Preserve all 4 edge types matching paper's DOT keys:
            # 0=control, 1=data, 2=call, 3=name
            if key == 1:
                kind = 'data'
            elif key == 2:
                kind = 'call'
            elif key == 3:
                kind = 'name_flow'
            else:
                kind = 'control'
            delta.edges.append(DeltaEdge(src=src_id, dst=dst_id, kind=kind))

    return delta, change_node_ids, community_map


def _find_dot_file(entry):
    """ entry metadata merged.dot File"""
    if _CORPORA_ROOT is None:
        return None
    meta = entry.get('metadata', {})
    repo = meta.get('repo')
    commit_pair = meta.get('commit_pair')
    if not repo or not commit_pair:
        return None

    commit_dir = _CORPORA_ROOT / repo / commit_pair
    if not commit_dir.exists():
        return None

 # concepts >= 2 Directory2 2
    # concepts Directory
    best = None
    for sd in sorted(commit_dir.iterdir(), key=lambda x: x.name, reverse=True):
        if sd.is_dir():
            dot = sd / "merged.dot"
            if dot.exists():
                best = dot
                break
    return best


def predict_flexeme_from_dot(entry: Dict) -> List[Optional[int]]:
    """
 Build DOT File Flexeme Predict

 Roslyn PDG Run WL kernel
    """
    dot_path = _find_dot_file(entry)
    if dot_path is None:
        # diff-based
        return predict_flexeme_faithful(entry)

    nodes, edges = _parse_dot_file(dot_path)
    delta, change_node_ids, community_map = _dot_to_delta_structures(nodes, edges)

    n_change = len(change_node_ids)
    if n_change == 0:
        n_lines = sum(1 for f in entry['ground_truth']['files']
                      for la in f['lines'] if la['type'] in ('add', 'del'))
        return [0] * n_lines

    if n_change == 1:
        # ChangeNode
        return [0]

    # Build DIRECTED adjacency (matching paper's nx.single_source_shortest_path_length)
    adj, edge_kinds = _build_directed_adjacency(delta)
    sim_matrix = wl_subtree_kernel_on_subgraphs(
        delta, change_node_ids, adj, edge_kinds, k=10, khop=1
    )

    # Complete-linkage clustering
    cluster_ids = agglomerative_cluster(sim_matrix, threshold=0.5)

    return cluster_ids
