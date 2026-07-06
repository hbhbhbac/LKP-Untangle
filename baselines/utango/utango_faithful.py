"""
utango_faithful.py — UTANGO (2022 FSE)

  1. Label-GCN: AST GCN
  2. AST
  3.

 tree-sitter
  -
  - AST Label-GCN label
  -
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..ast_parser import ParsedFile, parse_c_file, map_line_to_function, is_c_file
from ..source_resolver import FileSource, reconstruct_pre_post_from_diff

_HUNK_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@\s*(.*)')
_IDENT_RE = re.compile(r'\b([a-zA-Z_]\w{2,})\b')
_STOP_WORDS = {
    'int', 'char', 'void', 'long', 'short', 'unsigned', 'signed',
    'float', 'double', 'struct', 'enum', 'union', 'typedef',
    'const', 'static', 'extern', 'volatile', 'register', 'inline',
    'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case',
    'break', 'continue', 'goto', 'default', 'sizeof', 'NULL', 'null',
    'true', 'false', 'bool', 'include', 'define', 'ifdef', 'ifndef',
    'endif', 'elif', 'undef', 'pragma',
}

# AST Label-GCN
_AST_LABELS = {
    'decl': 0, 'assign': 1, 'call': 2, 'control': 3,
    'return': 4, 'macro': 5, 'other': 6,
}


def predict_utango_faithful(entry: Dict) -> List[Optional[int]]:
    """
    UTANGO faithful
    """
    diff_text = entry['source']['diff']
    num_groups = entry['ground_truth']['num_groups']

    file_sources = reconstruct_pre_post_from_diff(diff_text)
    parsed_files = {}
    for fs in file_sources:
        source = fs.post_content or fs.pre_content
        if source and is_c_file(fs.filepath):
            parsed_files[fs.filepath] = parse_c_file(source)

    nodes, adj = _build_graph_faithful(diff_text, parsed_files)
    n = len(nodes)

    if n == 0:
        n_lines = sum(1 for f in entry['ground_truth']['files']
                      for la in f['lines'] if la['type'] in ('add', 'del'))
        return [0] * n_lines

    # TF-IDF + AST label one-hot
    features = _build_features_faithful(nodes)

    # GCN
    if n <= 500:
        features = _gcn_propagate(features, adj, rounds=2)

    sim = _cosine_similarity(features)

    if n > 50:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        dist = 1.0 - np.clip(sim, 0, 1)
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='average')
        labels = fcluster(Z, t=num_groups, criterion='maxclust')
        cluster_ids = [int(l - 1) for l in labels]
    else:
        cluster_ids = _agglomerative_avg(sim, num_groups)

    cluster_ids = _clone_postprocess(nodes, cluster_ids)

    result = []
    idx = 0
    for f in entry['ground_truth']['files']:
        for la in f['lines']:
            if la['type'] in ('add', 'del'):
                if idx < len(cluster_ids):
                    result.append(cluster_ids[idx])
                else:
                    result.append(0)
                idx += 1

    return result


def _build_graph_faithful(diff_text: str,
                          parsed_files: Dict[str, ParsedFile]) -> Tuple[List[Dict], np.ndarray]:
    """
     = = AST
      - hunk (weight=0.5)
      - (AST-based, weight=0.4)
      - (weight=0.2 * overlap)
      - AST (weight=0.1)
    """
    nodes = []
    current_file = None
    current_hunk_func = None
    post_line = 0

    for line in diff_text.splitlines():
        if line.startswith('diff --git'):
            parts = line.split(' b/')
            current_file = parts[-1].strip() if len(parts) >= 2 else ''
            continue
        if line.startswith('+++ b/'):
            current_file = line[6:].strip()
            continue
        if line.startswith('--- '):
            continue

        m = _HUNK_RE.match(line)
        if m:
            post_line = int(m.group(3))
            current_hunk_func = m.group(5).strip() if m.group(5) else None
            continue

        if (line.startswith('+') and not line.startswith('+++')) or \
           (line.startswith('-') and not line.startswith('---')):
            content = line[1:]
            tokens = [t for t in _IDENT_RE.findall(content) if t not in _STOP_WORDS]

            # AST-based
            func = current_hunk_func
            if line.startswith('+') and current_file in parsed_files:
                real_func = map_line_to_function(parsed_files[current_file], post_line)
                if real_func:
                    func = real_func

            # AST
            ast_label = _classify_line(content)

            nodes.append({
                'idx': len(nodes),
                'file': current_file,
                'func': func,
                'content': content,
                'tokens': tokens,
                'identifiers': set(tokens),
                'ast_label': ast_label,
                'type': 'add' if line.startswith('+') else 'del',
            })

            if line.startswith('+'):
                post_line += 1
        else:
            post_line += 1

    n = len(nodes)
    if n == 0:
        return nodes, np.zeros((0, 0))

    adj = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            ni, nj = nodes[i], nodes[j]
            weight = 0.0

            # hunk
            if abs(i - j) == 1 and ni['file'] == nj['file']:
                weight += 0.5

            # (AST-based)
            if ni['file'] == nj['file'] and ni['func'] and ni['func'] == nj['func']:
                weight += 0.4

            shared = ni['identifiers'] & nj['identifiers']
            if shared:
                weight += 0.2 * min(len(shared) / 3.0, 1.0)

            # AST
            if ni['ast_label'] == nj['ast_label'] and ni['ast_label'] != 'other':
                weight += 0.1

            if weight > 0:
                adj[i][j] = weight
                adj[j][i] = weight

    return nodes, adj


def _classify_line(content: str) -> str:
    """ AST """
    stripped = content.strip()
    if not stripped:
        return 'other'
    if stripped.startswith('#'):
        return 'macro'
    if stripped.startswith('return'):
        return 'return'
    if any(stripped.startswith(kw) for kw in ('if', 'for', 'while', 'switch', 'else', 'do')):
        return 'control'
    if '(' in stripped and '=' not in stripped.split('(')[0]:
        return 'call'
    if '=' in stripped and '==' not in stripped:
        return 'assign'
    if any(kw in stripped for kw in ('int ', 'char ', 'void ', 'struct ', 'unsigned ',
                                      'long ', 'static ', 'const ')):
        return 'decl'
    return 'other'


def _build_features_faithful(nodes: List[Dict]) -> np.ndarray:
    """
    TF-IDF + AST label one-hot
    """
    n = len(nodes)

    # TF-IDF
    vocab = {}
    doc_freq = Counter()
    tokenized = []

    for node in nodes:
        tokens = node['tokens']
        tokenized.append(tokens)
        for t in set(tokens):
            doc_freq[t] += 1

    top_words = [w for w, _ in doc_freq.most_common(500)]
    for i, w in enumerate(top_words):
        vocab[w] = i

    n_vocab = len(vocab)
    n_labels = len(_AST_LABELS)
    dim = n_vocab + n_labels

    features = np.zeros((n, dim))

    for i, (node, tokens) in enumerate(zip(nodes, tokenized)):
        # TF-IDF
        tf = Counter(tokens)
        for t, count in tf.items():
            if t in vocab:
                j = vocab[t]
                idf = math.log(n / (doc_freq[t] + 1)) + 1
                features[i, j] = count * idf

        # AST label one-hot
        label_idx = _AST_LABELS.get(node['ast_label'], _AST_LABELS['other'])
        features[i, n_vocab + label_idx] = 1.0

    # L2
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    features = features / norms

    return features


def _gcn_propagate(features: np.ndarray, adj: np.ndarray,
                   rounds: int = 2) -> np.ndarray:
    """GCN """
    n = features.shape[0]
    if n == 0:
        return features

    degree = adj.sum(axis=1)
    degree[degree == 0] = 1
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt
    adj_hat = adj_norm + np.eye(n)

    h = features
    for _ in range(rounds):
        h = adj_hat @ h

    norms = np.linalg.norm(h, axis=1, keepdims=True)
    norms[norms == 0] = 1
    h = h / norms

    return h


def _cosine_similarity(features: np.ndarray) -> np.ndarray:
    """"""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = features / norms
    return normalized @ normalized.T


def _agglomerative_avg(sim_matrix: np.ndarray, num_groups: int) -> List[int]:
    """Average-linkage """
    n = sim_matrix.shape[0]
    if n <= num_groups:
        return list(range(n))

    active = list(range(n))
    membership = {i: [i] for i in range(n)}

    while len(active) > num_groups:
        best_sim = -2
        best_pair = (0, 1)
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                ci, cj = active[ii], active[jj]
                total = sum(sim_matrix[a, b]
                            for a in membership[ci]
                            for b in membership[cj])
                avg = total / (len(membership[ci]) * len(membership[cj]))
                if avg > best_sim:
                    best_sim = avg
                    best_pair = (ii, jj)

        ii, jj = best_pair
        ci, cj = active[ii], active[jj]
        membership[ci] = membership[ci] + membership[cj]
        del membership[cj]
        active.pop(jj)

    result = [0] * n
    for gid, key in enumerate(active):
        for idx in membership[key]:
            result[idx] = gid
    return result


def _clone_postprocess(nodes: List[Dict], cluster_ids: List[int]) -> List[int]:
    """ majority voting """
    content_groups = defaultdict(list)
    for i, node in enumerate(nodes):
        content = node['content'].strip()
        if len(content) > 10:
            content_groups[content].append(i)

    result = list(cluster_ids)
    for content, indices in content_groups.items():
        if len(indices) < 2:
            continue
        votes = Counter(result[i] for i in indices)
        majority = votes.most_common(1)[0][0]
        for i in indices:
            result[i] = majority

    return result
