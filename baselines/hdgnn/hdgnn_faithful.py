"""
hdgnn_faithful.py — HD-GNN (2024 FSE)

  1. AST Entity Reference Graph
  2. Code Change Relation Graph (CCRG): hunk
  3. Edge features: /
  4. + connected components
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..ast_parser import parse_c_file, ParsedFile, is_c_file
from ..source_resolver import FileSource, reconstruct_pre_post_from_diff, resolve_entry
from ..entity_extractor import (
    EntityGraph, extract_entities, get_entities_for_lines, get_shared_entities
)

_HUNK_RE = re.compile(r'^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@\s*(.*)')
_IDENT_RE = re.compile(r'\b([a-zA-Z_]\w{2,})\b')
_FUNC_CALL_RE = re.compile(r'\b([a-zA-Z_]\w+)\s*\(')
_STOP_WORDS = {
    'int', 'char', 'void', 'long', 'short', 'unsigned', 'signed',
    'float', 'double', 'struct', 'enum', 'union', 'typedef',
    'const', 'static', 'extern', 'volatile', 'register', 'inline',
    'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case',
    'break', 'continue', 'goto', 'default', 'sizeof', 'NULL', 'null',
    'true', 'false', 'bool', 'include', 'define', 'ifdef', 'ifndef',
    'endif', 'elif', 'undef', 'pragma', 'printk', 'dev_err',
    'dev_warn', 'dev_info', 'pr_err', 'pr_info', 'pr_warn',
}


# ---------------------------------------------------------------------------
# Code Change (hunk )
# ---------------------------------------------------------------------------

def extract_code_changes_faithful(diff_text: str,
                                  file_sources: List[FileSource],
                                  entity_graph: EntityGraph) -> List[Dict]:
    """
     code changes (hunk )
    - AST
    - entity graph
    """
    changes = []
    current_file = None
    current_func = None
    current_change = None

    # post
    parsed_files = {}
    for fs in file_sources:
        if fs.post_content and is_c_file(fs.filepath):
            parsed_files[fs.filepath] = parse_c_file(fs.post_content)

    post_line = 0

    for line in diff_text.splitlines():
        if line.startswith('diff --git'):
            if current_change and current_change['lines']:
                _enrich_change(current_change, entity_graph)
                changes.append(current_change)
                current_change = None
            parts = line.split(' b/')
            current_file = parts[-1].strip() if len(parts) >= 2 else ''
            current_func = None
            continue
        if line.startswith('+++ b/'):
            current_file = line[6:].strip()
            continue

        m = _HUNK_RE.match(line)
        if m:
            if current_change and current_change['lines']:
                _enrich_change(current_change, entity_graph)
                changes.append(current_change)

            hunk_func = m.group(1).strip() if m.group(1) else None
            # hunk header post
            hunk_header = line
            hm = re.match(r'^@@ -\d+(?:,\d+)? \+(\d+)', hunk_header)
            post_line = int(hm.group(1)) if hm else 0

            current_change = {
                'file': current_file,
                'func': hunk_func,
                'lines': [],
                'line_nums': [],
                'entities': set(),
                'func_calls': set(),
                'ast_entities': set(),
            }
            continue

        if current_change is not None:
            if line.startswith('+') and not line.startswith('+++'):
                content = line[1:]
                current_change['lines'].append(content)
                current_change['line_nums'].append(post_line)
                post_line += 1

                idents = set(_IDENT_RE.findall(content)) - _STOP_WORDS
                current_change['entities'].update(idents)
                calls = set(_FUNC_CALL_RE.findall(content)) - _STOP_WORDS
                current_change['func_calls'].update(calls)

                # AST
                if current_file in parsed_files:
                    from ..ast_parser import map_line_to_function
                    real_func = map_line_to_function(
                        parsed_files[current_file], post_line - 1
                    )
                    if real_func:
                        current_change['func'] = real_func

            elif line.startswith('-') and not line.startswith('---'):
                content = line[1:]
                current_change['lines'].append(content)
                idents = set(_IDENT_RE.findall(content)) - _STOP_WORDS
                current_change['entities'].update(idents)
                calls = set(_FUNC_CALL_RE.findall(content)) - _STOP_WORDS
                current_change['func_calls'].update(calls)
            else:
                post_line += 1

    if current_change and current_change['lines']:
        _enrich_change(current_change, entity_graph)
        changes.append(current_change)

    return changes


def _enrich_change(change: Dict, entity_graph: EntityGraph):
    """ entity graph change"""
    # change
    entity_names = set(entity_graph.entities.keys())
    change['ast_entities'] = change['entities'] & entity_names


# ---------------------------------------------------------------------------
# & Edge Scoring
# ---------------------------------------------------------------------------

def compute_node_features(changes: List[Dict],
                          entity_graph: EntityGraph) -> np.ndarray:
    """
    - TF-IDF entity graph
    - one-hot
    - one-hot
    - AST
    -
    """
    n = len(changes)
    if n == 0:
        return np.zeros((0, 0))

    all_entities = Counter()
    for c in changes:
        all_entities.update(c['entities'])

    # entity graph
    known_entities = set(entity_graph.entities.keys())
    top_entities = []
    for e, _ in all_entities.most_common(300):
        top_entities.append(e)
    entity_vocab = {e: i for i, e in enumerate(top_entities)}

    files = sorted(set(c['file'] for c in changes))
    file_vocab = {f: i for i, f in enumerate(files)}

    funcs = sorted(set(c['func'] for c in changes if c['func']))
    func_vocab = {f: i for i, f in enumerate(funcs)}

    dim = len(entity_vocab) + len(file_vocab) + len(func_vocab) + 4
    features = np.zeros((n, dim))

    # IDF
    doc_freq = Counter()
    for c in changes:
        for e in set(c['entities']) & set(entity_vocab.keys()):
            doc_freq[e] += 1

    for i, c in enumerate(changes):
        # TF-IDF
        for e in c['entities']:
            if e in entity_vocab:
                tf = 1.0
                idf = np.log(n / (doc_freq.get(e, 1) + 1)) + 1
                features[i, entity_vocab[e]] = tf * idf

        # one-hot
        offset = len(entity_vocab)
        if c['file'] in file_vocab:
            features[i, offset + file_vocab[c['file']]] = 1.0

        # one-hot
        offset2 = offset + len(file_vocab)
        if c['func'] and c['func'] in func_vocab:
            features[i, offset2 + func_vocab[c['func']]] = 1.0

        features[i, -4] = len(c['lines']) / 50.0
        features[i, -3] = len(c['entities']) / 20.0
        features[i, -2] = len(c['ast_entities']) / 10.0
        features[i, -1] = len(c['func_calls']) / 10.0

    # L2
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    features = features / norms

    return features


def compute_edge_scores(changes: List[Dict],
                        node_features: np.ndarray,
                        entity_graph: EntityGraph) -> np.ndarray:
    """
     CCRG
    Edge features:
      - Jaccard
      - AST Jaccard
      - Jaccard
      -
      -
      -
      - Entity graph
    """
    n = len(changes)
    scores = np.zeros((n, n))

    # entity graph
    ref_pairs = set()
    for ref in entity_graph.references:
        ref_pairs.add((ref.src, ref.dst))
        ref_pairs.add((ref.dst, ref.src))

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = changes[i], changes[j]

            # Jaccard
            entity_sim = _jaccard(ci['entities'], cj['entities'])

            # AST Jaccard
            ast_entity_sim = _jaccard(ci['ast_entities'], cj['ast_entities'])

            # Jaccard
            call_sim = _jaccard(ci['func_calls'], cj['func_calls'])

            same_file = 1.0 if ci['file'] == cj['file'] else 0.0

            same_func = 1.0 if (ci['func'] and ci['func'] == cj['func']) else 0.0

            cos_sim = 0.0
            if node_features.shape[1] > 0:
                cos_sim = float(np.dot(node_features[i], node_features[j]))

            # Entity graph
            ref_score = 0.0
            shared_refs = ci['ast_entities'] & cj['ast_entities']
            if shared_refs:
                ref_score = min(len(shared_refs) / 3.0, 1.0)

            score = (
                0.20 * entity_sim +
                0.15 * ast_entity_sim +
                0.15 * call_sim +
                0.10 * same_file +
                0.15 * same_func +
                0.15 * cos_sim +
                0.10 * ref_score
            )

            scores[i, j] = score
            scores[j, i] = score

    return scores


def _jaccard(set_a: Set, set_b: Set) -> float:
    """Jaccard """
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def cluster_from_scores(scores: np.ndarray, num_groups: int) -> List[int]:
    """
     pairwise cluster
     → connected components → num_groups
    """
    n = scores.shape[0]
    if n <= num_groups:
        return list(range(n))

    lo, hi = 0.0, 1.0
    best_result = list(range(n))
    best_diff = n

    for _ in range(25):
        threshold = (lo + hi) / 2
        clusters = _connected_components(scores, threshold)
        n_clusters = len(set(clusters))

        diff = abs(n_clusters - num_groups)
        if diff < best_diff:
            best_diff = diff
            best_result = clusters

        if n_clusters == num_groups:
            break
        elif n_clusters > num_groups:
            hi = threshold
        else:
            lo = threshold

    n_clusters = len(set(best_result))
    if n_clusters > num_groups:
        best_result = _force_merge(best_result, scores, num_groups)
    elif n_clusters < num_groups:
        best_result = _force_split(best_result, scores, num_groups)

    return best_result


def _connected_components(scores: np.ndarray, threshold: float) -> List[int]:
    """Union-Find """
    n = scores.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if scores[i, j] >= threshold:
                union(i, j)

    root_map = {}
    result = []
    for i in range(n):
        root = find(i)
        if root not in root_map:
            root_map[root] = len(root_map)
        result.append(root_map[root])
    return result


def _force_merge(clusters: List[int], scores: np.ndarray, target: int) -> List[int]:
    """ cluster """
    cluster_members = defaultdict(list)
    for i, c in enumerate(clusters):
        cluster_members[c].append(i)

    active = list(cluster_members.keys())
    while len(active) > target:
        best_sim = -1
        best_pair = (0, 1)
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                ci, cj = active[ii], active[jj]
                avg_sim = np.mean([scores[a, b]
                                   for a in cluster_members[ci]
                                   for b in cluster_members[cj]])
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_pair = (ii, jj)
        ii, jj = best_pair
        ci, cj = active[ii], active[jj]
        cluster_members[ci].extend(cluster_members[cj])
        del cluster_members[cj]
        active.pop(jj)

    result = [0] * len(clusters)
    for gid, key in enumerate(active):
        for idx in cluster_members[key]:
            result[idx] = gid
    return result


def _force_split(clusters: List[int], scores: np.ndarray, target: int) -> List[int]:
    """ cluster """
    result = list(clusters)
    next_id = max(result) + 1

    while len(set(result)) < target:
        counter = Counter(result)
        largest = counter.most_common(1)[0][0]
        members = [i for i, c in enumerate(result) if c == largest]
        if len(members) < 2:
            break
        mid = len(members) // 2
        for idx in members[mid:]:
            result[idx] = next_id
        next_id += 1

    return result


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# Linux bare repo
_LINUX_REPO_PATH = None


def set_repo_path(path):
    """ Linux bare repo full-project entity graph"""
    global _LINUX_REPO_PATH
    from pathlib import Path
    _LINUX_REPO_PATH = Path(path) if path else None


def predict_hdgnn_faithful(entry: Dict) -> List[Optional[int]]:
    """
    HD-GNN faithful

    1. diff git repo
    2. entity reference graph
    3. code changes (hunk )
    4.
    5. group
    6. hunk-level → line-level
    """
    diff_text = entry['source']['diff']
    num_groups = entry['ground_truth']['num_groups']

    # full-project
    if _LINUX_REPO_PATH and _LINUX_REPO_PATH.exists():
        resolved = resolve_entry(entry, repo_path=_LINUX_REPO_PATH)
        if resolved.resolved_from_git:
            file_sources = resolved.files
        else:
            file_sources = reconstruct_pre_post_from_diff(diff_text)
    else:
        file_sources = reconstruct_pre_post_from_diff(diff_text)

    # entity graph
    entity_graph = extract_entities(file_sources)

    # code changes
    changes = extract_code_changes_faithful(diff_text, file_sources, entity_graph)

    if not changes:
        n_lines = sum(1 for f in entry['ground_truth']['files']
                      for la in f['lines'] if la['type'] in ('add', 'del'))
        return [0] * n_lines

    node_features = compute_node_features(changes, entity_graph)
    edge_scores = compute_edge_scores(changes, node_features, entity_graph)

    if len(changes) <= num_groups:
        change_clusters = list(range(len(changes)))
    else:
        change_clusters = cluster_from_scores(edge_scores, num_groups)

    # hunk-level → line-level
    return _map_to_lines(diff_text, change_clusters)


def _map_to_lines(diff_text: str, change_clusters: List[int]) -> List[int]:
    """ hunk-level cluster line-level"""
    result = []
    change_idx = 0
    current_hunk_lines = 0
    in_hunk = False

    for line in diff_text.splitlines():
        if line.startswith('diff --git') or line.startswith('---') or line.startswith('+++'):
            continue

        if _HUNK_RE.match(line):
            if in_hunk and current_hunk_lines > 0:
                change_idx += 1
            current_hunk_lines = 0
            in_hunk = True
            continue

        if in_hunk:
            if (line.startswith('+') and not line.startswith('+++')) or \
               (line.startswith('-') and not line.startswith('---')):
                cid = change_clusters[min(change_idx, len(change_clusters) - 1)] \
                    if change_clusters else 0
                result.append(cid)
                current_hunk_lines += 1

    return result
