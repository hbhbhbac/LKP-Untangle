"""
herzig_faithful.py — Herzig (2016 ESE) Confidence Voter

 5 confidence voters:
  1. FileDistance:
  2. PackageDistance:
  3. CallGraph: AST-based
  4. ChangeCouplings:
  5. DataDependency: def-use PDG-based

 tree-sitter
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..ast_parser import ParsedFile, parse_c_file, map_line_to_function, is_c_file
from ..source_resolver import FileSource, reconstruct_pre_post_from_diff
from ..pdg_builder import PDG, build_pdg, get_neighbors

_HUNK_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@\s*(.*)')
_IDENT_RE = re.compile(r'\b([a-zA-Z_]\w{2,})\b')
_FUNC_CALL_RE = re.compile(r'\b([a-zA-Z_]\w+)\s*\(')
_STOP_WORDS = {
    'int', 'char', 'void', 'long', 'short', 'unsigned', 'signed',
    'float', 'double', 'struct', 'enum', 'union', 'typedef',
    'const', 'static', 'extern', 'volatile', 'register', 'inline',
    'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case',
    'break', 'continue', 'goto', 'default', 'sizeof', 'NULL', 'null',
    'true', 'false', 'bool', 'include', 'define', 'ifdef', 'ifndef',
    'endif', 'elif', 'undef', 'pragma',
}


def predict_herzig_faithful(entry: Dict) -> List[Optional[int]]:
    """
    Herzig faithful
    """
    diff_text = entry['source']['diff']
    num_groups = entry['ground_truth']['num_groups']

    file_sources = reconstruct_pre_post_from_diff(diff_text)
    parsed_files = {}
    pdg_files = {}
    call_graph = defaultdict(set)  # caller -> {callees}

    for fs in file_sources:
        source = fs.post_content or fs.pre_content
        if source and is_c_file(fs.filepath):
            parsed = parse_c_file(source)
            parsed_files[fs.filepath] = parsed
            pdg_files[fs.filepath] = build_pdg(parsed)
            for call in parsed.calls:
                if call.caller_function:
                    call_graph[call.caller_function].add(call.callee)

    ops = _parse_ops_with_ast(diff_text, parsed_files)

    if not ops:
        n_lines = sum(1 for f in entry['ground_truth']['files']
                      for la in f['lines'] if la['type'] in ('add', 'del'))
        return [0] * n_lines

    n = len(ops)

    # case
    if n > 300:
        return _fast_cluster_faithful(ops, num_groups, call_graph)

    conf_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            c = _compute_confidence_faithful(
                ops[i], ops[j], call_graph, pdg_files
            )
            conf_matrix[i][j] = c
            conf_matrix[j][i] = c

    # Max-linkage
    cluster_ids = _agglomerative_max_linkage(conf_matrix, num_groups)

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


def _parse_ops_with_ast(diff_text: str,
                        parsed_files: Dict[str, ParsedFile]) -> List[Dict]:
    """ diff AST """
    ops = []
    current_file = None
    current_hunk_func = None
    post_line = 0
    idx = 0

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

        if line.startswith('+') and not line.startswith('+++'):
            content = line[1:]
            idents = set(_IDENT_RE.findall(content)) - _STOP_WORDS
            calls = set(_FUNC_CALL_RE.findall(content)) - _STOP_WORDS

            # AST-based
            func = current_hunk_func
            if current_file in parsed_files:
                real_func = map_line_to_function(parsed_files[current_file], post_line)
                if real_func:
                    func = real_func

            ops.append({
                'file': current_file,
                'line_no': post_line,
                'func': func,
                'content': content,
                'identifiers': idents,
                'func_calls': calls,
                'idx': idx,
            })
            post_line += 1
            idx += 1
        elif line.startswith('-') and not line.startswith('---'):
            content = line[1:]
            idents = set(_IDENT_RE.findall(content)) - _STOP_WORDS
            calls = set(_FUNC_CALL_RE.findall(content)) - _STOP_WORDS

            ops.append({
                'file': current_file,
                'line_no': post_line,
                'func': current_hunk_func,
                'content': content,
                'identifiers': idents,
                'func_calls': calls,
                'idx': idx,
            })
            idx += 1
        else:
            post_line += 1

    return ops


def _compute_confidence_faithful(op1: Dict, op2: Dict,
                                 call_graph: Dict[str, Set[str]],
                                 pdg_files: Dict[str, PDG]) -> float:
    """
    5 voter
    """
    scores = []
    weights = []

    # Voter 1: FileDistance
    if op1['file'] == op2['file']:
        dist = abs(op1['line_no'] - op2['line_no'])
        fd = max(0.0, 1.0 - dist / 500.0)
        scores.append(fd)
        weights.append(0.25)

    # Voter 2: PackageDistance ()
    if op1['file'] != op2['file']:
        parts1 = op1['file'].split('/')
        parts2 = op2['file'].split('/')
        common = 0
        for a, b in zip(parts1, parts2):
            if a == b:
                common += 1
            else:
                break
        max_len = max(len(parts1), len(parts2))
        pd = common / max_len if max_len > 0 else 0.0
        scores.append(pd)
        weights.append(0.15)

    # Voter 3: CallGraph (AST-based)
    cg_score = _call_graph_score(op1, op2, call_graph)
    scores.append(cg_score)
    weights.append(0.25)

    # Voter 4: ChangeCouplings ()
    entity_sim = _entity_coupling(op1, op2)
    scores.append(entity_sim)
    weights.append(0.10)

    # Voter 5: DataDependency (PDG-based)
    dd_score = _data_dependency_faithful(op1, op2, pdg_files)
    scores.append(dd_score)
    weights.append(0.25)

    if not scores:
        return 0.0
    total_w = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_w


def _call_graph_score(op1: Dict, op2: Dict,
                      call_graph: Dict[str, Set[str]]) -> float:
    """
     Section 5.3: CallGraph voter
    Conf = 1 / (1 + shortest_path_distance)
     edge weight = 1/call_count weight=1
    """
    f1, f2 = op1['func'], op2['func']

    # : distance = 0
    if f1 and f1 == f2:
        return 1.0

    if not f1 or not f2:
        # fallback
        calls1 = op1.get('func_calls', set())
        calls2 = op2.get('func_calls', set())
        shared_calls = calls1 & calls2
        if shared_calls:
            return 0.3
        return 0.0

    # BFS (: caller→callee callee→caller)
    dist = _bfs_shortest_path(f1, f2, call_graph)
    if dist is None:
        calls1 = op1.get('func_calls', set())
        calls2 = op2.get('func_calls', set())
        shared_calls = calls1 & calls2
        if shared_calls:
            return 0.2
        return 0.0

    # : Conf = 1 / (1 + distance)
    return 1.0 / (1.0 + dist)


def _bfs_shortest_path(src: str, dst: str,
                       call_graph: Dict[str, Set[str]],
                       max_depth: int = 5) -> Optional[int]:
    """
    BFS
    """
    if src == dst:
        return 0

    reverse_graph = defaultdict(set)
    for caller, callees in call_graph.items():
        for callee in callees:
            reverse_graph[callee].add(caller)

    # BFS (: call_graph reverse_graph)
    visited = {src}
    queue = [(src, 0)]
    head = 0

    while head < len(queue):
        node, depth = queue[head]
        head += 1

        if depth >= max_depth:
            continue

        # (caller → callee)
        for neighbor in call_graph.get(node, set()):
            if neighbor == dst:
                return depth + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

        # (callee → caller)
        for neighbor in reverse_graph.get(node, set()):
            if neighbor == dst:
                return depth + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return None


def _entity_coupling(op1: Dict, op2: Dict) -> float:
    """ ChangeCouplings"""
    shared = op1['identifiers'] & op2['identifiers']
    if not shared:
        return 0.0
    union = op1['identifiers'] | op2['identifiers']
    return len(shared) / len(union) if union else 0.0


def _data_dependency_faithful(op1: Dict, op2: Dict,
                              pdg_files: Dict[str, PDG]) -> float:
    """
     PDG
     def-use
    """
    shared = op1['identifiers'] & op2['identifiers']
    if not shared:
        return 0.0

    base_score = len(shared) / max(len(op1['identifiers'] | op2['identifiers']), 1)

    # PDG data flow
    if op1['file'] == op2['file'] and op1['file'] in pdg_files:
        pdg = pdg_files[op1['file']]
        # data
        line1 = op1['line_no']
        line2 = op2['line_no']
        if line1 in pdg.line_to_node and line2 in pdg.line_to_node:
            nid1 = pdg.line_to_node[line1]
            nid2 = pdg.line_to_node[line2]
            for neighbor_id, kind in get_neighbors(pdg, nid1):
                if neighbor_id == nid2 and kind == 'data':
                    return min(base_score + 0.4, 1.0)
            for n1_neighbor, k1 in get_neighbors(pdg, nid1):
                if k1 == 'data':
                    for n2_neighbor, k2 in get_neighbors(pdg, n1_neighbor):
                        if n2_neighbor == nid2 and k2 == 'data':
                            return min(base_score + 0.2, 1.0)

    return base_score


def _agglomerative_max_linkage(conf_matrix: np.ndarray,
                               num_groups: int) -> List[int]:
    """Max-linkage num_groups"""
    n = conf_matrix.shape[0]
    if n <= num_groups:
        return list(range(n))

    # scipy
    if n > 50:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        dist = 1.0 - np.clip(conf_matrix, 0, 1)
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='complete')  # complete = max linkage
        labels = fcluster(Z, t=num_groups, criterion='maxclust')
        return [int(l - 1) for l in labels]

    active = list(range(n))
    membership = {i: {i} for i in range(n)}

    while len(active) > num_groups:
        best_conf = -1
        best_pair = (0, 1)
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                ci, cj = active[ii], active[jj]
                max_c = max(conf_matrix[a][b]
                            for a in membership[ci]
                            for b in membership[cj])
                if max_c > best_conf:
                    best_conf = max_c
                    best_pair = (ii, jj)

        ii, jj = best_pair
        ci, cj = active[ii], active[jj]
        membership[ci] = membership[ci] | membership[cj]
        del membership[cj]
        active.pop(jj)

    result = [0] * n
    for gid, cluster_key in enumerate(active):
        for op_idx in membership[cluster_key]:
            result[op_idx] = gid
    return result


def _fast_cluster_faithful(ops: List[Dict], num_groups: int,
                           call_graph: Dict[str, Set[str]]) -> List[int]:
    """ case + """
    # (file, func)
    groups = defaultdict(list)
    for i, op in enumerate(ops):
        key = (op['file'], op['func'] or '')
        groups[key].append(i)

    group_keys = list(groups.keys())

    if len(group_keys) <= num_groups:
        result = [0] * len(ops)
        for gid, key in enumerate(group_keys):
            for idx in groups[key]:
                result[idx] = gid
        return result

    cluster_members = [groups[k] for k in group_keys]
    cluster_funcs = [k[1] for k in group_keys]

    while len(cluster_members) > num_groups:
        best_sim = -1
        best_pair = (0, 1)
        for i in range(len(cluster_members)):
            for j in range(i + 1, len(cluster_members)):
                sim = 0.0
                fi, fj = cluster_funcs[i], cluster_funcs[j]
                if group_keys[i][0] == group_keys[j][0]:
                    sim += 0.3
                if fi and fj:
                    if fj in call_graph.get(fi, set()) or fi in call_graph.get(fj, set()):
                        sim += 0.5
                ids_i = set()
                ids_j = set()
                for idx in cluster_members[i][:5]:
                    ids_i.update(ops[idx]['identifiers'])
                for idx in cluster_members[j][:5]:
                    ids_j.update(ops[idx]['identifiers'])
                shared = ids_i & ids_j
                if shared:
                    sim += 0.2 * min(len(shared) / 5.0, 1.0)

                if sim > best_sim:
                    best_sim = sim
                    best_pair = (i, j)

        i, j = best_pair
        cluster_members[i] = cluster_members[i] + cluster_members[j]
        cluster_members.pop(j)
        cluster_funcs.pop(j)
        group_keys.pop(j)

    result = [0] * len(ops)
    for gid, members in enumerate(cluster_members):
        for idx in members:
            result[idx] = gid
    return result
