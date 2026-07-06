"""
smartcommit_faithful.py — SmartCommit (2021 FSE)

 Section 3
  1. Preprocessing: diff diff hunks
  2. Graph Construction: Diff Hunk Graph 4
     - Hard Links: (def-use, call dependency)weight=1.0 / 1/hop
     - Soft Links: / + proximity
     - Refactoring Links: /
     - Cosmetic Links: /
  3. Edge-Shrinking: Kruskal-like (threshold=0.6)

: Bo Shen et al., "SmartCommit: A Graph-Based Interactive Assistant for
Activity-Oriented Commits", ESEC/FSE 2021.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ..ast_parser import ParsedFile, parse_c_file, parse_file, map_line_to_function, is_c_file
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
    'endif', 'elif', 'undef', 'pragma', 'error', 'warning',
}

# SmartCommit ( Section 3.5.1: 0.6)
DEFAULT_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Diff Hunk
# ---------------------------------------------------------------------------

class DiffHunk:
    """diff hunk: SmartCommit """
    __slots__ = ('file_index', 'index', 'filepath', 'change_type',
                 'added_lines', 'removed_lines', 'post_start', 'pre_start',
                 'identifiers', 'calls', 'defines', 'uses',
                 'function', 'hunk_header_func', 'is_cosmetic')

    def __init__(self, file_index: int, index: int, filepath: str):
        self.file_index = file_index
        self.index = index
        self.filepath = filepath
        self.change_type = 'modify'  # 'add', 'del', 'modify'
        self.added_lines: List[str] = []
        self.removed_lines: List[str] = []
        self.post_start = 0
        self.pre_start = 0
        self.identifiers: Set[str] = set()
        self.calls: Set[str] = set()
        self.defines: Set[str] = set()
        self.uses: Set[str] = set()
        self.function: Optional[str] = None
        self.hunk_header_func: Optional[str] = None
        self.is_cosmetic = False

    @property
    def uid(self) -> str:
        return f"{self.file_index}:{self.index}"

    @property
    def all_content(self) -> str:
        return '\n'.join(self.added_lines + self.removed_lines)


class DiffHunkEdge:
    """Diff Hunk Graph """
    __slots__ = ('src', 'dst', 'edge_type', 'weight')

    def __init__(self, src: int, dst: int, edge_type: str, weight: float):
        self.src = src
        self.dst = dst
        self.edge_type = edge_type  # 'depend', 'similar', 'refactor', 'cosmetic'
        self.weight = weight


# ---------------------------------------------------------------------------
# 1. Preprocessing: diff diff hunks
# ---------------------------------------------------------------------------

def _parse_diff_hunks(diff_text: str,
                      parsed_files: Dict[str, ParsedFile]) -> List[DiffHunk]:
    """ diff diff hunks AST """
    hunks = []
    current_file = None
    file_index = -1
    hunk_index = 0
    current_hunk: Optional[DiffHunk] = None
    post_line = 0
    pre_line = 0

    for line in diff_text.splitlines():
        if line.startswith('diff --git'):
            if current_hunk and (current_hunk.added_lines or current_hunk.removed_lines):
                _finalize_hunk(current_hunk, parsed_files)
                hunks.append(current_hunk)
            parts = line.split(' b/')
            current_file = parts[-1].strip() if len(parts) >= 2 else ''
            file_index += 1
            hunk_index = 0
            current_hunk = None
            continue
        if line.startswith('+++ b/'):
            current_file = line[6:].strip()
            continue
        if line.startswith('--- '):
            continue

        m = _HUNK_RE.match(line)
        if m:
            if current_hunk and (current_hunk.added_lines or current_hunk.removed_lines):
                _finalize_hunk(current_hunk, parsed_files)
                hunks.append(current_hunk)
            pre_line = int(m.group(1))
            post_line = int(m.group(3))
            current_hunk = DiffHunk(file_index, hunk_index, current_file or '')
            current_hunk.pre_start = pre_line
            current_hunk.post_start = post_line
            current_hunk.hunk_header_func = m.group(5).strip() if m.group(5) else None
            hunk_index += 1
            continue

        if current_hunk is not None:
            if line.startswith('+') and not line.startswith('+++'):
                content = line[1:]
                current_hunk.added_lines.append(content)
                _extract_line_info(current_hunk, content)
                # AST-based function detection
                if current_file in parsed_files and not current_hunk.function:
                    func = map_line_to_function(parsed_files[current_file], post_line)
                    if func:
                        current_hunk.function = func
                post_line += 1
            elif line.startswith('-') and not line.startswith('---'):
                content = line[1:]
                current_hunk.removed_lines.append(content)
                _extract_line_info(current_hunk, content)
                pre_line += 1
            else:
                pre_line += 1
                post_line += 1

    if current_hunk and (current_hunk.added_lines or current_hunk.removed_lines):
        _finalize_hunk(current_hunk, parsed_files)
        hunks.append(current_hunk)

    return hunks


def _extract_line_info(hunk: DiffHunk, content: str):
    """"""
    idents = set(_IDENT_RE.findall(content)) - _STOP_WORDS
    hunk.identifiers.update(idents)
    hunk.uses.update(idents)

    for m in _FUNC_CALL_RE.finditer(content):
        name = m.group(1)
        if name not in _STOP_WORDS:
            hunk.calls.add(name)

    # ()
    if '=' in content and not content.strip().startswith('=='):
        parts = content.split('=', 1)
        left = parts[0].strip()
        left_idents = _IDENT_RE.findall(left)
        if left_idents:
            hunk.defines.add(left_idents[-1])


def _finalize_hunk(hunk: DiffHunk, parsed_files: Dict[str, ParsedFile]):
    """ hunk: change_type cosmetic """
    if hunk.added_lines and not hunk.removed_lines:
        hunk.change_type = 'add'
    elif hunk.removed_lines and not hunk.added_lines:
        hunk.change_type = 'del'
    else:
        hunk.change_type = 'modify'

    # AST function hunk header
    if not hunk.function and hunk.hunk_header_func:
        func_m = _FUNC_CALL_RE.search(hunk.hunk_header_func)
        if func_m and func_m.group(1) not in _STOP_WORDS:
            hunk.function = func_m.group(1)

    # Cosmetic detection: /
    hunk.is_cosmetic = _is_cosmetic_change(hunk)


def _is_cosmetic_change(hunk: DiffHunk) -> bool:
    """ cosmetic change (/)"""
    if not hunk.added_lines or not hunk.removed_lines:
        return False

    stripped_add = [l.strip() for l in hunk.added_lines if l.strip()]
    stripped_del = [l.strip() for l in hunk.removed_lines if l.strip()]

    if stripped_add == stripped_del:
        return True

    non_comment_add = [l for l in stripped_add
                       if not l.startswith('//') and not l.startswith('/*') and not l.startswith('*')]
    non_comment_del = [l for l in stripped_del
                       if not l.startswith('//') and not l.startswith('/*') and not l.startswith('*')]
    if non_comment_add == non_comment_del:
        return True

    return False


# ---------------------------------------------------------------------------
# 2. Graph Construction: Diff Hunk Graph
# ---------------------------------------------------------------------------

def _build_diff_hunk_graph(hunks: List[DiffHunk],
                            pdg_files: Dict[str, PDG]) -> List[DiffHunkEdge]:
    """
     Diff Hunk Graph

     Section 3.4 4 :
    1. Hard Links (depend): /, weight=1.0 1/hop
    2. Soft Links (similar): + proximity
    3. Refactoring Links (refactor):
    4. Cosmetic Links (cosmetic): /
    """
    edges = []
    n = len(hunks)

    # 1. Hard Links:
    edges.extend(_compute_hard_links(hunks, pdg_files))

    # 2. Soft Links: +
    edges.extend(_compute_soft_links(hunks))

    # 3. Refactoring Links:
    edges.extend(_compute_refactoring_links(hunks))

    # 4. Cosmetic Links: cosmetic
    edges.extend(_compute_cosmetic_links(hunks))

    return edges


def _compute_hard_links(hunks: List[DiffHunk],
                        pdg_files: Dict[str, PDG]) -> List[DiffHunkEdge]:
    """
    Hard Links ( Section 3.4):
    - Direct dependency (def-use): hunk A defines var x, hunk B uses (not defines) x → weight=1.0
    - Call dependency: hunk A modifies func f definition, hunk B calls f → weight=1.0
    - Same function proximity: same enclosing function → weight=0.8
    - Indirect dependency: different functions, shared identifiers → weight=1/hop (0.5)

    : A B
     struct field
    """
    edges = []
    n = len(hunks)

    for i in range(n):
        for j in range(i + 1, n):
            hi, hj = hunks[i], hunks[j]

            # Direct def-use: A defines var x, B uses x but does NOT also define x
            # ( struct field name def-use)
            i_defines_j_uses = hi.defines & (hj.uses - hj.defines)
            j_defines_i_uses = hj.defines & (hi.uses - hi.defines)

            if i_defines_j_uses or j_defines_i_uses:
                edges.append(DiffHunkEdge(i, j, 'depend', 1.0))
                continue

            # Call dependency: hunk A f hunk B f
            if hi.function and hi.function in hj.calls:
                edges.append(DiffHunkEdge(i, j, 'depend', 1.0))
                continue
            if hj.function and hj.function in hi.calls:
                edges.append(DiffHunkEdge(i, j, 'depend', 1.0))
                continue

            # Same function: enclosing function → proximity
            if (hi.filepath == hj.filepath and
                    hi.function and hj.function and hi.function == hj.function):
                edges.append(DiffHunkEdge(i, j, 'depend', 0.8))
                continue

            # Indirect dependency:
            if hi.filepath == hj.filepath and hi.function and hj.function:
                shared_ids = hi.identifiers & hj.identifiers
                total_ids = len(hi.identifiers | hj.identifiers)
                if total_ids > 0 and len(shared_ids) >= 3:
                    # weight = Jaccard 0.3-0.5
                    jaccard = len(shared_ids) / total_ids
                    w = min(0.5, 0.3 + jaccard * 0.4)
                    edges.append(DiffHunkEdge(i, j, 'depend', w))

    return edges


def _compute_soft_links(hunks: List[DiffHunk]) -> List[DiffHunkEdge]:
    """
    Soft Links ( Section 3.4):
    - Similarity: (textual_sim + structural_sim) / 2
    - Proximity: 1 / max_hop_to_common_ancestor (/)
    """
    edges = []
    n = len(hunks)

    for i in range(n):
        for j in range(i + 1, n):
            hi, hj = hunks[i], hunks[j]

            # hard link
            # (Jaccard of identifiers)
            if not hi.identifiers or not hj.identifiers:
                continue

            intersection = hi.identifiers & hj.identifiers
            union_size = len(hi.identifiers | hj.identifiers)
            if union_size == 0:
                continue

            text_sim = len(intersection) / union_size

            # Proximity:
            prox = 0.0
            if hi.filepath == hj.filepath:
                if hi.function and hi.function == hj.function:
                    prox = 0.9 #
                elif hi.function and hj.function:
                    prox = 0.3 #
                else:
                    prox = 0.2 #
            # : prox = 0

            # Combined similarity (: average of textual + structural)
            combined = (text_sim + prox) / 2.0

            if combined > 0.2: #
                edges.append(DiffHunkEdge(i, j, 'similar', combined))

    return edges


def _compute_refactoring_links(hunks: List[DiffHunk]) -> List[DiffHunkEdge]:
    """
    Refactoring Links ( Section 3.4):
    rename refactoring:
     hunk " → "
     rename refactoring

    : old→new
    """
    edges = []
    n = len(hunks)

    # modify hunk rename patterns: (old_name, new_name)
    hunk_renames = []
    for h in hunks:
        renames = set()
        if h.added_lines and h.removed_lines:
            added_idents = set()
            removed_idents = set()
            for line in h.added_lines:
                added_idents.update(set(_IDENT_RE.findall(line)) - _STOP_WORDS)
            for line in h.removed_lines:
                removed_idents.update(set(_IDENT_RE.findall(line)) - _STOP_WORDS)

            # Rename pattern:
            old_only = removed_idents - added_idents
            new_only = added_idents - removed_idents

            # old new rename
            if 1 <= len(old_only) <= 2 and 1 <= len(new_only) <= 2:
                for old_name in old_only:
                    for new_name in new_only:
                        renames.add((old_name, new_name))
        hunk_renames.append(renames)

    # rename pattern hunks
    for i in range(n):
        if not hunk_renames[i]:
            continue
        for j in range(i + 1, n):
            if not hunk_renames[j]:
                continue
            shared_renames = hunk_renames[i] & hunk_renames[j]
            if shared_renames:
                edges.append(DiffHunkEdge(i, j, 'refactor', 1.0))

    return edges


def _compute_cosmetic_links(hunks: List[DiffHunk]) -> List[DiffHunkEdge]:
    """
    Cosmetic Links ( Section 3.4):
     cosmetic ()
    Weight:
    """
    edges = []
    cosmetic_indices = [i for i, h in enumerate(hunks) if h.is_cosmetic]

    for idx in range(len(cosmetic_indices)):
        for jdx in range(idx + 1, len(cosmetic_indices)):
            i, j = cosmetic_indices[idx], cosmetic_indices[jdx]
            # cosmetic
            w = 0.85 if hunks[i].filepath == hunks[j].filepath else 0.4
            edges.append(DiffHunkEdge(i, j, 'cosmetic', w))

    return edges


# ---------------------------------------------------------------------------
# 3. Edge-Shrinking Algorithm ( Algorithm 1)
# ---------------------------------------------------------------------------

def _edge_shrinking(n_hunks: int, edges: List[DiffHunkEdge],
                    threshold: float = DEFAULT_THRESHOLD) -> List[int]:
    """
    Edge-Shrinking Algorithm ( Algorithm 1):

    1. : hunk group (disjoint-set)
    2. (weight desc, src asc, dst asc)
    3. :
       - group → skip
       - weight >= threshold → merge (union)
       - weight < threshold → break
    4. hunk group id
    """
    # Union-Find
    parent = list(range(n_hunks))
    rank = [0] * n_hunks

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    # weight (: )
    best_weight = {}
    for edge in edges:
        key = (min(edge.src, edge.dst), max(edge.src, edge.dst))
        if key not in best_weight or edge.weight > best_weight[key]:
            best_weight[key] = edge.weight

    # : weight , src , dst
    sorted_edges = sorted(best_weight.items(),
                          key=lambda x: (-x[1], x[0][0], x[0][1]))

    for (src, dst), weight in sorted_edges:
        if weight < threshold:
            break
        if find(src) != find(dst):
            union(src, dst)

    # group id
    root_to_id = {}
    result = []
    for i in range(n_hunks):
        root = find(i)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        result.append(root_to_id[root])

    return result


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def predict_smartcommit_faithful(entry: Dict,
                                 threshold: float = DEFAULT_THRESHOLD) -> List[Optional[int]]:
    """
    SmartCommit faithful

     (Bo Shen et al. 2021 FSE "SmartCommit"):
    1. Preprocessing: diff diff hunks
    2. Graph Construction: Diff Hunk Graph (4)
    3. Edge-Shrinking: Kruskal-like (threshold=0.6)

    : group id
    """
    diff_text = entry['source']['diff']

    file_sources = reconstruct_pre_post_from_diff(diff_text)
    parsed_files = {}
    pdg_files = {}

    for fs in file_sources:
        source = fs.post_content or fs.pre_content
        if source:
            if is_c_file(fs.filepath):
                parsed = parse_c_file(source)
            else:
                parsed = parse_file(source, fs.filepath)
            parsed_files[fs.filepath] = parsed
            pdg_files[fs.filepath] = build_pdg(parsed)

    # 1. Preprocessing: diff hunks
    hunks = _parse_diff_hunks(diff_text, parsed_files)

    if not hunks:
        n_lines = sum(1 for f in entry['ground_truth']['files']
                      for la in f['lines'] if la['type'] in ('add', 'del'))
        return [0] * n_lines

    # 2. Graph Construction
    edges = _build_diff_hunk_graph(hunks, pdg_files)

    # 3. Edge-Shrinking
    hunk_groups = _edge_shrinking(len(hunks), edges, threshold)

    # hunk-level groups → line-level groups
    return _map_hunks_to_lines(hunks, hunk_groups, entry, diff_text)


def _map_hunks_to_lines(hunks: List[DiffHunk], hunk_groups: List[int],
                         entry: Dict, diff_text: str) -> List[int]:
    """
     hunk
     hunk group id
    """
    # diff hunk
    line_to_hunk = {}
    current_file = None
    file_index = -1
    hunk_index = 0
    line_idx = 0

    for line in diff_text.splitlines():
        if line.startswith('diff --git'):
            parts = line.split(' b/')
            current_file = parts[-1].strip() if len(parts) >= 2 else ''
            file_index += 1
            hunk_index = 0
            continue
        if line.startswith('+++ b/'):
            current_file = line[6:].strip()
            continue
        if line.startswith('--- '):
            continue

        m = _HUNK_RE.match(line)
        if m:
            hunk_index += 1 # diff hunk +1
            continue

        if line.startswith('+') and not line.startswith('+++'):
            line_to_hunk[line_idx] = (file_index, hunk_index - 1)
            line_idx += 1
        elif line.startswith('-') and not line.startswith('---'):
            line_to_hunk[line_idx] = (file_index, hunk_index - 1)
            line_idx += 1

    # (file_index, hunk_local_index) -> hunk list index
    hunk_uid_to_idx = {}
    for i, h in enumerate(hunks):
        hunk_uid_to_idx[(h.file_index, h.index)] = i

    result = []
    n_lines = sum(1 for f in entry['ground_truth']['files']
                  for la in f['lines'] if la['type'] in ('add', 'del'))

    for idx in range(n_lines):
        if idx in line_to_hunk:
            fi, hi = line_to_hunk[idx]
            hunk_list_idx = hunk_uid_to_idx.get((fi, hi))
            if hunk_list_idx is not None:
                result.append(hunk_groups[hunk_list_idx])
            else:
                result.append(0)
        else:
            result.append(0)

    return result
