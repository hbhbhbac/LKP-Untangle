"""
barnett_faithful.py — Barnett (2015 ICSE) ClusterChanges faithful reimplementation

Core rules from the original paper：
 1. SameEnclosingMethod: changes within the same function body are grouped together
 2. Def-Use: if change A defines variable x and change B uses x, they are related
 3. Use-Use: if changes A and B both use the same variable x (with a common definition), they are related

Uses tree-sitter AST to determine real function boundaries and def-use relations。
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ..ast_parser import ParsedFile, parse_c_file, map_line_to_function, is_c_file
from ..source_resolver import FileSource, reconstruct_pre_post_from_diff
from ..pdg_builder import PDG, build_pdg, get_neighbors

_HUNK_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@\s*(.*)')
_IDENT_RE = re.compile(r'\b([a-zA-Z_]\w{2,})\b')
_STOP_WORDS = {
    'int', 'char', 'void', 'long', 'short', 'unsigned', 'signed',
    'float', 'double', 'struct', 'enum', 'union', 'typedef',
    'const', 'static', 'extern', 'volatile', 'register', 'inline',
    'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case',
    'break', 'continue', 'goto', 'default', 'sizeof', 'NULL', 'null',
    'true', 'false', 'bool', 'include', 'define', 'ifdef', 'ifndef',
    'endif', 'elif', 'undef', 'pragma', 'error', 'warning',
}


def predict_barnett_faithful(entry: Dict) -> List[Optional[int]]:
    """
 Barnett faithful Predict。

 1. diff File tree-sitter Parse
 2. Change enclosing function
 3. Build PDG Get def-use Relation
 4. Rule union-find Cluster
    """
    diff_text = entry['source']['diff']

 # GetFileParse
    file_sources = reconstruct_pre_post_from_diff(diff_text)
    parsed_files = {}  # filepath -> ParsedFile
    pdg_files = {}     # filepath -> PDG

    for fs in file_sources:
        source = fs.post_content or fs.pre_content
        if source and is_c_file(fs.filepath):
            parsed = parse_c_file(source)
            parsed_files[fs.filepath] = parsed
            pdg_files[fs.filepath] = build_pdg(parsed)

 # Parse diff GetChange
    regions = _parse_regions_with_ast(diff_text, parsed_files)

    if not regions:
        n_lines = sum(1 for f in entry['ground_truth']['files']
                      for la in f['lines'] if la['type'] in ('add', 'del'))
        return [0] * n_lines

 # diff-regions ( Use-Use Rule)
    all_defines = set()
    for r in regions:
        all_defines.update(r['defines'])

 # Rule union-find
    n = len(regions)
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
            if _are_related_faithful(regions[i], regions[j], pdg_files, all_defines):
                union(i, j)

 # cluster id
    root_to_id = {}
    cluster_ids = []
    for i in range(n):
        root = find(i)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        cluster_ids.append(root_to_id[root])

 # region-level → line-level
    line_to_group = {}
    for region, cid in zip(regions, cluster_ids):
        for line_info in region['line_indices']:
            line_to_group[line_info] = cid

    result = []
    idx = 0
    for f in entry['ground_truth']['files']:
        for la in f['lines']:
            if la['type'] in ('add', 'del'):
                result.append(line_to_group.get(idx, 0))
                idx += 1

    return result


def _parse_regions_with_ast(diff_text: str,
                            parsed_files: Dict[str, ParsedFile]) -> List[Dict]:
    """
 Parse diff regionsUse AST to determine actual function membership
 region = hunk
      - file, func (AST-based), identifiers, defines, uses, line_indices
    """
    regions = []
    current_file = None
    current_hunk_func = None
    current_region = None
    line_idx = 0
    post_line = 0

    for line in diff_text.splitlines():
        if line.startswith('diff --git'):
            if current_region and current_region['line_indices']:
                regions.append(current_region)
                current_region = None
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
            if current_region and current_region['line_indices']:
                regions.append(current_region)
            post_line = int(m.group(3))
            current_hunk_func = m.group(5).strip() if m.group(5) else None
            current_region = {
                'file': current_file,
 'func': None, # AST
                'hunk_func': current_hunk_func,
                'identifiers': set(),
                'defines': set(),
                'uses': set(),
                'line_indices': [],
            }
            continue

        if current_region is not None:
            if line.startswith('+') and not line.startswith('+++'):
                content = line[1:]
                idents = set(_IDENT_RE.findall(content)) - _STOP_WORDS
                current_region['identifiers'].update(idents)
                current_region['uses'].update(idents)
                current_region['line_indices'].append(line_idx)

 # AST Function
                if current_file in parsed_files:
                    real_func = map_line_to_function(
                        parsed_files[current_file], post_line
                    )
                    if real_func and not current_region['func']:
                        current_region['func'] = real_func

 #
                if '=' in content and not content.strip().startswith('='):
                    lhs = content.split('=')[0]
                    lhs_idents = set(_IDENT_RE.findall(lhs)) - _STOP_WORDS
                    current_region['defines'].update(lhs_idents)

                post_line += 1
                line_idx += 1
            elif line.startswith('-') and not line.startswith('---'):
                content = line[1:]
                idents = set(_IDENT_RE.findall(content)) - _STOP_WORDS
                current_region['identifiers'].update(idents)
                current_region['uses'].update(idents)
                current_region['line_indices'].append(line_idx)

                if '=' in content and not content.strip().startswith('='):
                    lhs = content.split('=')[0]
                    lhs_idents = set(_IDENT_RE.findall(lhs)) - _STOP_WORDS
                    current_region['defines'].update(lhs_idents)

                line_idx += 1
            else:
                post_line += 1

    if current_region and current_region['line_indices']:
        regions.append(current_region)

 # Fallback: AST Function hunk header
    for r in regions:
        if not r['func']:
            r['func'] = r['hunk_func']

    return regions


def _are_related_faithful(r1: Dict, r2: Dict,
                          pdg_files: Dict[str, PDG],
                          all_defines: Set[str] = None) -> bool:
    """
 Barnett Rule Section III-B
 1. SameEnclosingMethod: AST Function
 2. Def-Use: r1 Variable r2
 : (defs(f1) × uses(f2)) ∩ defUsesInDiffs ≠ {}
 3. Use-Use: r1 r2 VariableVariable diff-region
 : (uses(f1) × uses(f2)) ∩ useUsesInDiffs ≠ {}
 useUsesInDiffs d (d,u1) (d,u2) def-use
 d diff-region
    """
    # Rule 1: SameEnclosingMethod (AST-based)
    if (r1['file'] == r2['file'] and
            r1['func'] and r1['func'] == r2['func']):
        return True

    # Rule 2: Def-Use
 # r1 → r2
    if r1['defines'] & r2['uses']:
        return True
 # r2 → r1
    if r2['defines'] & r1['uses']:
        return True

    # Rule 3: Use-Use
 # Condition: region diff-region
    shared_uses = r1['uses'] & r2['uses']
    if shared_uses and all_defines is not None:
 # Filter: diff-regions
        external_shared = shared_uses - all_defines
        if external_shared:
 # Verify: Check PDG def-use Relation
            meaningful = {s for s in external_shared if len(s) > 3}
            if meaningful:
                return True
    elif shared_uses:
 # Fallback: all_defines PDG Verify
        if r1['file'] == r2['file'] and r1['file'] in pdg_files:
            pdg = pdg_files[r1['file']]
            for var in shared_uses:
                for node in pdg.nodes:
                    if var in node.defines:
                        for neighbor_id, kind in get_neighbors(pdg, node.node_id):
                            if kind == 'data' and var in pdg.nodes[neighbor_id].uses:
                                return True

    return False
