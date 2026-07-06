"""
delta_pdg.py — Delta-PDG

  1. Full-project mode: PDG
  2. Diff-only mode (fallback): diff PDG

Full-project mode control/data
 WL kernel
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .ast_parser import ParsedFile, parse_c_file, parse_file, map_line_to_function
from .pdg_builder import PDG, PDGNode, PDGEdge, build_pdg, get_neighbors
from .source_resolver import FileSource, ResolvedEntry, reconstruct_pre_post_from_diff

_HUNK_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
_FUNC_NAME_RE = re.compile(r'\b([a-zA-Z_]\w+)\s*\(')


def _extract_func_name_from_hunk(hunk_func: Optional[str]) -> Optional[str]:
    """Extract just the function name from a hunk header context string."""
    if not hunk_func:
        return None
    # Hunk headers look like: "static int my_func(args...)" or "void do_thing(void)"
    m = _FUNC_NAME_RE.search(hunk_func)
    if m:
        name = m.group(1)
        # Skip common type keywords that might precede the real function name
        if name in ('static', 'inline', 'void', 'int', 'char', 'long', 'unsigned',
                    'signed', 'struct', 'enum', 'const', 'extern'):
            # Try to find the next match
            for m2 in _FUNC_NAME_RE.finditer(hunk_func):
                if m2.group(1) not in ('static', 'inline', 'void', 'int', 'char',
                                       'long', 'unsigned', 'signed', 'struct',
                                       'enum', 'const', 'extern'):
                    return m2.group(1)
        return name
    return None


@dataclass
class DeltaNode:
    """Delta-PDG """
    node_id: int
    line: int
    change_type: str  # 'add' or 'del'
    function: Optional[str]
    content: str
    identifiers: Set[str] = field(default_factory=set)
    # WL label ( = change_type:function)
    label: str = ''


@dataclass
class DeltaEdge:
    """Delta-PDG """
    src: int
    dst: int
    kind: str  # 'control', 'data', 'seq'
    # NFG: (Name Flow Graph annotation)
    flow_names: Set[str] = field(default_factory=set)


@dataclass
class DeltaPDG:
    """Delta-PDG: """
    nodes: List[DeltaNode] = field(default_factory=list)
    edges: List[DeltaEdge] = field(default_factory=list)


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


def build_delta_pdg(diff_text: str, file_sources: List[FileSource]) -> DeltaPDG:
    """
     delta-PDG

    1. diff pre/post
    2. pre/post PDG
    3. PDG
    4. diff
    """
    delta = DeltaPDG()

    # diff
    changed_lines = _parse_diff_changes(diff_text)
    if not changed_lines:
        return delta

    # PDG
    file_pdgs = {}  # filepath -> (pre_pdg, post_pdg)
    source_map = {fs.filepath: fs for fs in file_sources}

    for filepath in set(cl['file'] for cl in changed_lines):
        fs = source_map.get(filepath)
        if fs:
            pre_pdg = build_pdg(parse_file(fs.pre_content, filepath)) if fs.pre_content else PDG()
            post_pdg = build_pdg(parse_file(fs.post_content, filepath)) if fs.post_content else PDG()
            file_pdgs[filepath] = (pre_pdg, post_pdg)

    # delta
    for cl in changed_lines:
        idents = set(_IDENT_RE.findall(cl['content'])) - _STOP_WORDS
        filepath = cl['file']
        line_num = cl['line_num']

        # PDG
        function = None
        if filepath in file_pdgs:
            pre_pdg, post_pdg = file_pdgs[filepath]
            pdg = post_pdg if cl['type'] == 'add' else pre_pdg
            if line_num in pdg.line_to_node:
                nid = pdg.line_to_node[line_num]
                function = pdg.nodes[nid].function

        # Use cleaned function name for label
        func_label = function or _extract_func_name_from_hunk(cl.get('hunk_func'))
        node = DeltaNode(
            node_id=len(delta.nodes),
            line=line_num,
            change_type=cl['type'],
            function=func_label,
            content=cl['content'],
            identifiers=idents,
            label=f"{cl['type']}:{func_label or ''}",
        )
        delta.nodes.append(node)

    _add_sequential_edges(delta, changed_lines)
    _add_pdg_edges(delta, changed_lines, file_pdgs)
    _add_dataflow_edges(delta)
    # NFG: name flows
    _annotate_name_flows(delta)

    return delta


def _parse_diff_changes(diff_text: str) -> List[Dict]:
    """ diff """
    changes = []
    current_file = None
    current_func = None
    pre_line = 0
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
            pre_line = int(m.group(1))
            post_line = int(m.group(3))
            # hunk header
            rest = line[m.end():]
            current_func = rest.strip() if rest.strip() else None
            continue

        if line.startswith('+') and not line.startswith('+++'):
            changes.append({
                'file': current_file,
                'type': 'add',
                'line_num': post_line,
                'content': line[1:],
                'hunk_func': current_func,
            })
            post_line += 1
        elif line.startswith('-') and not line.startswith('---'):
            changes.append({
                'file': current_file,
                'type': 'del',
                'line_num': pre_line,
                'content': line[1:],
                'hunk_func': current_func,
            })
            pre_line += 1
        else:
            pre_line += 1
            post_line += 1

    return changes


def _add_sequential_edges(delta: DeltaPDG, changed_lines: List[Dict]):
    """ hunk """
    prev_idx = -1
    prev_file = None
    prev_hunk_func = None

    for i, cl in enumerate(changed_lines):
        key = (cl['file'], cl.get('hunk_func'))
        if prev_idx >= 0 and key == (prev_file, prev_hunk_func):
            delta.edges.append(DeltaEdge(src=prev_idx, dst=i, kind='seq'))
        prev_idx = i
        prev_file = cl['file']
        prev_hunk_func = cl.get('hunk_func')


def _add_pdg_edges(delta: DeltaPDG, changed_lines: List[Dict],
                   file_pdgs: Dict[str, Tuple[PDG, PDG]]):
    """ PDG """
    # (file, line, type) -> delta_node_id
    line_to_delta = {}
    for i, cl in enumerate(changed_lines):
        line_to_delta[(cl['file'], cl['line_num'], cl['type'])] = i

    for filepath, (pre_pdg, post_pdg) in file_pdgs.items():
        # add post_pdg del pre_pdg
        for change_type, pdg in [('add', post_pdg), ('del', pre_pdg)]:
            # delta
            delta_nodes_in_file = [
                (i, cl) for i, cl in enumerate(changed_lines)
                if cl['file'] == filepath and cl['type'] == change_type
            ]

            for idx, cl in delta_nodes_in_file:
                line_num = cl['line_num']
                if line_num not in pdg.line_to_node:
                    continue
                pdg_nid = pdg.line_to_node[line_num]

                # PDG
                for neighbor_id, edge_kind in get_neighbors(pdg, pdg_nid):
                    neighbor_line = pdg.nodes[neighbor_id].line
                    key = (filepath, neighbor_line, change_type)
                    if key in line_to_delta:
                        other_idx = line_to_delta[key]
                        if other_idx != idx:
                            delta.edges.append(DeltaEdge(
                                src=idx, dst=other_idx, kind=edge_kind
                            ))


def _add_dataflow_edges(delta: DeltaPDG):
    """"""
    n = len(delta.nodes)
    ident_to_nodes = {}
    for i, node in enumerate(delta.nodes):
        for ident in node.identifiers:
            ident_to_nodes.setdefault(ident, []).append(i)

    added_edges = set()
    # seq
    seq_pairs = set()
    for edge in delta.edges:
        if edge.kind == 'seq':
            seq_pairs.add((min(edge.src, edge.dst), max(edge.src, edge.dst)))

    for ident, node_ids in ident_to_nodes.items():
        if len(node_ids) > 20: #
            continue
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                ni, nj = node_ids[i], node_ids[j]
                edge_key = (min(ni, nj), max(ni, nj))
                # seq
                if edge_key in seq_pairs or edge_key in added_edges:
                    continue
                delta.edges.append(DeltaEdge(
                    src=ni, dst=nj, kind='data',
                    flow_names={ident}
                ))
                added_edges.add(edge_key)


def _annotate_name_flows(delta: DeltaPDG):
    """
    NFG (Name Flow Graph)
     Flexeme Section 3.2 - PDG
     (u, v)flow_names = identifiers(u) ∩ identifiers(v)
     flow names WL kernel
    """
    for edge in delta.edges:
        if edge.src < len(delta.nodes) and edge.dst < len(delta.nodes):
            src_idents = delta.nodes[edge.src].identifiers
            dst_idents = delta.nodes[edge.dst].identifiers
            # =
            shared = src_idents & dst_idents
            if shared:
                edge.flow_names = edge.flow_names | shared


# ---------------------------------------------------------------------------
# Full-project Delta-PDG ()
# ---------------------------------------------------------------------------

def build_delta_pdg_full(diff_text: str, file_sources: List[FileSource]) -> DeltaPDG:
    """
     full-project delta-PDG

     build_delta_pdg
    1. git repo PDG
    2. 1-hop
    3. PDG

     WL kernel
    """
    delta = DeltaPDG()

    # diff
    changed_lines = _parse_diff_changes(diff_text)
    if not changed_lines:
        return delta

    # PDG
    file_pdgs = {}
    source_map = {fs.filepath: fs for fs in file_sources}

    for filepath in set(cl['file'] for cl in changed_lines):
        fs = source_map.get(filepath)
        if fs:
            pre_pdg = build_pdg(parse_file(fs.pre_content, filepath)) if fs.pre_content else PDG()
            post_pdg = build_pdg(parse_file(fs.post_content, filepath)) if fs.post_content else PDG()
            file_pdgs[filepath] = (pre_pdg, post_pdg)

    changed_functions = set()  # (filepath, function_name)
    for cl in changed_lines:
        filepath = cl['file']
        if filepath in file_pdgs:
            pre_pdg, post_pdg = file_pdgs[filepath]
            pdg = post_pdg if cl['type'] == 'add' else pre_pdg
            if cl['line_num'] in pdg.line_to_node:
                nid = pdg.line_to_node[cl['line_num']]
                func = pdg.nodes[nid].function
                if func:
                    changed_functions.add((filepath, func))

    # delta +
    node_map = {}  # (filepath, line_num, type) -> delta_node_id
    context_nodes = set() # ()

    for cl in changed_lines:
        idents = set(_IDENT_RE.findall(cl['content'])) - _STOP_WORDS
        filepath = cl['file']
        line_num = cl['line_num']

        function = None
        if filepath in file_pdgs:
            pre_pdg, post_pdg = file_pdgs[filepath]
            pdg = post_pdg if cl['type'] == 'add' else pre_pdg
            if line_num in pdg.line_to_node:
                nid = pdg.line_to_node[line_num]
                function = pdg.nodes[nid].function

        func_label = function or _extract_func_name_from_hunk(cl.get('hunk_func'))
        node = DeltaNode(
            node_id=len(delta.nodes),
            line=line_num,
            change_type=cl['type'],
            function=func_label,
            content=cl['content'],
            identifiers=idents,
            label=f"{cl['type']}:{func_label or ''}",
        )
        delta.nodes.append(node)
        node_map[(filepath, line_num, cl['type'])] = node.node_id

    # PDG WL
    changed_line_set = {(cl['file'], cl['line_num'], cl['type']) for cl in changed_lines}

    for filepath, func_name in changed_functions:
        if filepath not in file_pdgs:
            continue
        pre_pdg, post_pdg = file_pdgs[filepath]

        # post
        for nid in post_pdg.func_nodes.get(func_name, []):
            pdg_node = post_pdg.nodes[nid]
            key = (filepath, pdg_node.line, 'add')
            if key not in node_map:
                idents = pdg_node.uses | pdg_node.defines
                ctx_node = DeltaNode(
                    node_id=len(delta.nodes),
                    line=pdg_node.line,
                    change_type='ctx', #
                    function=func_name,
                    content=pdg_node.content,
                    identifiers=idents,
                    label=f"ctx:{func_name}",
                )
                delta.nodes.append(ctx_node)
                node_map[(filepath, pdg_node.line, 'ctx')] = ctx_node.node_id
                context_nodes.add(ctx_node.node_id)

    # PDG delta
    _add_full_pdg_edges(delta, file_pdgs, node_map, changed_functions)
    _add_sequential_edges(delta, changed_lines)
    _annotate_name_flows(delta)

    return delta


def _add_full_pdg_edges(delta: DeltaPDG, file_pdgs: Dict,
                        node_map: Dict, changed_functions: Set[Tuple[str, str]]):
    """ PDG delta """
    added_edges = set()

    for filepath, func_name in changed_functions:
        if filepath not in file_pdgs:
            continue
        _, post_pdg = file_pdgs[filepath]

        # post_pdg
        func_node_ids = set(post_pdg.func_nodes.get(func_name, []))
        for edge in post_pdg.edges:
            if edge.src not in func_node_ids or edge.dst not in func_node_ids:
                continue

            src_line = post_pdg.nodes[edge.src].line
            dst_line = post_pdg.nodes[edge.dst].line

            # delta
            src_delta = (node_map.get((filepath, src_line, 'add')) or
                         node_map.get((filepath, src_line, 'ctx')))
            dst_delta = (node_map.get((filepath, dst_line, 'add')) or
                         node_map.get((filepath, dst_line, 'ctx')))

            if src_delta is not None and dst_delta is not None and src_delta != dst_delta:
                edge_key = (min(src_delta, dst_delta), max(src_delta, dst_delta), edge.kind)
                if edge_key not in added_edges:
                    delta.edges.append(DeltaEdge(
                        src=src_delta, dst=dst_delta, kind=edge.kind
                    ))
                    added_edges.add(edge_key)
