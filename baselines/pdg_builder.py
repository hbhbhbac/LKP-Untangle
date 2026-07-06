"""
pdg_builder.py — (PDG)

 ParsedFile
  - (CFG): //
  - (def-use):
  - PDG

 scope-based reaching definition
 commit untangling
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .ast_parser import ParsedFile, FunctionDef, FuncCall, IdentRef, CFNode, Assignment


@dataclass
class PDGNode:
    """PDG = /"""
    node_id: int
    line: int
    end_line: int
    kind: str  # 'stmt', 'decl', 'call', 'control', 'return'
    function: Optional[str]
    content: str = ''
    defines: Set[str] = field(default_factory=set) #
    uses: Set[str] = field(default_factory=set) #


@dataclass
class PDGEdge:
    """PDG """
    src: int  # node_id
    dst: int  # node_id
    kind: str  # 'control', 'data', 'call'


@dataclass
class PDG:
    """"""
    nodes: List[PDGNode] = field(default_factory=list)
    edges: List[PDGEdge] = field(default_factory=list)
    line_to_node: Dict[int, int] = field(default_factory=dict)
    func_nodes: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))


def build_pdg(parsed: ParsedFile) -> PDG:
    """
     ParsedFile PDG
    1.
    2. CFG +
    3. def-use
    """
    pdg = PDG()
    if not parsed.source_lines:
        return pdg

    # Step 1: —
    line_info = _collect_line_info(parsed)

    for line_num in sorted(line_info.keys()):
        info = line_info[line_num]
        node = PDGNode(
            node_id=len(pdg.nodes),
            line=line_num,
            end_line=info.get('end_line', line_num),
            kind=info['kind'],
            function=info.get('function'),
            content=parsed.source_lines[line_num - 1] if line_num <= len(parsed.source_lines) else '',
            defines=info.get('defines', set()),
            uses=info.get('uses', set()),
        )
        pdg.nodes.append(node)
        pdg.line_to_node[line_num] = node.node_id
        if node.function:
            pdg.func_nodes[node.function].append(node.node_id)

    # Step 2: CFG — +
    _build_cfg_edges(pdg, parsed)

    # Step 3: — scope-based def-use
    _build_dataflow_edges(pdg)

    return pdg


def _collect_line_info(parsed: ParsedFile) -> Dict[int, Dict]:
    """"""
    info = {}

    for call in parsed.calls:
        if call.line not in info:
            info[call.line] = {
                'kind': 'call', 'function': call.caller_function,
                'defines': set(), 'uses': set(),
            }
        info[call.line]['uses'].add(call.callee)

    for decl in parsed.declarations:
        if decl.line not in info:
            info[decl.line] = {
                'kind': 'decl',
                'function': decl.scope if decl.scope != 'global' else None,
                'defines': set(), 'uses': set(),
            }
        info[decl.line]['defines'].add(decl.name)

    for cf in parsed.control_flow:
        if cf.start_line not in info:
            info[cf.start_line] = {
                'kind': 'control', 'function': cf.enclosing_function,
                'end_line': cf.end_line,
                'defines': set(), 'uses': set(),
            }
        else:
            info[cf.start_line]['kind'] = 'control'
            info[cf.start_line]['end_line'] = cf.end_line

    # — uses
    for ident in parsed.identifiers:
        if ident.line not in info:
            info[ident.line] = {
                'kind': 'stmt', 'function': ident.enclosing_function,
                'defines': set(), 'uses': set(),
            }
        info[ident.line]['uses'].add(ident.name)

    # — target define
    for assign in parsed.assignments:
        line = assign.line
        if line not in info:
            info[line] = {
                'kind': 'stmt', 'function': assign.enclosing_function,
                'defines': set(), 'uses': set(),
            }
        info[line]['defines'].add(assign.target)

    # '=' define
    # ( assignments )

    return info


def _build_cfg_edges(pdg: PDG, parsed: ParsedFile):
    """"""
    for func_name, node_ids in pdg.func_nodes.items():
        sorted_ids = sorted(node_ids, key=lambda nid: pdg.nodes[nid].line)
        for i in range(len(sorted_ids) - 1):
            pdg.edges.append(PDGEdge(
                src=sorted_ids[i], dst=sorted_ids[i + 1], kind='control'
            ))

    # if/for/while body
    for node in pdg.nodes:
        if node.kind == 'control' and node.end_line > node.line:
            # body
            for other in pdg.nodes:
                if (other.node_id != node.node_id and
                        other.function == node.function and
                        node.line < other.line <= node.end_line):
                    pdg.edges.append(PDGEdge(
                        src=node.node_id, dst=other.node_id, kind='control'
                    ))
                    break #


def _build_dataflow_edges(pdg: PDG):
    """
     (def-use)
    Scope-based A x
     B x A.line < B.line data
    """
    func_defs = defaultdict(list)  # func -> [(node_id, var_name)]
    func_uses = defaultdict(list)  # func -> [(node_id, var_name)]

    for node in pdg.nodes:
        func = node.function or '__global__'
        for var in node.defines:
            func_defs[func].append((node.node_id, var))
        for var in node.uses:
            func_uses[func].append((node.node_id, var))

    # def → use
    for func in func_defs:
        defs = func_defs[func]
        uses = func_uses.get(func, [])
        for def_nid, def_var in defs:
            def_line = pdg.nodes[def_nid].line
            for use_nid, use_var in uses:
                if (use_var == def_var and
                        use_nid != def_nid and
                        pdg.nodes[use_nid].line > def_line):
                    pdg.edges.append(PDGEdge(
                        src=def_nid, dst=use_nid, kind='data'
                    ))

    # def-use
    global_defs = func_defs.get('__global__', [])
    for func, uses in func_uses.items():
        if func == '__global__':
            continue
        for def_nid, def_var in global_defs:
            for use_nid, use_var in uses:
                if use_var == def_var:
                    pdg.edges.append(PDGEdge(
                        src=def_nid, dst=use_nid, kind='data'
                    ))


def get_neighbors(pdg: PDG, node_id: int) -> List[Tuple[int, str]]:
    """ (node_id, edge_kind)"""
    neighbors = []
    for edge in pdg.edges:
        if edge.src == node_id:
            neighbors.append((edge.dst, edge.kind))
        elif edge.dst == node_id:
            neighbors.append((edge.src, edge.kind))
    return neighbors


def get_subgraph_for_lines(pdg: PDG, lines: Set[int]) -> Tuple[List[int], List[PDGEdge]]:
    """
    delta-PDG
     lines +
    """
    target_nodes = set()
    for line in lines:
        if line in pdg.line_to_node:
            target_nodes.add(pdg.line_to_node[line])

    extended_nodes = set(target_nodes)
    for nid in target_nodes:
        for neighbor_id, _ in get_neighbors(pdg, nid):
            extended_nodes.add(neighbor_id)

    sub_edges = []
    for edge in pdg.edges:
        if edge.src in extended_nodes and edge.dst in extended_nodes:
            sub_edges.append(edge)

    return sorted(extended_nodes), sub_edges
