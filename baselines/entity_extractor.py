"""
entity_extractor.py — + Entity Reference Graph

 AST
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .ast_parser import ParsedFile, parse_c_file, FunctionDef, VarDecl, FuncCall
from .source_resolver import FileSource

import re

_MACRO_RE = re.compile(r'^\s*#\s*define\s+(\w+)')
_STRUCT_RE = re.compile(r'\b(?:struct|enum|union)\s+(\w+)')
_TYPEDEF_RE = re.compile(r'\btypedef\b.*\b(\w+)\s*;')


@dataclass
class Entity:
    """"""
    name: str
    kind: str  # 'function', 'variable', 'struct', 'macro', 'typedef'
    file: str
    line: int
    scope: str = 'global'  # 'global' or function name


@dataclass
class EntityRef:
    """"""
    src: str #
    dst: str #
    kind: str  # 'call', 'use', 'type_use'


@dataclass
class EntityGraph:
    """"""
    entities: Dict[str, Entity] = field(default_factory=dict)
    references: List[EntityRef] = field(default_factory=list)
    entity_by_file: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))


def extract_entities(file_sources: List[FileSource]) -> EntityGraph:
    """
    """
    graph = EntityGraph()

    for fs in file_sources:
        source = fs.post_content or fs.pre_content
        if not source:
            continue

        parsed = parse_c_file(source)
        _extract_from_parsed(parsed, fs.filepath, graph)
        _extract_macros_and_structs(source, fs.filepath, graph)

    _build_references(file_sources, graph)

    return graph


def _extract_from_parsed(parsed: ParsedFile, filepath: str, graph: EntityGraph):
    """ ParsedFile """
    for func in parsed.functions:
        entity = Entity(
            name=func.name, kind='function',
            file=filepath, line=func.start_line,
        )
        graph.entities[func.name] = entity
        graph.entity_by_file[filepath].append(func.name)

    for decl in parsed.declarations:
        if decl.scope == 'global':
            entity = Entity(
                name=decl.name, kind='variable',
                file=filepath, line=decl.line,
            )
            if decl.name not in graph.entities:
                graph.entities[decl.name] = entity
                graph.entity_by_file[filepath].append(decl.name)


def _extract_macros_and_structs(source: str, filepath: str, graph: EntityGraph):
    """"""
    for line in source.splitlines():
        m = _MACRO_RE.match(line)
        if m:
            name = m.group(1)
            if name not in graph.entities:
                graph.entities[name] = Entity(
                    name=name, kind='macro', file=filepath, line=0,
                )
                graph.entity_by_file[filepath].append(name)

        # //
        for m in _STRUCT_RE.finditer(line):
            name = m.group(1)
            if name not in graph.entities:
                graph.entities[name] = Entity(
                    name=name, kind='struct', file=filepath, line=0,
                )
                graph.entity_by_file[filepath].append(name)


def _build_references(file_sources: List[FileSource], graph: EntityGraph):
    """"""
    entity_names = set(graph.entities.keys())

    for fs in file_sources:
        source = fs.post_content or fs.pre_content
        if not source:
            continue

        parsed = parse_c_file(source)

        # → call
        for call in parsed.calls:
            if call.callee in entity_names and call.caller_function:
                graph.references.append(EntityRef(
                    src=call.caller_function,
                    dst=call.callee,
                    kind='call',
                ))

        # → use
        for ident in parsed.identifiers:
            if (ident.name in entity_names and
                    ident.enclosing_function and
                    ident.name != ident.enclosing_function):
                graph.references.append(EntityRef(
                    src=ident.enclosing_function,
                    dst=ident.name,
                    kind='use',
                ))


def get_entities_for_lines(source: str, lines: Set[int],
                           graph: EntityGraph) -> Set[str]:
    """"""
    entity_names = set(graph.entities.keys())
    result = set()

    source_lines = source.splitlines()
    ident_re = re.compile(r'\b([a-zA-Z_]\w{2,})\b')

    for line_num in lines:
        if 1 <= line_num <= len(source_lines):
            line_content = source_lines[line_num - 1]
            for m in ident_re.finditer(line_content):
                name = m.group(1)
                if name in entity_names:
                    result.add(name)

    return result


def get_shared_entities(entities_a: Set[str], entities_b: Set[str]) -> float:
    """ Jaccard """
    if not entities_a and not entities_b:
        return 0.0
    intersection = entities_a & entities_b
    union = entities_a | entities_b
    return len(intersection) / len(union) if union else 0.0
