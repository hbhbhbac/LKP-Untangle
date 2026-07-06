"""
ast_parser.py — tree-sitter C/C# source code parser

Extract from C/C# source codeFunctionVariableFunction
If tree-sitter is unavailable, falls back to enhanced regex parsing。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class FunctionDef:
    name: str
    start_line: int
    end_line: int
    parameters: List[str] = field(default_factory=list)


@dataclass
class VarDecl:
    name: str
    type_name: str
    scope: str  # 'global' or function name
    line: int


@dataclass
class FuncCall:
    callee: str
    caller_function: Optional[str]
    line: int


@dataclass
class IdentRef:
    name: str
    enclosing_function: Optional[str]
    line: int


@dataclass
class CFNode:
    kind: str  # 'if', 'for', 'while', 'switch', 'goto', 'return'
    start_line: int
    end_line: int
    enclosing_function: Optional[str]


@dataclass
class Assignment:
    target: str
    enclosing_function: Optional[str]
    line: int


@dataclass
class ParsedFile:
    functions: List[FunctionDef] = field(default_factory=list)
    declarations: List[VarDecl] = field(default_factory=list)
    assignments: List[Assignment] = field(default_factory=list)
    calls: List[FuncCall] = field(default_factory=list)
    identifiers: List[IdentRef] = field(default_factory=list)
    control_flow: List[CFNode] = field(default_factory=list)
    source_lines: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# tree-sitter Parse ()
# ---------------------------------------------------------------------------

_TS_AVAILABLE = False
_TS_PARSER = None
_TS_LANGUAGE = None

try:
    import tree_sitter_c as tsc
    from tree_sitter import Language, Parser

    _TS_LANGUAGE = Language(tsc.language())
    _TS_PARSER = Parser(_TS_LANGUAGE)
    _TS_AVAILABLE = True
except ImportError:
    pass

# C# tree-sitter support
_TS_CSHARP_AVAILABLE = False
_TS_CSHARP_PARSER = None

try:
    import tree_sitter_c_sharp as ts_csharp
    from tree_sitter import Language, Parser

    _TS_CSHARP_LANGUAGE = Language(ts_csharp.language())
    _TS_CSHARP_PARSER = Parser(_TS_CSHARP_LANGUAGE)
    _TS_CSHARP_AVAILABLE = True
except ImportError:
    pass

# Java tree-sitter support
_TS_JAVA_AVAILABLE = False
_TS_JAVA_PARSER = None

try:
    import tree_sitter_java as ts_java
    from tree_sitter import Language, Parser

    _TS_JAVA_LANGUAGE = Language(ts_java.language())
    _TS_JAVA_PARSER = Parser(_TS_JAVA_LANGUAGE)
    _TS_JAVA_AVAILABLE = True
except ImportError:
    pass


def _ts_parse(source_text: str) -> ParsedFile:
    """ tree-sitter Parse C """
    tree = _TS_PARSER.parse(source_text.encode('utf-8'))
    root = tree.root_node
    lines = source_text.splitlines()

    parsed = ParsedFile(source_lines=lines)
    _ts_walk(root, parsed, enclosing_func=None)
    return parsed


def _ts_walk(node, parsed: ParsedFile, enclosing_func: Optional[str]):
    """ AST Node"""
    ntype = node.type

    if ntype == 'function_definition':
        func_name = _ts_get_func_name(node)
        if func_name:
            params = _ts_get_params(node)
            parsed.functions.append(FunctionDef(
                name=func_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parameters=params,
            ))
            # Add parameters as VarDecl at function start line
            func_line = node.start_point[0] + 1
            for param_name in params:
                parsed.declarations.append(VarDecl(
                    name=param_name,
                    type_name='param',
                    scope=func_name,
                    line=func_line,
                ))
                # NodeUpdate enclosing_func
            for child in node.children:
                _ts_walk(child, parsed, enclosing_func=func_name)
            return

    elif ntype == 'declaration':
        decls = _ts_extract_declarations(node, enclosing_func)
        parsed.declarations.extend(decls)

    elif ntype == 'assignment_expression':
        left = node.child_by_field_name('left')
        target = _ts_extract_lhs_identifier(left)
        if target:
            parsed.assignments.append(Assignment(
                target=target,
                enclosing_function=enclosing_func,
                line=node.start_point[0] + 1,
            ))

    elif ntype == 'update_expression':
        # i++, --j etc. — these both use and define
        arg = node.child_by_field_name('argument')
        target = _ts_extract_lhs_identifier(arg)
        if target:
            parsed.assignments.append(Assignment(
                target=target,
                enclosing_function=enclosing_func,
                line=node.start_point[0] + 1,
            ))

    elif ntype == 'call_expression':
        callee = _ts_get_call_name(node)
        if callee:
            parsed.calls.append(FuncCall(
                callee=callee,
                caller_function=enclosing_func,
                line=node.start_point[0] + 1,
            ))

    elif ntype == 'identifier':
        #
        parent_type = node.parent.type if node.parent else ''
        if parent_type not in ('function_declarator', 'type_identifier',
                               'preproc_def', 'preproc_function_def'):
            parsed.identifiers.append(IdentRef(
                name=node.text.decode('utf-8'),
                enclosing_function=enclosing_func,
                line=node.start_point[0] + 1,
            ))

    elif ntype in ('if_statement', 'for_statement', 'while_statement',
                   'do_statement', 'switch_statement', 'goto_statement',
                   'return_statement'):
        kind_map = {
            'if_statement': 'if', 'for_statement': 'for',
            'while_statement': 'while', 'do_statement': 'while',
            'switch_statement': 'switch', 'goto_statement': 'goto',
            'return_statement': 'return',
        }
        parsed.control_flow.append(CFNode(
            kind=kind_map[ntype],
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            enclosing_function=enclosing_func,
        ))

    # Node
    for child in node.children:
        _ts_walk(child, parsed, enclosing_func)


def _ts_get_func_name(node) -> Optional[str]:
    """ function_definition NodeExtractFunction"""
    declarator = node.child_by_field_name('declarator')
    if declarator:
        # pointer_declarator
        while declarator.type in ('pointer_declarator', 'parenthesized_declarator'):
            for child in declarator.children:
                if child.type in ('function_declarator', 'identifier'):
                    declarator = child
                    break
            else:
                break
        if declarator.type == 'function_declarator':
            for child in declarator.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
        elif declarator.type == 'identifier':
            return declarator.text.decode('utf-8')
    return None


def _ts_get_params(node) -> List[str]:
    """ExtractFunctionParameter"""
    params = []
    declarator = node.child_by_field_name('declarator')
    if not declarator:
        return params
    # Navigate through pointer_declarator wrapping if needed
    inner = declarator
    while inner.type in ('pointer_declarator', 'parenthesized_declarator'):
        found = False
        for child in inner.children:
            if child.type in ('function_declarator', 'pointer_declarator',
                              'parenthesized_declarator'):
                inner = child
                found = True
                break
        if not found:
            break
    if inner.type != 'function_declarator':
        return params
        # parameter_list
    for child in inner.children:
        if child.type == 'parameter_list':
            for param in child.children:
                if param.type == 'parameter_declaration':
                    decl = param.child_by_field_name('declarator')
                    if decl:
                        name = _ts_unwrap_declarator(decl)
                        if name:
                            params.append(name)
    return params


def _ts_unwrap_declarator(node) -> Optional[str]:
    """Recursively unwrap pointer/array/init/function declarators to find the identifier name."""
    if node is None:
        return None
    if node.type == 'identifier':
        return node.text.decode('utf-8')
    if node.type in ('pointer_declarator', 'array_declarator', 'parenthesized_declarator'):
        for child in node.children:
            if child.type in ('identifier', 'pointer_declarator', 'array_declarator',
                              'parenthesized_declarator', 'function_declarator',
                              'init_declarator'):
                result = _ts_unwrap_declarator(child)
                if result:
                    return result
        # fallback: try any child
        for child in node.children:
            result = _ts_unwrap_declarator(child)
            if result:
                return result
    if node.type == 'init_declarator':
        decl = node.child_by_field_name('declarator')
        if decl:
            return _ts_unwrap_declarator(decl)
    if node.type == 'function_declarator':
        # e.g. (*func_ptr)(args) — get the identifier from the declarator child
        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode('utf-8')
            if child.type in ('parenthesized_declarator', 'pointer_declarator'):
                result = _ts_unwrap_declarator(child)
                if result:
                    return result
    return None


def _ts_extract_lhs_identifier(node) -> Optional[str]:
    """Extract the base identifier from an assignment LHS (handles field_expression, pointer, subscript)."""
    if node is None:
        return None
    if node.type == 'identifier':
        return node.text.decode('utf-8')
    if node.type == 'field_expression':
        # e.g. data->field or data.field — target is the base object
        arg = node.child_by_field_name('argument')
        if arg:
            return _ts_extract_lhs_identifier(arg)
    if node.type == 'pointer_expression':
        # e.g. *ptr
        for child in node.children:
            if child.type != '*':
                return _ts_extract_lhs_identifier(child)
    if node.type == 'subscript_expression':
        # e.g. arr[i]
        arg = node.child_by_field_name('argument')
        if arg:
            return _ts_extract_lhs_identifier(arg)
    if node.type == 'parenthesized_expression':
        for child in node.children:
            if child.type not in ('(', ')'):
                return _ts_extract_lhs_identifier(child)
    return None


def _ts_extract_declarations(node, enclosing_func: Optional[str]) -> List[VarDecl]:
    """ declaration NodeExtractVariable"""
    decls = []
    type_node = node.child_by_field_name('type')
    type_name = type_node.text.decode('utf-8') if type_node else 'unknown'

    for child in node.children:
        if child.type == 'identifier':
            decls.append(VarDecl(
                name=child.text.decode('utf-8'),
                type_name=type_name,
                scope=enclosing_func or 'global',
                line=node.start_point[0] + 1,
            ))
        elif child.type in ('init_declarator', 'pointer_declarator',
                            'array_declarator', 'function_declarator',
                            'parenthesized_declarator'):
            name = _ts_unwrap_declarator(child)
            if name:
                decls.append(VarDecl(
                    name=name,
                    type_name=type_name,
                    scope=enclosing_func or 'global',
                    line=node.start_point[0] + 1,
                ))
    return decls


def _ts_get_call_name(node) -> Optional[str]:
    """ call_expression ExtractFunction"""
    func_node = node.child_by_field_name('function')
    if func_node:
        if func_node.type == 'identifier':
            return func_node.text.decode('utf-8')
        elif func_node.type == 'field_expression':
            # struct->method
            field = func_node.child_by_field_name('field')
            if field:
                return field.text.decode('utf-8')
    return None


# ---------------------------------------------------------------------------
# tree-sitter C# Parse
# ---------------------------------------------------------------------------

def _ts_csharp_parse(source_text: str) -> ParsedFile:
    """ tree-sitter Parse C# """
    tree = _TS_CSHARP_PARSER.parse(source_text.encode('utf-8'))
    root = tree.root_node
    lines = source_text.splitlines()

    parsed = ParsedFile(source_lines=lines)
    _ts_csharp_walk(root, parsed, enclosing_func=None)
    return parsed


def _ts_csharp_walk(node, parsed: ParsedFile, enclosing_func: Optional[str]):
    """ C# AST Node"""
    ntype = node.type

    # Method/constructor/property declarations
    if ntype in ('method_declaration', 'constructor_declaration',
                 'operator_declaration', 'conversion_operator_declaration'):
        func_name = _ts_csharp_get_method_name(node)
        if func_name:
            params = _ts_csharp_get_params(node)
            parsed.functions.append(FunctionDef(
                name=func_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parameters=params,
            ))
            for param_name in params:
                parsed.declarations.append(VarDecl(
                    name=param_name, type_name='param',
                    scope=func_name, line=node.start_point[0] + 1,
                ))
            for child in node.children:
                _ts_csharp_walk(child, parsed, enclosing_func=func_name)
            return

    elif ntype == 'property_declaration':
        prop_name = _ts_csharp_get_identifier(node)
        if prop_name:
            parsed.declarations.append(VarDecl(
                name=prop_name, type_name='property',
                scope=enclosing_func or 'class',
                line=node.start_point[0] + 1,
            ))

    elif ntype == 'local_declaration_statement':
        decls = _ts_csharp_extract_local_decls(node, enclosing_func)
        parsed.declarations.extend(decls)

    elif ntype == 'field_declaration':
        decls = _ts_csharp_extract_field_decls(node, enclosing_func)
        parsed.declarations.extend(decls)

    elif ntype == 'assignment_expression':
        left = node.child_by_field_name('left')
        target = _ts_csharp_extract_identifier(left)
        if target:
            parsed.assignments.append(Assignment(
                target=target, enclosing_function=enclosing_func,
                line=node.start_point[0] + 1,
            ))

    elif ntype == 'invocation_expression':
        callee = _ts_csharp_get_invocation_name(node)
        if callee:
            parsed.calls.append(FuncCall(
                callee=callee, caller_function=enclosing_func,
                line=node.start_point[0] + 1,
            ))

    elif ntype == 'identifier':
        parent_type = node.parent.type if node.parent else ''
        if parent_type not in ('class_declaration', 'namespace_declaration',
                               'using_directive', 'type_parameter'):
            parsed.identifiers.append(IdentRef(
                name=node.text.decode('utf-8'),
                enclosing_function=enclosing_func,
                line=node.start_point[0] + 1,
            ))

    elif ntype in ('if_statement', 'for_statement', 'for_each_statement',
                   'while_statement', 'do_statement', 'switch_statement',
                   'return_statement', 'throw_statement'):
        kind_map = {
            'if_statement': 'if', 'for_statement': 'for',
            'for_each_statement': 'for', 'while_statement': 'while',
            'do_statement': 'while', 'switch_statement': 'switch',
            'return_statement': 'return', 'throw_statement': 'return',
        }
        parsed.control_flow.append(CFNode(
            kind=kind_map[ntype],
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            enclosing_function=enclosing_func,
        ))

    for child in node.children:
        _ts_csharp_walk(child, parsed, enclosing_func)


def _ts_csharp_get_method_name(node) -> Optional[str]:
    """Extract method name from a method/constructor declaration."""
    for child in node.children:
        if child.type == 'identifier':
            return child.text.decode('utf-8')
    return None


def _ts_csharp_get_identifier(node) -> Optional[str]:
    """Get the first identifier child of a node."""
    if node is None:
        return None
    for child in node.children:
        if child.type == 'identifier':
            return child.text.decode('utf-8')
    return None


def _ts_csharp_extract_identifier(node) -> Optional[str]:
    """Extract base identifier from an expression (handles member access)."""
    if node is None:
        return None
    if node.type == 'identifier':
        return node.text.decode('utf-8')
    if node.type == 'member_access_expression':
        name_node = node.child_by_field_name('name')
        if name_node:
            return name_node.text.decode('utf-8')
    if node.type == 'element_access_expression':
        expr = node.child_by_field_name('expression')
        return _ts_csharp_extract_identifier(expr)
    return None


def _ts_csharp_get_params(node) -> List[str]:
    """Extract parameter names from a method declaration."""
    params = []
    for child in node.children:
        if child.type == 'parameter_list':
            for param in child.children:
                if param.type == 'parameter':
                    for pc in param.children:
                        if pc.type == 'identifier':
                            params.append(pc.text.decode('utf-8'))
                            break
    return params


def _ts_csharp_extract_local_decls(node, enclosing_func: Optional[str]) -> List[VarDecl]:
    """Extract local variable declarations."""
    decls = []
    for child in node.children:
        if child.type == 'variable_declaration':
            type_name = 'var'
            type_node = child.child_by_field_name('type')
            if type_node:
                type_name = type_node.text.decode('utf-8')
            for declarator in child.children:
                if declarator.type == 'variable_declarator':
                    name_node = declarator.child_by_field_name('name')
                    if not name_node:
                        for dc in declarator.children:
                            if dc.type == 'identifier':
                                name_node = dc
                                break
                    if name_node:
                        decls.append(VarDecl(
                            name=name_node.text.decode('utf-8'),
                            type_name=type_name,
                            scope=enclosing_func or 'local',
                            line=node.start_point[0] + 1,
                        ))
    return decls


def _ts_csharp_extract_field_decls(node, enclosing_func: Optional[str]) -> List[VarDecl]:
    """Extract class field declarations."""
    decls = []
    for child in node.children:
        if child.type == 'variable_declaration':
            type_name = 'field'
            type_node = child.child_by_field_name('type')
            if type_node:
                type_name = type_node.text.decode('utf-8')
            for declarator in child.children:
                if declarator.type == 'variable_declarator':
                    for dc in declarator.children:
                        if dc.type == 'identifier':
                            decls.append(VarDecl(
                                name=dc.text.decode('utf-8'),
                                type_name=type_name,
                                scope=enclosing_func or 'class',
                                line=node.start_point[0] + 1,
                            ))
                            break
    return decls


def _ts_csharp_get_invocation_name(node) -> Optional[str]:
    """Extract invoked method name."""
    func = node.child_by_field_name('function')
    if not func:
        # Try first non-argument-list child
        for child in node.children:
            if child.type != 'argument_list':
                func = child
                break
    if func:
        if func.type == 'identifier':
            return func.text.decode('utf-8')
        if func.type == 'member_access_expression':
            name_node = func.child_by_field_name('name')
            if name_node:
                return name_node.text.decode('utf-8')
        if func.type == 'generic_name':
            for child in func.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
    return None


# ---------------------------------------------------------------------------
# tree-sitter Java Parse
# ---------------------------------------------------------------------------

def _ts_java_parse(source_text: str) -> ParsedFile:
    """ tree-sitter Parse Java """
    tree = _TS_JAVA_PARSER.parse(source_text.encode('utf-8'))
    root = tree.root_node
    lines = source_text.splitlines()

    parsed = ParsedFile(source_lines=lines)
    _ts_java_walk(root, parsed, enclosing_func=None)
    return parsed


def _ts_java_walk(node, parsed: ParsedFile, enclosing_func: Optional[str]):
    """ Java AST Node"""
    ntype = node.type

    # Method/constructor declarations
    if ntype in ('method_declaration', 'constructor_declaration'):
        func_name = _ts_java_get_method_name(node)
        if func_name:
            params = _ts_java_get_params(node)
            parsed.functions.append(FunctionDef(
                name=func_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parameters=params,
            ))
            for param_name in params:
                parsed.declarations.append(VarDecl(
                    name=param_name, type_name='param',
                    scope=func_name, line=node.start_point[0] + 1,
                ))
            for child in node.children:
                _ts_java_walk(child, parsed, enclosing_func=func_name)
            return

    elif ntype == 'field_declaration':
        decls = _ts_java_extract_field_decls(node, enclosing_func)
        parsed.declarations.extend(decls)

    elif ntype == 'local_variable_declaration':
        decls = _ts_java_extract_local_decls(node, enclosing_func)
        parsed.declarations.extend(decls)

    elif ntype == 'assignment_expression':
        left = node.child_by_field_name('left')
        target = _ts_java_extract_identifier(left)
        if target:
            parsed.assignments.append(Assignment(
                target=target, enclosing_function=enclosing_func,
                line=node.start_point[0] + 1,
            ))

    elif ntype == 'update_expression':
        arg = node.child_by_field_name('argument')
        if arg is None:
            # try first identifier child
            for child in node.children:
                if child.type == 'identifier':
                    arg = child
                    break
        target = _ts_java_extract_identifier(arg)
        if target:
            parsed.assignments.append(Assignment(
                target=target, enclosing_function=enclosing_func,
                line=node.start_point[0] + 1,
            ))

    elif ntype == 'method_invocation':
        callee = _ts_java_get_invocation_name(node)
        if callee:
            parsed.calls.append(FuncCall(
                callee=callee, caller_function=enclosing_func,
                line=node.start_point[0] + 1,
            ))

    elif ntype == 'identifier':
        parent_type = node.parent.type if node.parent else ''
        if parent_type not in ('class_declaration', 'interface_declaration',
                               'package_declaration', 'import_declaration',
                               'type_identifier', 'enum_declaration',
                               'annotation_type_declaration'):
            parsed.identifiers.append(IdentRef(
                name=node.text.decode('utf-8'),
                enclosing_function=enclosing_func,
                line=node.start_point[0] + 1,
            ))

    elif ntype in ('if_statement', 'for_statement', 'enhanced_for_statement',
                   'while_statement', 'do_statement', 'switch_expression',
                   'switch_statement', 'return_statement', 'throw_statement'):
        kind_map = {
            'if_statement': 'if', 'for_statement': 'for',
            'enhanced_for_statement': 'for', 'while_statement': 'while',
            'do_statement': 'while', 'switch_expression': 'switch',
            'switch_statement': 'switch', 'return_statement': 'return',
            'throw_statement': 'return',
        }
        parsed.control_flow.append(CFNode(
            kind=kind_map[ntype],
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            enclosing_function=enclosing_func,
        ))

    for child in node.children:
        _ts_java_walk(child, parsed, enclosing_func)


def _ts_java_get_method_name(node) -> Optional[str]:
    """Extract method/constructor name from a Java declaration."""
    name_node = node.child_by_field_name('name')
    if name_node:
        return name_node.text.decode('utf-8')
    # Constructor: look for identifier child
    for child in node.children:
        if child.type == 'identifier':
            return child.text.decode('utf-8')
    return None


def _ts_java_get_params(node) -> List[str]:
    """Extract parameter names from a Java method declaration."""
    params = []
    param_list = node.child_by_field_name('parameters')
    if param_list is None:
        for child in node.children:
            if child.type == 'formal_parameters':
                param_list = child
                break
    if param_list:
        for child in param_list.children:
            if child.type == 'formal_parameter' or child.type == 'spread_parameter':
                name_node = child.child_by_field_name('name')
                if name_node:
                    params.append(name_node.text.decode('utf-8'))
                else:
                    for pc in child.children:
                        if pc.type == 'identifier':
                            params.append(pc.text.decode('utf-8'))
                            break
    return params


def _ts_java_extract_identifier(node) -> Optional[str]:
    """Extract base identifier from a Java expression."""
    if node is None:
        return None
    if node.type == 'identifier':
        return node.text.decode('utf-8')
    if node.type == 'field_access':
        field = node.child_by_field_name('field')
        if field:
            return field.text.decode('utf-8')
    if node.type == 'array_access':
        arr = node.child_by_field_name('array')
        return _ts_java_extract_identifier(arr)
    return None


def _ts_java_extract_field_decls(node, enclosing_func: Optional[str]) -> List[VarDecl]:
    """Extract Java field declarations."""
    decls = []
    type_name = 'field'
    type_node = node.child_by_field_name('type')
    if type_node:
        type_name = type_node.text.decode('utf-8')
    for child in node.children:
        if child.type == 'variable_declarator':
            name_node = child.child_by_field_name('name')
            if name_node:
                decls.append(VarDecl(
                    name=name_node.text.decode('utf-8'),
                    type_name=type_name,
                    scope=enclosing_func or 'class',
                    line=node.start_point[0] + 1,
                ))
    return decls


def _ts_java_extract_local_decls(node, enclosing_func: Optional[str]) -> List[VarDecl]:
    """Extract Java local variable declarations."""
    decls = []
    type_name = 'var'
    type_node = node.child_by_field_name('type')
    if type_node:
        type_name = type_node.text.decode('utf-8')
    for child in node.children:
        if child.type == 'variable_declarator':
            name_node = child.child_by_field_name('name')
            if name_node:
                decls.append(VarDecl(
                    name=name_node.text.decode('utf-8'),
                    type_name=type_name,
                    scope=enclosing_func or 'local',
                    line=node.start_point[0] + 1,
                ))
    return decls


def _ts_java_get_invocation_name(node) -> Optional[str]:
    """Extract invoked method name from Java method_invocation."""
    name_node = node.child_by_field_name('name')
    if name_node:
        return name_node.text.decode('utf-8')
    # Fallback: member access
    obj = node.child_by_field_name('object')
    if obj and obj.type == 'identifier':
        return obj.text.decode('utf-8')
    return None


# ---------------------------------------------------------------------------
# Regex fallback Parse ( tree-sitter )
# ---------------------------------------------------------------------------

_FUNC_DEF_RE = re.compile(
    r'^(?:static\s+)?(?:inline\s+)?(?:[\w\s\*]+?)\s+'
    r'(\w+)\s*\([^)]*\)\s*\{',
    re.MULTILINE
)
_VAR_DECL_RE = re.compile(
    r'^\s*(?:static\s+)?(?:const\s+)?'
    r'((?:unsigned\s+|signed\s+)?(?:int|char|long|short|float|double|void|'
    r'struct\s+\w+|enum\s+\w+|u8|u16|u32|u64|s8|s16|s32|s64|size_t|'
    r'bool|__\w+)[\s\*]+)'
    r'(\w+)\s*[;=\[,]',
    re.MULTILINE
)
_FUNC_CALL_RE = re.compile(r'\b([a-zA-Z_]\w+)\s*\(')
_IDENT_RE = re.compile(r'\b([a-zA-Z_]\w{2,})\b')
_ASSIGN_RE = re.compile(r'^\s*(\w+)\s*(?:[+\-*/%&|^]|<<|>>)?=(?!=)')
_CONTROL_RE = re.compile(
    r'^\s*(if|for|while|do|switch|goto|return)\b',
    re.MULTILINE
)

_C_KEYWORDS = {
    'int', 'char', 'void', 'long', 'short', 'unsigned', 'signed',
    'float', 'double', 'struct', 'enum', 'union', 'typedef',
    'const', 'static', 'extern', 'volatile', 'register', 'inline',
    'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case',
    'break', 'continue', 'goto', 'default', 'sizeof', 'NULL', 'null',
    'true', 'false', 'bool', 'include', 'define', 'ifdef', 'ifndef',
    'endif', 'elif', 'undef', 'pragma',
}


def _regex_parse(source_text: str) -> ParsedFile:
    """Regex-based fallback parser"""
    lines = source_text.splitlines()
    parsed = ParsedFile(source_lines=lines)

    # Function
    current_func = None
    brace_depth = 0
    func_start = 0

    for i, line in enumerate(lines, 1):
        #
        brace_depth += line.count('{') - line.count('}')

        m = _FUNC_DEF_RE.match(line)
        if m and brace_depth <= 1:
            if current_func:
                parsed.functions[-1].end_line = i - 1
            current_func = m.group(1)
            parsed.functions.append(FunctionDef(
                name=current_func, start_line=i, end_line=i,
            ))
            func_start = i

        if brace_depth == 0 and current_func:
            parsed.functions[-1].end_line = i
            current_func = None

    # ExtractFunction
    for i, line in enumerate(lines, 1):
        enc_func = _find_enclosing_func(parsed.functions, i)

        for m in _FUNC_CALL_RE.finditer(line):
            name = m.group(1)
            if name not in _C_KEYWORDS:
                parsed.calls.append(FuncCall(
                    callee=name, caller_function=enc_func, line=i
                ))

        for m in _IDENT_RE.finditer(line):
            name = m.group(1)
            if name not in _C_KEYWORDS:
                parsed.identifiers.append(IdentRef(
                    name=name, enclosing_function=enc_func, line=i
                ))

        #
        cm = _CONTROL_RE.match(line)
        if cm:
            parsed.control_flow.append(CFNode(
                kind=cm.group(1), start_line=i, end_line=i,
                enclosing_function=enc_func,
            ))

        #
        am = _ASSIGN_RE.match(line)
        if am:
            target = am.group(1)
            if target not in _C_KEYWORDS:
                parsed.assignments.append(Assignment(
                    target=target,
                    enclosing_function=enc_func,
                    line=i,
                ))

    return parsed


def _find_enclosing_func(functions: List[FunctionDef], line: int) -> Optional[str]:
    """Function"""
    for f in functions:
        if f.start_line <= line <= f.end_line:
            return f.name
    return None


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def parse_c_file(source_text: str) -> ParsedFile:
    """
 Parse C Return ParsedFile
 tree-sitterfallback regex
    """
    if not source_text or not source_text.strip():
        return ParsedFile()

    if _TS_AVAILABLE:
        try:
            return _ts_parse(source_text)
        except Exception:
            pass

    return _regex_parse(source_text)


def parse_csharp_file(source_text: str) -> ParsedFile:
    """
 Parse C# Return ParsedFile
 tree-sitter-c-sharpfallback regex
    """
    if not source_text or not source_text.strip():
        return ParsedFile()

    if _TS_CSHARP_AVAILABLE:
        try:
            return _ts_csharp_parse(source_text)
        except Exception:
            pass

    # Fallback: regex parser C# (Function)
    return _regex_parse(source_text)


def parse_java_file(source_text: str) -> ParsedFile:
    """
 Parse Java Return ParsedFile
 tree-sitter-javafallback regex
    """
    if not source_text or not source_text.strip():
        return ParsedFile()

    if _TS_JAVA_AVAILABLE:
        try:
            return _ts_java_parse(source_text)
        except Exception:
            pass

    # Fallback: regex parser catches basic patterns
    return _regex_parse(source_text)


def parse_file(source_text: str, filepath: str = '') -> ParsedFile:
    """
 FileParse
    """
    if is_csharp_file(filepath):
        return parse_csharp_file(source_text)
    if is_java_file(filepath):
        return parse_java_file(source_text)
    return parse_c_file(source_text)


def map_line_to_function(parsed: ParsedFile, line: int) -> Optional[str]:
    """Function"""
    return _find_enclosing_func(parsed.functions, line)


def is_c_file(filepath: str) -> bool:
    """ C File"""
    return filepath.endswith(('.c', '.h', '.S'))


def is_csharp_file(filepath: str) -> bool:
    """ C# File"""
    return filepath.endswith('.cs')


def is_java_file(filepath: str) -> bool:
    """ Java File"""
    return filepath.endswith('.java')
