"""
Grader contract mirrored from README.MD — allowed builtins and stdlib modules.
"""
from __future__ import annotations

import builtins
import importlib
from typing import Dict, FrozenSet

# Single source of truth (order matches README).
ALLOWED_MODULES: FrozenSet[str] = frozenset(
    {
        "itertools",
        "math",
        "re",
        "functools",
        "statistics",
        "collections",
        "string",
    }
)

ALLOWED_BUILTINS: FrozenSet[str] = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "bytes",
        "chr",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "hash",
        "hex",
        "int",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "__import__",
    }
)

def _safe_import(name: str, *args, **kwargs):
    if name not in ALLOWED_MODULES:
        raise ImportError(f"module {name!r} is not in the allowed list")
    return importlib.import_module(name)


def build_grader_namespace() -> Dict[str, object]:
    """
    Namespace suitable for eval() of a submission lambda.
    Only allowed builtin names are injected; __import__ is restricted to ALLOWED_MODULES.
    """
    ns: Dict[str, object] = {"__builtins__": {}}
    for name in ALLOWED_BUILTINS:
        if name == "__import__":
            ns["__import__"] = _safe_import
        else:
            ns[name] = getattr(builtins, name)
    return ns


def scan_disallowed_import_calls(source: str) -> list[str]:
    """
    Lightweight check: string scan for __import__('x') / __import__("x") where x not allowed.
    Full validation is done via AST + runtime in mini_grader.
    """
    import ast
    import re

    issues: list[str] = []
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as e:
        return [f"parse: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__import__":
            if node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    if arg0.value not in ALLOWED_MODULES:
                        issues.append(f"__import__({arg0.value!r}) not allowed")
    # Regex fallback for dynamic cases the AST might miss in malformed input
    for m in re.finditer(r"__import__\(\s*['\"]([^'\"]+)['\"]", source):
        if m.group(1) not in ALLOWED_MODULES:
            issues.append(f"__import__({m.group(1)!r}) not allowed")
    return issues
