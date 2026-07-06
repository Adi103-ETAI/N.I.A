"""Lightweight Python code intelligence.

Pure-Python AST-based implementation — no real LSP server required.
Provides just enough surface area to support the ``lsp`` tool:

- ``list_document_symbols(file_path)`` — top-level classes/functions in a file.
- ``workspace_symbol_search(root, query)`` — recursive grep across ``*.py``.
- ``go_to_definition(root, file_path, symbol, line, character)`` — find where a
  name is defined.
- ``find_references(root, file_path, symbol, line, character)`` — find all
  usages of a name.
- ``hover(root, file_path, symbol, line, character)`` — return signature +
  docstring for a name.

All functions are synchronous (the tool wraps them in ``async def``).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Symbol types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolLocation:
    """A symbol found by the LSP helpers."""

    kind: str  # "function", "class", "method", "variable"
    name: str
    path: Path
    line: int  # 1-based
    character: int  # 1-based
    signature: str | None = None
    docstring: str | None = None


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _kind_of(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "function"
    if isinstance(node, ast.ClassDef):
        return "class"
    return "variable"


def _signature(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = ast.unparse(node.args) if hasattr(ast, "unparse") else "(...)"
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}{args}"
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        if bases:
            return f"class {node.name}({bases})"
        return f"class {node.name}"
    return None


def _docstring_of(node: ast.AST) -> str | None:
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        if isinstance(first.value.value, str):
            return first.value.value
    return None


def _walk_python_files(root: Path):
    """Yield ``*.py`` files under ``root``, skipping common junk directories."""
    skip_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache", "build", "dist"}
    if root.is_file() and root.suffix == ".py":
        yield root
        return
    for path in root.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        yield path


def _parse_file(path: Path) -> ast.Module | None:
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        return ast.parse(src, filename=str(path))
    except SyntaxError:
        return None


def _line_of(node: ast.AST) -> int:
    return getattr(node, "lineno", 1) or 1


def _character_of(node: ast.AST) -> int:
    return getattr(node, "col_offset", 0) + 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_document_symbols(file_path: Path) -> list[SymbolLocation]:
    """Return top-level classes and functions defined in ``file_path``.

    Methods inside classes are also returned, nested under their parent class.
    """
    tree = _parse_file(file_path)
    if tree is None:
        return []

    out: list[SymbolLocation] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = _kind_of(node)
            if isinstance(node, ast.ClassDef):
                # methods
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out.append(
                            SymbolLocation(
                                kind="method",
                                name=f"{node.name}.{child.name}",
                                path=file_path,
                                line=_line_of(child),
                                character=_character_of(child),
                                signature=_signature(child),
                                docstring=_docstring_of(child),
                            )
                        )
            out.append(
                SymbolLocation(
                    kind=kind,
                    name=node.name,
                    path=file_path,
                    line=_line_of(node),
                    character=_character_of(node),
                    signature=_signature(node),
                    docstring=_docstring_of(node),
                )
            )
    return out


def workspace_symbol_search(root: Path, query: str) -> list[SymbolLocation]:
    """Return all symbols (functions/classes/methods) whose name contains ``query``."""
    if not query:
        return []
    needle = query.lower()
    out: list[SymbolLocation] = []
    for path in _walk_python_files(root):
        for sym in list_document_symbols(path):
            if needle in sym.name.lower():
                out.append(sym)
    return out


def _resolve_symbol_at(
    file_path: Path,
    symbol: str | None,
    line: int | None,
    character: int | None,
) -> str | None:
    """Return the symbol name to look up, given either an explicit name or a position."""
    if symbol:
        return symbol
    if line is None:
        return None
    try:
        src = file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    if line < 1 or line > len(src):
        return None
    text = src[line - 1]
    if character and character >= 1:
        # Walk left from the column to find an identifier boundary.
        i = min(character - 1, len(text) - 1)
        while i >= 0 and (not text[i].isalnum() and text[i] != "_"):
            i -= 1
        end = i + 1
        while i >= 0 and (text[i].isalnum() or text[i] == "_"):
            i -= 1
        start = i + 1
        token = text[start:end]
        # If the next char is '.', grab the rightmost component.
        if "." in token:
            token = token.rsplit(".", 1)[-1]
        return token or None
    return None


def go_to_definition(
    root: Path,
    file_path: Path,
    symbol: str | None = None,
    line: int | None = None,
    character: int | None = None,
) -> list[SymbolLocation]:
    """Find where ``symbol`` is defined within ``root``."""
    name = _resolve_symbol_at(file_path, symbol, line, character)
    if not name:
        return []
    results: list[SymbolLocation] = []
    for path in _walk_python_files(root):
        for sym in list_document_symbols(path):
            if sym.name == name or sym.name.endswith(f".{name}"):
                results.append(sym)
    return results


def find_references(
    root: Path,
    file_path: Path,
    symbol: str | None = None,
    line: int | None = None,
    character: int | None = None,
) -> list[tuple[Path, int, str]]:
    """Find all references to ``symbol`` across the workspace.

    Returns a list of ``(path, line, text)`` tuples.
    """
    name = _resolve_symbol_at(file_path, symbol, line, character)
    if not name:
        return []
    # Use a word-boundary regex so we don't match substrings.
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    out: list[tuple[Path, int, str]] = []
    for path in _walk_python_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for idx, text in enumerate(lines, start=1):
            if pattern.search(text):
                out.append((path, idx, text.strip()))
    return out


def hover(
    root: Path,
    file_path: Path,
    symbol: str | None = None,
    line: int | None = None,
    character: int | None = None,
) -> SymbolLocation | None:
    """Return signature + docstring for the symbol at the given position."""
    name = _resolve_symbol_at(file_path, symbol, line, character)
    if not name:
        return None
    # First look in the same file (most common case).
    for sym in list_document_symbols(file_path):
        if sym.name == name or sym.name.endswith(f".{name}"):
            return sym
    # Then scan the workspace.
    for path in _walk_python_files(root):
        if path == file_path:
            continue
        for sym in list_document_symbols(path):
            if sym.name == name or sym.name.endswith(f".{name}"):
                return sym
    return None
