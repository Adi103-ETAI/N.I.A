"""Glob pattern matching utilities.

Provides file pattern matching using ripgrep or native Python.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import List, Optional


def glob_files(
    pattern: str,
    root_dir: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None,
    use_ripgrep: bool = True,
) -> List[str]:
    """Find files matching a glob pattern.

    Args:
        pattern: The glob pattern to match (e.g., '*.py', '**/*.ts').
        root_dir: The root directory to search from. Defaults to cwd.
        ignore_patterns: Patterns to ignore (e.g., ['node_modules', '.git']).
        use_ripgrep: Whether to use ripgrep for faster searching.

    Returns:
        A list of matching file paths.
    """
    root = Path(root_dir) if root_dir else Path.cwd()

    if use_ripgrep:
        try:
            return _glob_with_ripgrep(pattern, root, ignore_patterns)
        except Exception:
            # Fall back to native glob
            pass

    return _glob_native(pattern, root, ignore_patterns)


def _glob_with_ripgrep(
    pattern: str,
    root: Path,
    ignore_patterns: Optional[List[str]] = None,
) -> List[str]:
    """Use ripgrep's --files for fast file listing."""
    import asyncio
    from .ripgrep import rip_grep

    args = ["--files"]

    # Convert glob pattern to ripgrep glob
    if pattern:
        args.extend(["--glob", pattern])

    if ignore_patterns:
        for p in ignore_patterns:
            args.extend(["--glob", f"!{p}"])

    try:
        result = asyncio.get_event_loop().run_until_complete(
            rip_grep(args, str(root))
        )
        return [os.path.join(str(root), f) for f in result]
    except Exception:
        return []


def _glob_native(
    pattern: str,
    root: Path,
    ignore_patterns: Optional[List[str]] = None,
) -> List[str]:
    """Native Python glob implementation."""
    if not pattern:
        return []

    # Use Path.glob for the pattern
    matches = root.glob(pattern)

    result = []
    for match in matches:
        if match.is_file():
            # Check ignore patterns
            if ignore_patterns and _should_ignore(str(match), ignore_patterns):
                continue
            result.append(str(match))

    return sorted(result)


def _should_ignore(path: str, patterns: List[str]) -> bool:
    """Check if a path should be ignored based on patterns."""
    path_parts = Path(path).parts

    for pattern in patterns:
        # Check if any part of the path matches the pattern
        for part in path_parts:
            if fnmatch.fnmatch(part, pattern):
                return True
        # Also check the full path
        if fnmatch.fnmatch(path, pattern):
            return True

    return False


def match_files(
    directory: str,
    include: Optional[str] = None,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """Match files in a directory with include/exclude patterns.

    Args:
        directory: The directory to search.
        include: A glob pattern for files to include (e.g., '*.py').
        exclude: A list of glob patterns for files to exclude.

    Returns:
        A sorted list of matching file paths.
    """
    root = Path(directory)
    if not root.is_dir():
        return []

    if include:
        files = root.glob(f"**/{include}")
    else:
        files = root.glob("**/*")

    result = []
    for f in files:
        if f.is_file():
            if exclude and _should_ignore(str(f), exclude):
                continue
            result.append(str(f))

    return sorted(result)


def is_glob_pattern(pattern: str) -> bool:
    """Check if a string contains glob special characters."""
    special_chars = set("*?[]{}")
    return any(c in special_chars for c in pattern)
