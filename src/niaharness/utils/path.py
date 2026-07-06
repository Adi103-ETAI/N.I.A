"""Path utilities.

Provides path expansion, normalization, traversal detection, and config-key
normalization.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Optional


def expand_path(path_str: str, base_dir: Optional[str] = None) -> str:
    """Expand a path that may contain ~ to an absolute path.

    Handles:
    - ``~`` → user home
    - ``~/path`` → path within home
    - absolute paths → returned normalized
    - relative paths → resolved against base_dir (defaults to cwd)
    """
    if not isinstance(path_str, str):
        raise TypeError(f"Path must be a string, got {type(path_str).__name__}")

    actual_base = base_dir or os.getcwd()

    if "\0" in path_str or "\0" in actual_base:
        raise ValueError("Path contains null bytes")

    trimmed = path_str.strip()
    if not trimmed:
        return os.path.normpath(actual_base)

    if trimmed == "~":
        return str(Path.home())

    if trimmed.startswith("~/"):
        return str(Path.home() / trimmed[2:])

    if os.path.isabs(trimmed):
        return os.path.normpath(trimmed)

    return os.path.normpath(os.path.join(actual_base, trimmed))


def to_relative_path(absolute_path: str, base: str | None = None) -> str:
    """Convert an absolute path to relative from cwd (or ``base`` if given).

    Returns the relative path if under the base, otherwise the original
    absolute path to preserve clarity.
    """
    if base is None:
        base = os.getcwd()
    try:
        rel = os.path.relpath(absolute_path, base)
        if rel.startswith(".."):
            return absolute_path
        return rel
    except ValueError:
        return absolute_path


def get_directory_for_path(path_str: str) -> str:
    """Get the directory for a given path.

    If the path is a directory, returns it. If it's a file or doesn't exist,
    returns the parent directory.
    """
    abs_path = expand_path(path_str)

    # Skip filesystem operations for UNC paths to prevent NTLM credential leaks
    if abs_path.startswith("\\\\") or abs_path.startswith("//"):
        return os.path.dirname(abs_path)

    try:
        if os.path.isdir(abs_path):
            return abs_path
    except (OSError, IOError):
        pass

    return os.path.dirname(abs_path)


def contains_path_traversal(path_str: str) -> bool:
    """Check if a path contains directory traversal patterns.

    Returns True if the path contains patterns like ``../`` or ``..\\`` that
    navigate to parent directories.
    """
    return bool(re.search(r"(?:^|[/\\])\.\.(?:[/\\]|$)", path_str))


def normalize_path_for_config_key(path_str: str) -> str:
    """Normalize a path for use as a JSON config key.

    Converts all backslashes to forward slashes for consistent serialization.
    """
    normalized = os.path.normpath(path_str)
    return normalized.replace("\\", "/")


def sanitize_path(path_str: str) -> str:
    """Sanitize a path by removing potentially dangerous components."""
    normalized = os.path.normpath(path_str)
    # Remove any null bytes
    normalized = normalized.replace("\0", "")
    return normalized
