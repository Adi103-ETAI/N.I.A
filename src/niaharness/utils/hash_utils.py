"""Hashing utilities.

Provides functions for computing file and content hashes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional


def hash_content(content: str | bytes, algorithm: str = "sha256") -> str:
    """Compute a hash of the given content.

    Args:
        content: The content to hash (string or bytes).
        algorithm: The hash algorithm to use (default: sha256).

    Returns:
        The hex digest of the hash.
    """
    h = hashlib.new(algorithm)
    if isinstance(content, str):
        h.update(content.encode("utf-8"))
    else:
        h.update(content)
    return h.hexdigest()


def hash_file(file_path: str, algorithm: str = "sha256") -> Optional[str]:
    """Compute a hash of a file's contents.

    Args:
        file_path: Path to the file to hash.
        algorithm: The hash algorithm to use (default: sha256).

    Returns:
        The hex digest of the hash, or None if the file cannot be read.
    """
    try:
        h = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError):
        return None


def hash_string(content: str, algorithm: str = "sha256") -> str:
    """Compute a hash of a string (convenience wrapper).

    Args:
        content: The string to hash.
        algorithm: The hash algorithm to use (default: sha256).

    Returns:
        The hex digest of the hash.
    """
    return hash_content(content.encode("utf-8"), algorithm)


def short_hash(content: str | bytes, length: int = 8) -> str:
    """Compute a short hash of the given content.

    Useful for generating unique IDs or abbreviated references.
    """
    return hash_content(content)[:length]
