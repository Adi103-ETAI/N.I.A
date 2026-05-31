"""JSON utilities.

Provides JSON reading with BOM stripping and safe parsing.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, TypeVar

T = TypeVar("T")

UTF8_BOM = "\ufeff"


def strip_bom(content: str) -> str:
    """Strip UTF-8 BOM from content.

    PowerShell 5.x writes UTF-8 with BOM by default. This strips the BOM
    to prevent JSON parsing errors.
    """
    if content.startswith(UTF8_BOM):
        return content[1:]
    return content


def safe_parse_json(
    content: str,
    default: Optional[T] = None,
) -> Any:
    """Safely parse JSON content with BOM stripping.

    Returns the parsed JSON or the default value if parsing fails.
    """
    try:
        return json.loads(strip_bom(content))
    except (json.JSONDecodeError, ValueError):
        return default


def parse_json_strict(content: str) -> Any:
    """Parse JSON content strictly, raising on errors.

    Strips BOM before parsing. Raises json.JSONDecodeError on failure.
    """
    return json.loads(strip_bom(content))


def to_json(obj: Any, indent: Optional[int] = None, sort_keys: bool = False) -> str:
    """Serialize an object to a JSON string.

    Uses compact separators by default (no extra whitespace).
    """
    if indent is not None:
        return json.dumps(obj, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
    return json.dumps(
        obj, separators=(",", ":"), sort_keys=sort_keys, ensure_ascii=False
    )


def pretty_json(obj: Any, indent: int = 2) -> str:
    """Serialize an object to a pretty-printed JSON string."""
    return to_json(obj, indent=indent, sort_keys=False)


def load_json_file(file_path: str, default: Optional[T] = None) -> Any:
    """Load and parse a JSON file.

    Returns the parsed content or the default value if the file cannot be
    read or parsed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return safe_parse_json(f.read(), default)
    except (OSError, IOError):
        return default


def save_json_file(
    file_path: str,
    obj: Any,
    indent: Optional[int] = None,
    sort_keys: bool = False,
) -> bool:
    """Save an object to a JSON file.

    Returns True on success, False on failure.
    """
    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            if indent is not None:
                json.dump(
                    obj,
                    f,
                    indent=indent,
                    sort_keys=sort_keys,
                    ensure_ascii=False,
                )
            else:
                json.dump(
                    obj,
                    f,
                    separators=(",", ":"),
                    sort_keys=sort_keys,
                    ensure_ascii=False,
                )
        return True
    except (OSError, IOError):
        return False


import os  # noqa: E402 (needed for save_json_file)
