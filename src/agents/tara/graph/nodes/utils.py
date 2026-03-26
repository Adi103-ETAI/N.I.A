"""Compatibility shim for TARA JSON parsing helpers.

Reusable implementations moved to ``src.core.utils.text_utils``.
"""

from src.core.utils.text_utils import (
    _sanitize_json_string,
    _extract_json_objects,
    _parse_llama_tool_calls,
)

__all__ = [
    "_sanitize_json_string",
    "_extract_json_objects",
    "_parse_llama_tool_calls",
]
