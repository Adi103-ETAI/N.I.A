"""Compatibility shim for NIA graph helper utilities.

Reusable implementations moved to ``src.core.utils.graph_utils``.
"""

from src.core.utils.graph_utils import (
    get_vision_keywords,
    get_prompts,
    summarize_oldest,
    asummarize_oldest,
)

__all__ = [
    "get_vision_keywords",
    "get_prompts",
    "summarize_oldest",
    "asummarize_oldest",
]
