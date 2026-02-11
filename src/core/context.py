"""Backward-compatible context module.

This module re-exports OSContext and get_os_context from src.core.platform
to maintain compatibility with imports using 'src.core.context'.
"""
from src.core.platform import OSContext, get_os_context

__all__ = [
    "OSContext",
    "get_os_context",
]
