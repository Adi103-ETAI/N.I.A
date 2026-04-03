"""src.core.os — OS abstraction package.

Public API for cross-platform OS context.

Sub-modules:
    platform  — canonical implementation (OSContext class)
    context   — API companion module

Re-exports:
    OSContext      — singleton class with OS detection and directory paths
    get_os_context — accessor for the global OSContext singleton

Backward compatibility:
    ``src.core.os_context`` remains available through compat aliasing.
"""
from src.core.os.platform import OSContext, get_os_context

__all__ = [
    "OSContext",
    "get_os_context",
]
