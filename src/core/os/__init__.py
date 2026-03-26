"""src.core.os — OS Abstraction Package.

Public API for cross-platform OS context.

Sub-modules:
    platform  — canonical implementation (OSContext class)
    context   — alias/shim for the context API

Re-exports:
    OSContext      — Singleton class with OS detection and directory paths
    get_os_context — Accessor for the global OSContext singleton

Backward-compat shims at root level:
    ``from src.core.context import get_os_context``  also works.
    ``from src.core.platform import get_os_context``  also works.
"""
from src.core.os.platform import OSContext, get_os_context

__all__ = [
    "OSContext",
    "get_os_context",
]
