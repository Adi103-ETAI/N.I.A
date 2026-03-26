"""Backward-compat shim for OS context imports.

Legacy code may import from ``src.core.context``. The canonical
implementation now lives in ``src.core.os.context``.
"""

from src.core.os.context import OSContext, get_os_context

__all__ = ["OSContext", "get_os_context"]
