"""Backward-compat shim for OS context imports.

Legacy code may import from ``src.core.platform``. The canonical
implementation now lives in ``src.core.os.platform``.
"""

from src.core.os.platform import OSContext, get_os_context

__all__ = ["OSContext", "get_os_context"]
