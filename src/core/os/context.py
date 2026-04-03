"""OS Context shim — public API alias for OSContext.

This file lives inside ``src.core.os`` and provides the canonical
``context`` access path within the submodule::

    from src.core.os.context import OSContext, get_os_context

The implementation lives in ``src.core.os.platform``.
Importers should prefer ``src.core.os`` or ``src.core.os.context``.
"""
from src.core.os.platform import OSContext, get_os_context  # noqa: F401

__all__ = ["OSContext", "get_os_context"]
