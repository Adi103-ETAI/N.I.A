"""OS Context shim — public API alias for OSContext.

This file lives inside ``src.core.os`` and provides the canonical
``context`` access path within the submodule::

    from src.core.os.context import OSContext, get_os_context

The implementation lives in ``src.core.os.platform``.
The root-level ``src.core.context`` is a backward-compat shim that
points here, so all existing imports continue to work unchanged.
"""
from src.core.os.platform import OSContext, get_os_context  # noqa: F401

__all__ = ["OSContext", "get_os_context"]
