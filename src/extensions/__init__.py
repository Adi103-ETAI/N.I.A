# src/extensions/__init__.py
"""Extension system for N.I.A."""

from src.extensions.base import BaseExtension
from src.extensions.loader import ExtensionLoader
from src.extensions.compat import enable_compatibility_mode, disable_compatibility_mode

__all__ = [
    "BaseExtension",
    "ExtensionLoader",
    "enable_compatibility_mode",
    "disable_compatibility_mode",
]
