# Desktop automation tools (window, screen, UIA, app launching)
"""
Desktop automation subpackage.

Contains tools for window management, screen capture, UI automation, and app launching.
"""
from .window_manager import get_registry, WindowRegistry, WindowInfo

__all__ = [
    "get_registry",
    "WindowRegistry", 
    "WindowInfo",
]
