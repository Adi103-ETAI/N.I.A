"""Core NIA public exports.

This module exposes the commonly used core symbols for convenience.

IMPORTANT: All imports are LAZY to enable fast startup.
Heavy modules (memory, engine) are imported on-demand, not at package import.
"""
from typing import TYPE_CHECKING

# 🌊 LAZY LOADING: Only declare __all__ with eventual exports
__all__ = [
    "MemoryManager",
    "NIAAssistant", 
    "check_dependencies",
    "print_system_status",
]

# =============================================================================
# TYPE_CHECKING Block: IDE-only imports (no runtime cost)
# =============================================================================
if TYPE_CHECKING:
    from .memory import MemoryManager
    from .engine import NIAAssistant
    from .health import check_dependencies, print_system_status


# =============================================================================
# Lazy Import Functions (actual runtime access)
# =============================================================================

def __getattr__(name: str):
    """Lazy loading of heavy modules on first access.
    
    This enables:
    - Fast `from core.logger import setup_logger` (no memory/engine loaded)
    - `from core import MemoryManager` still works (loads on access)
    """
    if name == "MemoryManager":
        from .memory import MemoryManager
        return MemoryManager
    elif name == "NIAAssistant":
        from .engine import NIAAssistant
        return NIAAssistant
    elif name == "check_dependencies":
        from .health import check_dependencies
        return check_dependencies
    elif name == "print_system_status":
        from .health import print_system_status
        return print_system_status
    raise AttributeError(f"module 'core' has no attribute '{name}'")

