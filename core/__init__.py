"""Core NIA public exports.

This module exposes the commonly used core symbols for convenience.
Only functional modules are exported; legacy placeholders have been removed.
"""

__all__ = []

# Memory management (the only existing module)
try:
    from .memory import InMemoryMemory, MemoryManager
    __all__.extend(["InMemoryMemory", "MemoryManager"])
except Exception:
    InMemoryMemory = None
    MemoryManager = None

# Engine (main application)
try:
    from .engine import NIAAssistant
    __all__.append("NIAAssistant")
except Exception:
    NIAAssistant = None

# Health check
try:
    from .health import check_dependencies, print_system_status
    __all__.extend(["check_dependencies", "print_system_status"])
except Exception:
    check_dependencies = None
    print_system_status = None
