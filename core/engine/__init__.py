"""N.I.A. Engine Package.

Modular package containing the NIAAssistant orchestrator.

This package provides backward compatibility for existing imports:
    from core.engine import NIAAssistant  # Works!
"""
from .system import NIAAssistant

# Re-export module-level items for backward compatibility
from .system import _load_engine_config, _ENGINE_CONFIG, _COMMANDS, _HELP_TEXT

__all__ = [
    "NIAAssistant",
    "_load_engine_config",
    "_ENGINE_CONFIG", 
    "_COMMANDS",
    "_HELP_TEXT",
]
