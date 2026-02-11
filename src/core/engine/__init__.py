"""N.I.A. Engine Package.

Modular package containing the NIAAssistant orchestrator.

This package provides backward compatibility for existing imports:
    from src.core.engine import NIAAssistant  # Works!
"""
from .orchestrator import NIAAssistant

__all__ = [
    "NIAAssistant",
]
