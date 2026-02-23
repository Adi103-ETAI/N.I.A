"""src/core/memory — N.I.A. 4-Layer Hybrid Memory System.

Public API (unchanged from src.core.memory):
    from src.core.memory import MemoryManager, get_memory_manager
"""
from src.core.memory.manager import MemoryManager, get_memory_manager

__all__ = ["MemoryManager", "get_memory_manager"]
