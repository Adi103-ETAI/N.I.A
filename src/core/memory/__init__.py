"""src/core/memory — N.I.A. 4-Layer Hybrid Memory System.

Public API::

    from src.core.memory import MemoryManager, get_memory_manager
    from src.core.memory import NamespaceManager, get_namespace_manager
"""
from src.core.memory.manager import MemoryManager, get_memory_manager
from src.core.memory.namespaces import NamespaceManager, get_namespace_manager

__all__ = [
    "MemoryManager",
    "get_memory_manager",
    "NamespaceManager",
    "get_namespace_manager",
]
