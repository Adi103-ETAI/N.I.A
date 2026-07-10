"""Memory exports."""

from niaharness.memory.manager import (
    MemoryManager,
    StreamingContextScrubber,
    add_memory_entry,
    build_memory_context_block,
    get_memory_manager,
    list_memory_files,
    remove_memory_entry,
    reset_memory_manager,
    sanitize_context,
)
from niaharness.memory.memdir import load_memory_prompt
from niaharness.memory.paths import get_memory_entrypoint, get_project_memory_dir
from niaharness.memory.provider import MemoryProvider
from niaharness.memory.scan import scan_memory_files
from niaharness.memory.search import find_relevant_memories
from niaharness.memory.store import (
    BuiltinJsonMemoryProvider,
    DriftReport,
    MemoryEntry,
    MemoryProviderToolAdapter,
    MemoryStore,
    WriteGate,
    inject_memory_provider_tools,
    initialize_default_memory_manager,
    run_memory_setup_wizard,
)
from niaharness.memory.threat_patterns import first_threat_message, scan_for_threats

__all__ = [
    "BuiltinJsonMemoryProvider",
    "DriftReport",
    "MemoryEntry",
    "MemoryManager",
    "MemoryProvider",
    "MemoryProviderToolAdapter",
    "MemoryStore",
    "StreamingContextScrubber",
    "WriteGate",
    "add_memory_entry",
    "build_memory_context_block",
    "find_relevant_memories",
    "first_threat_message",
    "get_memory_entrypoint",
    "get_memory_manager",
    "get_project_memory_dir",
    "inject_memory_provider_tools",
    "initialize_default_memory_manager",
    "list_memory_files",
    "load_memory_prompt",
    "remove_memory_entry",
    "reset_memory_manager",
    "run_memory_setup_wizard",
    "sanitize_context",
    "scan_for_threats",
    "scan_memory_files",
]
