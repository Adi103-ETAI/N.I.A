"""
TARA 2.0 Tool Suite.

This package contains atomic tool modules for desktop automation.
Tools are auto-discovered by the interface module.

Structure:
    tara/tools/
    ├── interface.py      ← LangChain integration (dynamic loader)
    ├── window_manager.py ← WindowRegistry singleton
    ├── app_launcher.py   ← Process management
    ├── window_ops.py     ← Window control
    ├── uia_ops.py        ← UI Automation (semantic)
    ├── browser_ops.py    ← Playwright browser
    ├── file_ops.py       ← File system (3-tier security)
    ├── input_ops.py      ← Mouse/keyboard (coordinate fallback)
    ├── screen_ops.py     ← Screenshots
    └── system_ops.py     ← Clipboard, stats

Usage:
    # For LangChain/Agent integration
    from tara.tools.interface import get_tara_tools
    tools = get_tara_tools()
    
    # For direct function calls
    from tara.tools.app_launcher import launch_app
    result = launch_app("notepad")
    
    # For shared state
    from tara.tools.window_manager import get_registry
    registry = get_registry()

Plug & Play:
    Add new .py files to this directory with public functions.
    They will be auto-discovered as tools.
"""
from __future__ import annotations

# Primary exports for LangChain integration
from .interface import (
    get_tara_tools,
    get_tool_by_name,
    list_tool_names,
    get_tools_by_category,
    clear_cache,
)

# Registry is always available
from .window_manager import get_registry, WindowRegistry, WindowInfo

__all__ = [
    # Interface
    "get_tara_tools",
    "get_tool_by_name", 
    "list_tool_names",
    "get_tools_by_category",
    "clear_cache",
    # Registry
    "get_registry",
    "WindowRegistry",
    "WindowInfo",
]
