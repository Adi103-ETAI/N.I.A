"""Backward-compatible re-export for legacy tool registry imports."""

from src.capabilities.tool_registry import (
    ToolManifest,
    ToolRegistry,
    global_registry,
    get_tool,
    get_tool_manifest,
    get_scope,
    get_all_by_scope,
)

__all__ = [
    "ToolManifest",
    "ToolRegistry",
    "global_registry",
    "get_tool",
    "get_tool_manifest",
    "get_scope",
    "get_all_by_scope",
]
