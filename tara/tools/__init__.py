"""TARA Tools Registry.

Exports all tools available to the TARA agent.
"""
from __future__ import annotations

import logging
from typing import List, Any

logger = logging.getLogger(__name__)

# Initialize empty tool list
TARA_TOOLS: List[Any] = []

# =============================================================================
# Import and instantiate tools
# =============================================================================

# System tools
try:
    from .system import SystemStatsTool, DiskStatsTool, TimeTool
    system_stats = SystemStatsTool()
    disk_stats = DiskStatsTool()
    time_tool = TimeTool()
    TARA_TOOLS.extend([system_stats, disk_stats, time_tool])
    logger.debug("Loaded system tools")
except ImportError as e:
    logger.warning("Failed to load system tools: %s", e)

# Desktop tools
try:
    from .desktop import OpenAppTool, CloseAppTool, OpenURLTool, PlayYouTubeTool, CopyTool, PasteTool
    open_app = OpenAppTool()
    close_app = CloseAppTool()
    open_url = OpenURLTool()
    play_youtube = PlayYouTubeTool()
    copy_tool = CopyTool()
    paste_tool = PasteTool()
    TARA_TOOLS.extend([open_app, close_app, open_url, play_youtube, copy_tool, paste_tool])
    logger.debug("Loaded desktop tools")
except ImportError as e:
    logger.warning("Failed to load desktop tools: %s", e)

# Web tools
try:
    from .web import WebSearchTool
    web_search = WebSearchTool()
    TARA_TOOLS.append(web_search)
    logger.debug("Loaded web tools")
except ImportError as e:
    logger.warning("Failed to load web tools: %s", e)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TARA_TOOLS",
    # System
    "SystemStatsTool",
    "DiskStatsTool",
    "TimeTool",
    # Desktop
    "OpenAppTool",
    "CloseAppTool",
    "OpenURLTool",
    "PlayYouTubeTool",
    "CopyTool",
    "PasteTool",
    # Web
    "WebSearchTool",
]


def get_tool_names() -> List[str]:
    """Get names of all available tools."""
    return [tool.name for tool in TARA_TOOLS]


def print_tools() -> None:
    """Print all available TARA tools."""
    print(f"\nTARA Tools ({len(TARA_TOOLS)} available):")
    print("=" * 40)
    for tool in TARA_TOOLS:
        print(f"  - {tool.name}: {tool.description[:60]}...")
    print()
