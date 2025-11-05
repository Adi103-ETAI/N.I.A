"""Core NIA public exports.

This module exposes the commonly used core symbols so callers can write
`from core import ToolManager, BaseTool` instead of importing deep paths.

Note: imports are guarded where reasonable to avoid heavy optional
dependencies during test discovery.
"""



__all__ = []

# Core public exports (best-effort import to avoid hard failures at import time)
try:
	from .tool_manager import ToolManager  # primary tool manager implementation
	__all__.append("ToolManager")
except Exception:
	ToolManager = None

try:
	from .base_tool import BaseTool
	__all__.append("BaseTool")
except Exception:
	BaseTool = None

try:
	from .brain import CognitiveLoop
	__all__.append("CognitiveLoop")
except Exception:
	CognitiveLoop = None

try:
	from .memory import InMemoryMemory, MemoryManager
	__all__.extend(["InMemoryMemory", "MemoryManager"])
except Exception:
	InMemoryMemory = None
	MemoryManager = None

try:
	# expose tools package for convenience
	from . import tools as tools
	__all__.append("tools")
except Exception:
	tools = None


