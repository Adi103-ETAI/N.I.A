"""T.A.R.A. - Technical Agent for Reasoning & Analysis.

A specialist agent for NIA that handles:
- System monitoring (CPU, RAM, disk)
- Desktop automation (apps, clipboard)
- Web search (DuckDuckGo)

Usage:
    from tara import TaraAgent
    
    agent = TaraAgent()
    result = agent.run("Check system health")
"""
from __future__ import annotations

# Import agent (tools are discovered dynamically via ToolRegistry)
try:
    from .agent import TaraAgent
    _HAS_AGENT = True
except ImportError:
    TaraAgent = None  # type: ignore
    _HAS_AGENT = False


__version__ = "2.0.0"
__all__ = ["TaraAgent"]
