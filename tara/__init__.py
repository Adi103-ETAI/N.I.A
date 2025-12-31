"""T.A.R.A. - Technical Agent for Reasoning & Analysis.

A specialist agent for NIA that handles:
- System monitoring (CPU, RAM, disk)
- Desktop automation (apps, clipboard)
- Web search (DuckDuckGo)

Usage:
    from tara import TaraAgent, TARA_TOOLS
    
    agent = TaraAgent()
    result = agent.run("Check system health")
"""
from __future__ import annotations

# Import tools
try:
    from .tools import TARA_TOOLS
    _HAS_TOOLS = True
except ImportError:
    TARA_TOOLS = []
    _HAS_TOOLS = False

# Import agent
try:
    from .agent import TaraAgent
    _HAS_AGENT = True
except ImportError:
    TaraAgent = None  # type: ignore
    _HAS_AGENT = False


__version__ = "1.0.0"
__all__ = ["TaraAgent", "TARA_TOOLS"]
