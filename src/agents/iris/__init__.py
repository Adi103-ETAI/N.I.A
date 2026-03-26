"""I.R.I.S. - Intelligent Recognition & Image System.

Vision specialist agent for NIA providing screen analysis and visual understanding.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        IRIS System                              │
    │                                                                 │
    │  [Screenshot] → [Vision LLM] → [Analysis/OCR/UI Detection]      │
    │                      │                                          │
    │          ┌───────────┼───────────┐                              │
    │          ▼           ▼           ▼                              │
    │     [Describe]   [Find UI]   [Extract Text]                     │
    │      Screen      Elements      (OCR)                            │
    └─────────────────────────────────────────────────────────────────┘

Components:
    - IrisAgent: Main vision agent with tool bindings
    - capture_screen: Take screenshot and return base64
    - capture_screen_raw: Take screenshot and return PIL Image
    - Sentry: Background screen monitoring (start_sentry/stop_sentry)

Usage:
    # Via NIA (recommended):
    NIA routes vision queries to IRIS automatically
    
    # Direct usage:
    from src.agents.iris import IrisAgent, capture_screen
    
    agent = IrisAgent()
    if agent.is_ready:
        result = agent.analyze("What's on my screen?")

Version: 4.0.0
"""
from __future__ import annotations

# Import agent
try:
    from .agent import IrisAgent, run_iris_agent
    _HAS_AGENT = True
except ImportError:
    _HAS_AGENT = False
    IrisAgent = None  # type: ignore
    run_iris_agent = None  # type: ignore

# Import tools
try:
    from .tools import capture_screen, capture_screen_raw
    _HAS_TOOLS = True
except ImportError:
    _HAS_TOOLS = False
    capture_screen = None  # type: ignore
    capture_screen_raw = None  # type: ignore

# Import sentry
try:
    from .sentry import start_sentry, stop_sentry, is_sentry_running
    _HAS_SENTRY = True
except ImportError:
    _HAS_SENTRY = False
    start_sentry = None  # type: ignore
    stop_sentry = None  # type: ignore
    is_sentry_running = None  # type: ignore


__version__ = "3.1.0"

__all__ = [
    "IrisAgent",
    "run_iris_agent",
    "capture_screen",
    "capture_screen_raw",
    "start_sentry",
    "stop_sentry",
    "is_sentry_running",
]
