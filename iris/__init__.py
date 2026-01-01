"""IRIS Package - Intelligent Recognition & Image System.

Vision specialist agent for NIA.
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


__all__ = [
    "IrisAgent",
    "run_iris_agent",
    "capture_screen",
    "capture_screen_raw",
    "start_sentry",
    "stop_sentry",
    "is_sentry_running",
]
