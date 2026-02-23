# src/agents/__init__.py
"""N.I.A. Agent Layer — All Specialist Agents.

The N.I.A. system is composed of specialized agents, each with a distinct
role.  This package provides lazy access to each agent to avoid circular
imports and heavy optional dependencies on startup.

Agents:
    nia      — Supervisor & LangGraph orchestrator (the "General")
    tara     — Tool-use specialist for desktop/system/web automation
    iris     — Vision specialist powered by a multimodal LLM
    nola     — Voice I/O manager (requires audio hardware + dependencies)
    soldiers — Ephemeral Docker-based task agents (the Polyglot Swarm)
"""

# Lazy imports to avoid circular dependency and missing optional deps
__all__ = ['nia', 'tara', 'iris', 'nola']


def __getattr__(name):
    """Lazy import agents on first access."""
    if name == 'nia':
        from src.agents import nia as mod
        return mod
    elif name == 'tara':
        from src.agents import tara as mod
        return mod
    elif name == 'iris':
        from src.agents import iris as mod
        return mod
    elif name == 'nola':
        from src.agents import nola as mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

