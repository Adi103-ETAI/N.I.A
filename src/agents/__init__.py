# src/agents/__init__.py
"""Agent modules for N.I.A. system.

Agents:
- nia: Supervisor agent that coordinates all other agents
- tara: Tool execution agent
- iris: Vision/sentry agent
- nola: Voice I/O agent (requires audio dependencies)
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

