"""N.I.A — Neural Intelligence Assistant.

NIA is the agent. niaharness is its runtime (tools, permissions, hooks, MCP).
NIA owns: identity (SOUL.md), memory, personality, proactive review.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.nia.nia import NIA
    from agents.nia.core.personality import Personality
    from agents.nia.core.memory import Memory
    from agents.nia.core.context import Context

__all__ = ["NIA", "Personality", "Memory", "Context"]


def get_version() -> str:
    return "0.1.0"


def get_name() -> str:
    return "N.I.A"


def create_nia(working_directory: str | None = None) -> "NIA":
    """Create and return a new N.I.A instance."""
    from agents.nia.nia import NIA
    return NIA(working_directory=working_directory)
