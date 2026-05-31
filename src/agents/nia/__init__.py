"""N.I.A - Neural Intelligence Assistant.

The head that listens, speaks, and divides tasks.
OpenHarness is the hands that execute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.nia.nia import NIA
    from agents.nia.core.brain import NIABrain
    from agents.nia.core.personality import Personality
    from agents.nia.core.memory import Memory
    from agents.nia.core.context import Context

__all__ = ["NIA", "NIABrain", "Personality", "Memory", "Context"]


def get_version() -> str:
    return "0.1.0"


def get_name() -> str:
    return "N.I.A"


def create_nia(working_directory: str | None = None) -> "NIA":
    """Create and return a new N.I.A instance."""
    from agents.nia.nia import NIA
    return NIA(working_directory=working_directory)
