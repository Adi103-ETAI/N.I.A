"""NIA identity — personality, memory, context awareness, and orchestrator.

This package was moved from ``agents/nia/`` to unify the codebase
under ``niaharness/``. NIA's personality (JARVIS-like tone), memory
(preferences, facts, patterns), context (time-of-day, environment,
session tracking), and the NIA orchestrator class all live here.
"""

from niaharness.identity.personality import Mood, Personality, PersonalityConfig
from niaharness.identity.memory import Memory, MemoryEntry
from niaharness.identity.context import Context, Environment, SessionContext, TimeOfDay, UserState
from niaharness.identity.nia import NIA

__all__ = [
    "Context",
    "Environment",
    "Mood",
    "Memory",
    "MemoryEntry",
    "NIA",
    "Personality",
    "PersonalityConfig",
    "SessionContext",
    "TimeOfDay",
    "UserState",
]
