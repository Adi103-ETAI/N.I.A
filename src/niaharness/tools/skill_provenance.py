"""Skill write-origin provenance — ContextVar for distinguishing background-review skill writes from foreground user-directed writes.

Ported from Hermes Agent's ``tools/skill_provenance.py`` (78 LOC).

The curator only consolidates/prunes skills it autonomously created via
the background self-improvement review fork. Skills a user asks a
foreground agent to write belong to the user and must never be
auto-curated.

This module exposes a :class:`contextvars.ContextVar` that the
:class:`QueryEngine` sets before each tool loop so tool handlers
(e.g. ``skill_manage`` create) can check whether they are executing
inside the background-review fork.

The signal piggybacks on ``QueryEngine._memory_write_origin``, which is
already set to ``"background_review"`` for review-fork instances and
defaults to ``"assistant_tool"`` for normal (foreground) agents.

Usage::

    from niaharness.tools.skill_provenance import (
        set_current_write_origin,
        reset_current_write_origin,
        get_current_write_origin,
        is_background_review,
    )

    token = set_current_write_origin("background_review")
    try:
        ...  # tool runs here
    finally:
        reset_current_write_origin(token)

    # inside a tool:
    if is_background_review():
        mark_agent_created(skill_name)
"""

from __future__ import annotations

import contextvars
from typing import FrozenSet

logger = __import__("logging").getLogger(__name__)


# ---------------------------------------------------------------------------
# ContextVar — per-thread write origin
# ---------------------------------------------------------------------------

_write_origin: contextvars.ContextVar[str] = contextvars.ContextVar(
    "nia_skill_write_origin",
    default="foreground",
)

# The sentinel value the background review fork uses.
BACKGROUND_REVIEW = "background_review"


def set_current_write_origin(origin: str) -> contextvars.Token[str]:
    """Bind the active write origin to the current context.

    Returns a Token the caller must pass to :func:`reset_current_write_origin`
    in a ``finally`` block.
    """
    return _write_origin.set(origin or "foreground")


def reset_current_write_origin(token: contextvars.Token[str]) -> None:
    """Restore the prior write origin context."""
    _write_origin.reset(token)


def get_current_write_origin() -> str:
    """Return the active write origin.

    Default: ``"foreground"`` — any tool call made by a regular
    (non-review) agent, from the CLI, the gateway, cron, or a subagent.

    ``"background_review"`` — the self-improvement review fork; only
    skills created under this origin should be marked agent-created for
    curator management.
    """
    return _write_origin.get()


def is_background_review() -> bool:
    """Convenience: True iff the current write origin is the background review fork."""
    return get_current_write_origin() == BACKGROUND_REVIEW


# ---------------------------------------------------------------------------
# Read-before-write gate — per-context set of skill file paths the review
# fork has actually read via skill_view in the current review turn.
# ---------------------------------------------------------------------------

_background_review_read_paths: contextvars.ContextVar[FrozenSet[str]] = (
    contextvars.ContextVar(
        "nia_background_review_read_paths",
        default=frozenset(),
    )
)


def mark_background_review_skill_read(path: str) -> None:
    """Record that the review fork has read a skill file.

    Called by ``skill_view`` after returning file content to the model.
    No-ops unless :func:`is_background_review` is True (foreground agents
    are not subject to the read-before-write gate).
    """
    if not is_background_review():
        return
    if not path:
        return
    current = _background_review_read_paths.get()
    _background_review_read_paths.set(current | {path})


def background_review_has_read(path: str) -> bool:
    """Return True if the review fork has read *path* in the current turn."""
    if not path:
        return False
    return path in _background_review_read_paths.get()


def reset_background_review_read_marks() -> None:
    """Clear all read marks for the current review turn.

    Called at the start of each background review run so stale marks
    from a prior review don't carry over.
    """
    _background_review_read_paths.set(frozenset())


__all__ = [
    "BACKGROUND_REVIEW",
    "background_review_has_read",
    "get_current_write_origin",
    "is_background_review",
    "mark_background_review_skill_read",
    "reset_background_review_read_marks",
    "reset_current_write_origin",
    "set_current_write_origin",
]
