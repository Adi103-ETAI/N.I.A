"""Coordinator Schema — Pydantic models for the subagent coordination layer.

Defines the data contracts exchanged between the Coordinator graph,
the AsyncEventBus, and individual subagent invocations during a mission.

Models:
    ContextObservation  — emitted by subagents onto the event bus so the
                          Coordinator (and sibling agents) can build shared
                          situational awareness.
    BudgetExtensionRequest — raised when a subagent needs more steps than
                             originally budgeted in the MissionManifest.
    SwarmLimits         — hard ceilings on tree depth, total nodes, and
                          concurrency to prevent runaway spawning.

Constants:
    ROLE_TIMEOUTS       — per-role wall-clock timeout in seconds.
"""
from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


# =============================================================================
# ContextObservation
# =============================================================================

class ContextObservation(BaseModel):
    """An observation emitted by a subagent onto the event bus.

    The Coordinator appends these to the shared ``context_log`` so that
    sibling subagents and the Reflector can leverage cross-agent insights
    without direct coupling.
    """

    agent_id: str = Field(
        description="Identifier of the subagent that produced this observation.",
    )
    observation: str = Field(
        description="Free-text summary of what the subagent observed or learned.",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the observation was created.",
    )
    relevance_tags: List[str] = Field(
        default_factory=list,
        description="Tags indicating which plan steps or topics this observation relates to.",
    )


# =============================================================================
# BudgetExtensionRequest
# =============================================================================

class BudgetExtensionRequest(BaseModel):
    """Request for additional execution budget from a running subagent.

    When a subagent determines that the remaining step budget is
    insufficient to complete its objective, it raises this request.
    The Coordinator's Reflector node evaluates the justification
    and either grants the extension or escalates to the user.
    """

    agent_id: str = Field(
        description="Identifier of the subagent requesting more budget.",
    )
    current_step: int = Field(
        description="Zero-based index of the plan step currently being executed.",
    )
    steps_requested: int = Field(
        description="Number of additional steps the subagent is requesting.",
    )
    justification: str = Field(
        description="Explanation of why additional steps are needed.",
    )
    artifacts_produced_so_far: List[str] = Field(
        default_factory=list,
        description="Artifacts already produced, proving forward progress.",
    )
    tools_called_so_far: List[str] = Field(
        default_factory=list,
        description="Tools invoked so far during this subagent run.",
    )


# =============================================================================
# SwarmLimits
# =============================================================================

class SwarmLimits(BaseModel):
    """Hard ceilings on the subagent swarm to prevent runaway spawning.

    Treated as a frozen configuration object — instantiate once per
    mission and pass into the Coordinator state factory.
    """

    model_config = {"frozen": True}

    max_depth: int = Field(
        default=3,
        description="Maximum depth of the subagent tree (root = 0).",
    )
    max_total_nodes: int = Field(
        default=10,
        description="Maximum total subagent invocations across the entire mission.",
    )
    max_concurrent_leaves: int = Field(
        default=4,
        description="Maximum number of subagents executing in parallel at any time.",
    )


# =============================================================================
# Role Timeouts (seconds)
# =============================================================================

ROLE_TIMEOUTS: dict[str, int] = {
    "researcher": 30,
    "coder": 300,
    "reviewer": 60,
    "tara": 120,
    "iris": 45,
}
"""Per-role wall-clock timeout in seconds.

If a subagent exceeds its role timeout the Coordinator marks it as
``stuck`` and triggers the retry/escalation path.
"""


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ContextObservation",
    "BudgetExtensionRequest",
    "SwarmLimits",
    "ROLE_TIMEOUTS",
]
