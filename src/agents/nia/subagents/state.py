"""Coordinator State — LangGraph TypedDict for the subagent orchestration graph.

This module defines ``CoordinatorState``, the data-flow state used by the
Coordinator's LangGraph StateGraph.  Unlike the conversation-oriented
``AgentState`` (which uses the ``add_messages`` reducer), the Coordinator
operates as a *data-flow* graph: each field is a plain value or list that
nodes read and overwrite explicitly.

The state tracks:
    - The active MissionManifest (serialised as a dict).
    - Pending, active, and completed subagent task records.
    - A shared context log fed by event-bus observations.
    - Budget / retry bookkeeping.
    - Swarm topology counters (depth, total nodes).
    - Human-in-the-loop escalation fields.

Factory:
    ``create_coordinator_state(manifest)`` builds a ready-to-run state dict
    from an approved ``MissionManifest``.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from src.core.schema.mission import MissionManifest


# =============================================================================
# CoordinatorState TypedDict
# =============================================================================

class CoordinatorState(TypedDict, total=False):
    """LangGraph state for the Coordinator subagent-orchestration graph.

    Attributes:
        mission:
            Serialised MissionManifest (via ``model_dump()``).
        pending_steps:
            Plan steps not yet dispatched to a subagent.
        active_tasks:
            Currently running subagent tasks.  Each entry contains
            ``agent_id``, ``role``, ``objective``, and ``step_index``.
        completed_results:
            Accumulated SubagentResult dicts returned by finished subagents.
        context_log:
            Append-only list of ContextObservation dicts shared across
            the swarm via the event bus.
        retry_counts:
            Maps a step index (as string key) to its current retry count.
        budget_extensions:
            Approved BudgetExtensionRequest records for audit trail.
        tree_depth:
            Current depth of the subagent tree (root Coordinator = 0).
        total_nodes_spawned:
            Running total of subagent invocations in this mission.
        status:
            High-level phase the Coordinator is currently in.
        human_question:
            Question to surface to the user when ``status`` is
            ``"needs_human"``.
        final_output:
            Consolidated result string to return to the user once the
            mission reaches ``"completed"`` status.
    """

    mission: dict
    pending_steps: List[dict]
    active_tasks: List[dict]
    completed_results: List[dict]
    context_log: List[dict]
    retry_counts: dict
    budget_extensions: List[dict]
    tree_depth: int
    total_nodes_spawned: int
    status: Literal[
        "planning",
        "executing",
        "reflecting",
        "completed",
        "failed",
        "needs_human",
    ]
    human_question: Optional[str]
    final_output: Optional[str]


# =============================================================================
# State Factory
# =============================================================================

def create_coordinator_state(manifest: MissionManifest) -> dict:
    """Build an initial CoordinatorState dict from an approved MissionManifest.

    Each ``PlanStep`` in the manifest is serialised and placed into
    ``pending_steps`` so the Dispatcher node can consume them one by one
    (or in parallel batches).

    Args:
        manifest: The approved MissionManifest produced by the Planner.

    Returns:
        A plain dict conforming to ``CoordinatorState``, ready to be
        passed as the initial state to ``StateGraph.invoke()``.
    """
    manifest_dict = manifest.model_dump()

    pending = [
        {
            "step_index": idx,
            "description": step.description,
            "required_scopes": [s.value for s in step.required_scopes],
            "assigned_role": step.assigned_role,
        }
        for idx, step in enumerate(manifest.steps)
    ]

    return {
        "mission": manifest_dict,
        "pending_steps": pending,
        "active_tasks": [],
        "completed_results": [],
        "context_log": [],
        "retry_counts": {},
        "budget_extensions": [],
        "tree_depth": 0,
        "total_nodes_spawned": 0,
        "status": "planning",
        "human_question": None,
        "final_output": None,
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CoordinatorState",
    "create_coordinator_state",
]
