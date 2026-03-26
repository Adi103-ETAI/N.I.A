"""NIA Graph Node -- Planner Node.

Wraps the MissionPlanner and Pre-Flight approval gate into a single LangGraph
node.  This node is the graph entry point.

Flow (Sprint 4):
    user_input
      -> MissionPlanner.plan()
      -> run_preflight_approval()
      -> routing decision:
           * Multi-step / deep / agent_spawn  -> "coordinator"
           * Simple single-step               -> legacy direct node
"""
from __future__ import annotations

import logging
from src.core.schema.states import AgentState
from src.core.approval.preflight import run_preflight_approval

logger = logging.getLogger("NIA.Nodes.Planner")

__all__ = ["planner_node"]


def _needs_coordinator(manifest) -> bool:
    """Decide whether a mission should be dispatched to the Coordinator.

    The Coordinator is used for missions that are non-trivial:
      - More than one step in the plan
      - The ``agent_spawn`` scope was approved (multi-agent work)
      - Execution mode is ``"deep"``

    Simple single-step, read-only plans are routed directly to the
    appropriate legacy node (supervisor, tara, etc.) to avoid overhead.
    """
    from src.core.policy.scopes import CapabilityScope

    # Multiple steps -> coordinator
    if len(manifest.steps) > 1:
        return True

    # Agent-spawn scope approved -> coordinator
    if CapabilityScope.AGENT_SPAWN in manifest.approved_scopes:
        return True

    # Deep execution mode -> coordinator
    if manifest.execution_mode == "deep":
        return True

    return False


async def planner_node(state: AgentState) -> AgentState:
    """Mission Planner + Pre-Flight Gate + Coordinator Routing.

    1. Extracts user input from state.
    2. Calls ``MissionPlanner`` to produce a ``MissionManifest``.
    3. Runs the pre-flight approval gate.
    4. Routes to either:
       - ``"coordinator"`` for multi-step / deep / agent_spawn missions
       - A legacy direct node for simple single-step plans

    Returns state with:
        - ``metadata["manifest"]``  -- approved MissionManifest (dict)
        - ``next``                  -- target node name
    """
    from src.agents.nia.planner import MissionPlanner
    from src.core.schema.mission import MissionManifest
    from src.core.policy.scopes import CapabilityScope

    # ── Extract user input ──────────────────────────────────────────────
    user_input = state.get("user_input", "")
    if not user_input:
        from langchain_core.messages import HumanMessage

        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                user_input = content
                break

    if not user_input:
        logger.warning("Planner node received empty input -- defaulting to supervisor")
        return {**state, "next": "supervisor"}

    logger.info("Planning mission for: '%s...'", user_input[:80])

    # ── 1. Generate MissionManifest ─────────────────────────────────────
    planner = MissionPlanner()
    manifest = await planner.plan(user_input)
    if isinstance(manifest, dict):
        # Compatibility path for legacy dict manifests.
        mode = manifest.get("execution_mode", "standard")
        if mode == "quick":
            mode = "fast"
        steps = manifest.get("steps", [])
        mission_steps = []
        for step in steps:
            mission_steps.append(
                {
                    "description": step.get("instruction", step.get("description", "")),
                    "assigned_role": step.get("role", step.get("assigned_role", "coder")),
                    "required_scopes": [CapabilityScope.EXECUTE],
                }
            )
        manifest = MissionManifest.model_validate(
            {
                "mission_id": "legacy-001",
                "intent": user_input,
                "steps": mission_steps,
                "required_scopes": [manifest.get("scope", "read_only")],
                "execution_mode": mode,
            }
        )

    # ── 2. Pre-Flight Approval Gate ─────────────────────────────────────
    approved = await run_preflight_approval(manifest)
    if isinstance(approved, tuple):
        is_approved, _ = approved
        manifest.approved = bool(is_approved)
        if manifest.approved and not manifest.approved_scopes:
            manifest.approved_scopes = manifest.required_scopes[:]
    else:
        manifest = approved

    # ── 3. Cancelled? Short-circuit to supervisor ───────────────────────
    if not manifest.approved:
        logger.info("Mission cancelled at pre-flight gate.")
        return {
            **state,
            "next": "supervisor",
            "metadata": {
                **state.get("metadata", {}),
                "mission_cancelled": True,
            },
        }

    # ── 4. Persist manifest in metadata ─────────────────────────────────
    manifest_dump = manifest.model_dump()
    new_meta = {
        **state.get("metadata", {}),
        "manifest": manifest_dump,
        "mission_manifest": manifest_dump,  # backward compatibility
    }

    # ── 5. Route: coordinator vs. legacy direct node ────────────────────
    if _needs_coordinator(manifest):
        logger.info(
            "Plan approved -> routing to COORDINATOR "
            "(steps=%d, mode=%s)",
            len(manifest.steps),
            manifest.execution_mode,
        )
        return {**state, "next": "coordinator", "metadata": new_meta}

    # Legacy direct routing for simple single-step plans
    first_step_role = manifest.steps[0].assigned_role if manifest.steps else "planner"

    role_to_node = {
        "planner":    "supervisor",
        "researcher": "supervisor",
        "coder":      "tara",
        "reviewer":   "supervisor",
    }
    next_node = role_to_node.get(first_step_role, "supervisor")

    logger.info("Plan approved -> legacy routing to: %s", next_node)
    return {**state, "next": next_node, "metadata": new_meta}
