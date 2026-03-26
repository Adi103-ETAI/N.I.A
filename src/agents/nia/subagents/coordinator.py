"""Coordinator StateGraph — The swarm brain for N.I.A. Sprint 4.

Implements a LangGraph StateGraph that orchestrates parallel subagent
dispatch, result evaluation, and failure-driven reflection/retry loops.

Graph topology::

    [dispatch] --> [evaluate] --> coordinator_router
                                       |-- "dispatch"  --> [dispatch]   (more steps)
                                       |-- "reflect"   --> [reflect]    (retry needed)
                                       |-- END         <-- completed / failed / needs_human

Public API:
    ``run_coordinator(manifest)`` — run a full mission through the swarm.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger("NIA.Coordinator")

# Default timeout (seconds) for roles not in ROLE_TIMEOUTS.
_DEFAULT_TIMEOUT: int = 120

# Maximum retries per step before the step is declared failed.
_MAX_RETRIES: int = 3


# ============================================================================
# Helpers
# ============================================================================

def _role_to_dispatch_target(role: str) -> str:
    """Map a plan-step *assigned_role* to the dispatch target identifier.

    Returns:
        ``"tara"`` or ``"iris"``.  Every non-IRIS role routes through TARA
        because TARA is the universal tool-execution agent.
    """
    if role == "iris":
        return "iris"
    # coder, researcher, reviewer, planner, tara, and anything else -> tara
    return "tara"


def _get_timeout_for_role(role: str) -> int:
    """Return the wall-clock timeout (seconds) for *role*."""
    from src.core.schema.coordinator import ROLE_TIMEOUTS
    return ROLE_TIMEOUTS.get(role, _DEFAULT_TIMEOUT)


def _build_final_output(completed_results: List[dict], intent: str) -> str:
    """Synthesise a human-readable summary from all successful results."""
    successes = [r for r in completed_results if r.get("status") == "success"]
    if not successes:
        return f"Mission '{intent}' completed, but no subagent returned a successful result."

    parts: list[str] = []
    for idx, res in enumerate(successes, 1):
        snippet = (res.get("output") or "(no output)")[:500]
        parts.append(f"  [{idx}] ({res.get('agent_id', '?')}): {snippet}")

    header = f"Mission '{intent}' completed with {len(successes)} successful result(s):\n"
    return header + "\n".join(parts)


def _build_escalation_message(
    objective: str,
    results: List[dict],
    *,
    max_retries_hit: bool = False,
) -> str:
    """Build a human-readable escalation / needs_human message."""
    lines = [f"The coordinator needs human input for: {objective}"]
    if max_retries_hit:
        lines.append("Maximum retries exhausted.")
    for r in results:
        status = r.get("status", "unknown")
        output = (r.get("output") or "")[:200]
        lines.append(f"  - {r.get('agent_id', '?')} [{status}]: {output}")
    return "\n".join(lines)


# ============================================================================
# Node 1 — dispatch_node
# ============================================================================

async def dispatch_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Pick pending steps and fan-out to subagents in parallel.

    Respects ``SwarmLimits`` ceilings on total nodes and tree depth.
    Uses ``asyncio.gather(return_exceptions=True)`` so one failing task
    does not cancel siblings.
    """
    from src.core.schema.coordinator import SwarmLimits, ROLE_TIMEOUTS
    from src.core.schema.mission import MissionManifest, SubagentResult
    from src.core.bus.events import get_event_bus

    limits = SwarmLimits()  # frozen defaults
    pending: List[dict] = list(state.get("pending_steps", []))
    active: List[dict] = list(state.get("active_tasks", []))
    completed: List[dict] = list(state.get("completed_results", []))
    context_log: List[dict] = list(state.get("context_log", []))
    total_spawned: int = state.get("total_nodes_spawned", 0)
    tree_depth: int = state.get("tree_depth", 0)
    mission_dict: dict = state["mission"]

    # ---- Limits gate -------------------------------------------------------
    if total_spawned >= limits.max_total_nodes:
        logger.warning(
            "Swarm node limit reached (%d/%d). Stopping dispatch.",
            total_spawned, limits.max_total_nodes,
        )
        return {
            **state,
            "pending_steps": pending,
            "active_tasks": active,
            "completed_results": completed,
            "status": "completed" if not pending else "failed",
            "final_output": (
                _build_final_output(completed, mission_dict.get("intent", ""))
                if not pending
                else f"Swarm node limit ({limits.max_total_nodes}) reached with "
                     f"{len(pending)} step(s) still pending."
            ),
        }

    if tree_depth >= limits.max_depth:
        logger.warning(
            "Swarm depth limit reached (%d/%d). Stopping dispatch.",
            tree_depth, limits.max_depth,
        )
        return {
            **state,
            "pending_steps": pending,
            "active_tasks": active,
            "completed_results": completed,
            "status": "failed",
            "final_output": (
                f"Swarm depth limit ({limits.max_depth}) reached. "
                f"{len(pending)} step(s) still pending."
            ),
        }

    # ---- Select batch ------------------------------------------------------
    budget_remaining = limits.max_total_nodes - total_spawned
    batch_size = min(limits.max_concurrent_leaves, len(pending), budget_remaining)

    if batch_size == 0:
        # Nothing to dispatch — should not normally happen, but be safe.
        logger.debug("dispatch_node: nothing to dispatch (pending=%d)", len(pending))
        status = "completed" if not active else "executing"
        return {**state, "status": status}

    batch = pending[:batch_size]
    remaining_pending = pending[batch_size:]

    # ---- Build manifest for wrapper calls ----------------------------------
    manifest = MissionManifest(**mission_dict)

    # ---- Dispatch coroutines -----------------------------------------------
    async def _run_step(step: dict) -> SubagentResult:
        """Invoke the right agent wrapper with a per-role timeout."""
        role: str = step.get("assigned_role", "tara")
        objective: str = step.get("description", "")
        timeout: int = _get_timeout_for_role(role)
        target = _role_to_dispatch_target(role)

        try:
            if target == "iris":
                from src.capabilities.agents.invoke_iris import invoke_iris
                coro = invoke_iris(objective, manifest)
            else:
                from src.capabilities.agents.invoke_tara import invoke_tara
                coro = invoke_tara(objective, manifest)

            result = await asyncio.wait_for(coro, timeout=timeout)
            return result

        except asyncio.TimeoutError:
            agent_id = f"{target}-timeout-{uuid.uuid4().hex[:6]}"
            logger.warning(
                "Subagent timed out after %ds for step %s (role=%s)",
                timeout, step.get("step_index"), role,
            )
            return SubagentResult(
                agent_id=agent_id,
                status="stuck",
                output=f"Subagent timed out after {timeout}s.",
                failure_trace=f"asyncio.TimeoutError after {timeout}s for role '{role}'",
            )
        except Exception as exc:
            agent_id = f"{target}-error-{uuid.uuid4().hex[:6]}"
            logger.error(
                "Unexpected error dispatching step %s: %s",
                step.get("step_index"), exc, exc_info=True,
            )
            return SubagentResult(
                agent_id=agent_id,
                status="failed",
                output=f"Dispatch error: {exc}",
                failure_trace=str(exc),
            )

    # Fire all in parallel (return_exceptions=True keeps siblings alive).
    raw_results = await asyncio.gather(
        *(_run_step(s) for s in batch),
        return_exceptions=True,
    )

    # ---- Process gather results -------------------------------------------
    new_active: List[dict] = list(active)
    new_completed: List[dict] = list(completed)

    for step, raw in zip(batch, raw_results):
        if isinstance(raw, BaseException):
            # Unexpected exception that escaped _run_step — treat as failed.
            logger.error("gather exception for step %s: %s", step.get("step_index"), raw)
            result_dict = SubagentResult(
                agent_id=f"error-{uuid.uuid4().hex[:6]}",
                status="failed",
                output=f"Unhandled dispatch exception: {raw}",
                failure_trace=str(raw),
            ).model_dump()
        else:
            result_dict = raw.model_dump()

        # Tag result with step metadata for evaluate_node
        result_dict["_step_index"] = step.get("step_index")
        result_dict["_role"] = step.get("assigned_role", "tara")
        result_dict["_objective"] = step.get("description", "")

        # Results land directly in completed_results for evaluate_node.
        new_completed.append(result_dict)

    new_total_spawned = total_spawned + len(batch)

    # ---- Emit event --------------------------------------------------------
    bus = get_event_bus()
    try:
        await bus.emit("coordinator_dispatch", {
            "mission_id": mission_dict.get("mission_id"),
            "batch_size": len(batch),
            "total_spawned": new_total_spawned,
        })
    except Exception:
        logger.debug("Event bus emit failed (non-critical)", exc_info=True)

    logger.info(
        "Dispatched %d step(s); total_spawned=%d, pending=%d",
        len(batch), new_total_spawned, len(remaining_pending),
    )

    return {
        **state,
        "pending_steps": remaining_pending,
        "active_tasks": new_active,
        "completed_results": new_completed,
        "total_nodes_spawned": new_total_spawned,
        "tree_depth": tree_depth + 1,
        "status": "executing",
    }


# ============================================================================
# Node 2 — evaluate_node
# ============================================================================

async def evaluate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process completed subagent results and decide the next phase.

    For each result not yet evaluated:
        - ``success``            -> archive to context_log.
        - ``failed``             -> queue for reflection if retries remain.
        - ``scope_violation``    -> escalate to human immediately.
        - ``stuck``              -> escalate to human.
        - ``needs_clarification``-> escalate to human.

    After processing, determines the next status:
        - All steps done?  -> ``"completed"``
        - More pending?    -> ``"executing"``
        - Any need retry?  -> ``"reflecting"``
        - Human needed?    -> ``"needs_human"``
    """
    pending: List[dict] = list(state.get("pending_steps", []))
    active: List[dict] = list(state.get("active_tasks", []))
    completed: List[dict] = list(state.get("completed_results", []))
    context_log: List[dict] = list(state.get("context_log", []))
    retry_counts: dict = dict(state.get("retry_counts", {}))
    mission_dict: dict = state["mission"]

    needs_reflection: List[dict] = []
    needs_human: bool = False
    human_question: str | None = state.get("human_question")
    failed_hard: bool = False

    for result in completed:
        status = result.get("status", "")
        step_key = str(result.get("_step_index", "?"))
        objective = result.get("_objective", "")

        if status == "success":
            # Archive successful output to context log.
            context_log.append({
                "agent_id": result.get("agent_id", ""),
                "observation": (result.get("output") or "")[:1000],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step_index": result.get("_step_index"),
            })

        elif status == "failed":
            retries = retry_counts.get(step_key, 0)
            if retries < _MAX_RETRIES:
                needs_reflection.append(result)
            else:
                logger.warning(
                    "Step %s exhausted retries (%d). Marking hard-failed.",
                    step_key, retries,
                )
                failed_hard = True
                human_question = _build_escalation_message(
                    objective, [result], max_retries_hit=True,
                )

        elif status == "scope_violation":
            logger.warning("Scope violation on step %s — escalating.", step_key)
            needs_human = True
            human_question = (
                f"Scope violation during step {step_key}: "
                f"{result.get('output', 'Unknown scope issue')}"
            )

        elif status in ("stuck", "needs_clarification"):
            needs_human = True
            human_question = result.get("output") or (
                f"Subagent is {status} on step {step_key}. "
                "Please provide guidance."
            )

    # ---- Determine next status ---------------------------------------------
    # Clear completed_results so they are not re-evaluated on the next pass.
    # (Evaluated data is preserved in context_log for successes, and
    #  queued into pending_steps for reflections.)
    evaluated_completed: List[dict] = []

    if needs_human or (failed_hard and not needs_reflection):
        next_status = "needs_human" if needs_human else "failed"
    elif needs_reflection:
        next_status = "reflecting"
    elif not pending and not needs_reflection:
        next_status = "completed"
    else:
        next_status = "executing"

    # Build final output when done.
    final_output = state.get("final_output")
    if next_status == "completed":
        final_output = _build_final_output(
            completed, mission_dict.get("intent", ""),
        )
    elif next_status == "failed":
        final_output = _build_escalation_message(
            mission_dict.get("intent", ""),
            completed,
            max_retries_hit=True,
        )

    # Stash reflection candidates in state for reflect_node to pick up.
    # We use a transient key ``_needs_reflection`` to pass them forward.
    return {
        **state,
        "pending_steps": pending,
        "active_tasks": [],  # all tasks evaluated; none active now
        "completed_results": evaluated_completed,
        "context_log": context_log,
        "retry_counts": retry_counts,
        "status": next_status,
        "human_question": human_question,
        "final_output": final_output,
        "_needs_reflection": needs_reflection,
    }


# ============================================================================
# Node 3 — reflect_node
# ============================================================================

async def reflect_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Reformulate failed objectives and re-queue them for dispatch.

    Reads the transient ``_needs_reflection`` list left by evaluate_node,
    calls the LLM-backed ``reflect_and_reformulate`` engine for each, and
    pushes new pending steps with the rewritten objectives.
    """
    from src.agents.nia.subagents.reflect import reflect_and_reformulate

    pending: List[dict] = list(state.get("pending_steps", []))
    retry_counts: dict = dict(state.get("retry_counts", {}))
    needs_reflection: List[dict] = state.get("_needs_reflection", [])

    for result in needs_reflection:
        step_key = str(result.get("_step_index", "?"))
        original_objective = result.get("_objective", "")
        failure_trace = result.get("failure_trace") or result.get("output", "")
        attempt = retry_counts.get(step_key, 0) + 1

        reformulated = await reflect_and_reformulate(
            original_objective=original_objective,
            failure_trace=failure_trace,
            attempt_number=attempt,
        )

        retry_counts[step_key] = attempt

        # Re-queue with the new objective.
        pending.append({
            "step_index": result.get("_step_index"),
            "description": reformulated,
            "required_scopes": result.get("_required_scopes", []),
            "assigned_role": result.get("_role", "tara"),
        })

        logger.info(
            "Reflected step %s (attempt %d): %.100s...",
            step_key, attempt, reformulated,
        )

    return {
        **state,
        "pending_steps": pending,
        "retry_counts": retry_counts,
        "status": "executing",
        "_needs_reflection": [],  # consumed
    }


# ============================================================================
# Router
# ============================================================================

def coordinator_router(state: Dict[str, Any]) -> str:
    """Route the graph based on the current coordinator status.

    Returns:
        The name of the next node, or ``"__end__"`` for terminal states.
    """
    status = state.get("status", "completed")
    if status == "executing":
        return "dispatch"
    if status == "reflecting":
        return "reflect"
    # "completed", "failed", "needs_human" are all terminal.
    return "__end__"


# ============================================================================
# Graph Builder
# ============================================================================

def _build_coordinator_graph():
    """Construct and compile the Coordinator StateGraph.

    Lazy-imports LangGraph so the module can be imported without
    triggering heavy dependency loading at module scope.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready for ``ainvoke()``.
    """
    from langgraph.graph import StateGraph, END
    from src.core.schema.states import CoordinatorState

    graph = StateGraph(CoordinatorState)

    graph.add_node("dispatch", dispatch_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("reflect", reflect_node)

    graph.set_entry_point("dispatch")

    # dispatch always flows into evaluate
    graph.add_edge("dispatch", "evaluate")

    # evaluate branches via the router
    graph.add_conditional_edges(
        "evaluate",
        coordinator_router,
        {
            "dispatch": "dispatch",
            "reflect": "reflect",
            "__end__": END,
        },
    )

    # reflect loops back to dispatch
    graph.add_edge("reflect", "dispatch")

    compiled = graph.compile()
    logger.info("Coordinator StateGraph compiled.")
    return compiled


# ============================================================================
# Public API
# ============================================================================

async def run_coordinator(manifest) -> dict:
    """Run the full coordinator loop for an approved MissionManifest.

    This is the single entry point called by the outer NIA graph (or
    directly for testing).  It:

    1. Creates the initial ``CoordinatorState`` from the manifest.
    2. Builds and compiles the coordinator sub-graph.
    3. Executes the graph to completion via ``ainvoke()``.
    4. Emits a ``coordinator_complete`` event on the bus.
    5. Returns the final state as a plain dict.

    Args:
        manifest: An approved ``MissionManifest`` instance.

    Returns:
        The final ``CoordinatorState`` as a dict.  Key fields:
        - ``status``:  one of completed / failed / needs_human.
        - ``final_output``:  human-readable result summary.
        - ``context_log``:  full observation trail.
        - ``completed_results``:  raw subagent results.
    """
    from src.core.schema.states import create_coordinator_state
    from src.core.bus.events import get_event_bus

    logger.info(
        "Starting coordinator for mission '%s' (%d steps, mode=%s)",
        manifest.mission_id,
        len(manifest.steps),
        manifest.execution_mode,
    )

    # 1. Initial state
    initial_state = create_coordinator_state(manifest)

    # 2. Build graph
    compiled = _build_coordinator_graph()

    # 3. Execute
    try:
        final_state = await compiled.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Coordinator graph execution failed: %s", exc, exc_info=True)
        final_state = {
            **initial_state,
            "status": "failed",
            "final_output": f"Coordinator crashed: {exc}",
        }

    # 4. Emit completion event
    bus = get_event_bus()
    try:
        await bus.emit("coordinator_complete", {
            "mission_id": manifest.mission_id,
            "status": final_state.get("status"),
            "total_nodes_spawned": final_state.get("total_nodes_spawned", 0),
        })
    except Exception:
        logger.debug("coordinator_complete event emit failed (non-critical)", exc_info=True)

    logger.info(
        "Coordinator finished mission '%s' — status=%s, nodes_spawned=%d",
        manifest.mission_id,
        final_state.get("status"),
        final_state.get("total_nodes_spawned", 0),
    )

    return dict(final_state)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "dispatch_node",
    "evaluate_node",
    "reflect_node",
    "coordinator_router",
    "run_coordinator",
]
