"""NIA Graph Node -- Coordinator Node.

Bridges the NIA master graph (AgentState) to the Sprint 4 Coordinator
StateGraph.  Extracts the approved MissionManifest from planner metadata,
invokes ``run_coordinator``, and maps the result back to AgentState fields.

This node is reached when ``planner_node`` determines the mission needs
multi-step coordination (more than one plan step, ``agent_spawn`` scope,
or ``deep`` execution mode).

Flow:
    planner_node  ->  coordinator_node  ->  END
                          |
                          v
                   run_coordinator(manifest)
"""
from __future__ import annotations

import logging
from src.agents.nia.state import AgentState, AGENT_END

logger = logging.getLogger("NIA.Nodes.Coordinator")

__all__ = ["coordinator_node"]


async def coordinator_node(state: AgentState) -> AgentState:
    """Run the Coordinator loop and map results back to AgentState.

    Steps:
        1. Extract ``manifest`` dict from ``state["metadata"]["manifest"]``.
        2. Reconstruct a ``MissionManifest`` Pydantic model.
        3. Call ``run_coordinator(manifest)`` (the Sprint 4 coordinator).
        4. Map coordinator output to AgentState fields:
           - completed   -> final_response + AIMessage + END
           - needs_human -> final_response (question) + AIMessage + END
           - failed      -> final_response (error summary) + END
        5. Store coordinator results in metadata for traceability.

    Returns:
        Updated ``AgentState`` dict with ``messages``, ``final_response``,
        ``subagent_results``, and ``next`` set to ``AGENT_END``.
    """
    # ── Lazy imports (heavy deps, defensive) ───────────────────────────
    try:
        from langchain_core.messages import AIMessage
        _has_messages = True
    except ImportError:
        _has_messages = False
        AIMessage = None  # type: ignore

    try:
        from src.core.schema.mission import MissionManifest
    except ImportError as exc:
        logger.error("MissionManifest schema not available: %s", exc)
        err_text = "Internal error: mission schema unavailable."
        result_state: dict = {**state, "final_response": err_text, "next": AGENT_END}
        if _has_messages:
            result_state["messages"] = [AIMessage(content=err_text)]
        return result_state

    try:
        from src.agents.nia.subagents.coordinator import run_coordinator
    except ImportError as exc:
        logger.error(
            "Coordinator module not available: %s. "
            "Ensure src.agents.nia.subagents.coordinator is installed.",
            exc,
        )
        err_text = (
            "The Coordinator module is not yet available. "
            "This mission requires multi-step coordination that is still being deployed."
        )
        result_state = {**state, "final_response": err_text, "next": AGENT_END}
        if _has_messages:
            result_state["messages"] = [AIMessage(content=err_text)]
        return result_state

    # ── 1. Extract manifest from metadata ──────────────────────────────
    metadata = state.get("metadata", {})
    manifest_dict = metadata.get("manifest")

    if not manifest_dict:
        logger.error("Coordinator node invoked without a manifest in metadata")
        err_text = "Internal error: no mission manifest available for coordination."
        result_state = {**state, "final_response": err_text, "next": AGENT_END}
        if _has_messages:
            result_state["messages"] = [AIMessage(content=err_text)]
        return result_state

    # ── 2. Reconstruct MissionManifest ─────────────────────────────────
    try:
        manifest = MissionManifest(**manifest_dict)
    except Exception as exc:
        logger.error("Failed to reconstruct MissionManifest: %s", exc)
        err_text = f"Internal error: could not load mission manifest ({exc})."
        result_state = {**state, "final_response": err_text, "next": AGENT_END}
        if _has_messages:
            result_state["messages"] = [AIMessage(content=err_text)]
        return result_state

    logger.info(
        "Coordinator starting mission '%s' (%d steps, mode=%s)",
        manifest.mission_id,
        len(manifest.steps),
        manifest.execution_mode,
    )

    # ── 3. Invoke the Coordinator ──────────────────────────────────────
    try:
        coord_result = await run_coordinator(manifest)
    except Exception as exc:
        logger.exception("Coordinator execution failed: %s", exc)
        err_text = f"Mission execution failed: {exc}"
        new_meta = {**metadata, "coordinator_error": str(exc)}
        result_state = {
            **state,
            "final_response": err_text,
            "next": AGENT_END,
            "metadata": new_meta,
        }
        if _has_messages:
            result_state["messages"] = [AIMessage(content=err_text)]
        return result_state

    # ── 4. Map coordinator result into AgentState ──────────────────────
    status = coord_result.get("status", "failed")
    final_output = coord_result.get("final_output", "")
    human_question = coord_result.get("human_question", "")
    completed_results = coord_result.get("completed_results", [])
    context_log = coord_result.get("context_log", [])

    logger.info(
        "Coordinator finished mission '%s' with status='%s' (%d results)",
        manifest.mission_id,
        status,
        len(completed_results),
    )

    # ── 5. Build final response based on status ────────────────────────
    if status == "completed":
        response_text = final_output or "Mission completed successfully."
    elif status == "needs_human":
        response_text = human_question or "I need more information to proceed."
    else:
        # "failed" or unknown status
        response_text = (
            final_output
            or f"Mission '{manifest.mission_id}' coordination failed. "
               "Please try again or rephrase your request."
        )

    # Collect subagent result summaries as strings
    subagent_summaries: list[str] = []
    for r in completed_results:
        if isinstance(r, dict):
            subagent_summaries.append(r.get("output", str(r)))
        else:
            subagent_summaries.append(str(r))

    # Merge with any existing subagent_results
    existing_results = list(state.get("subagent_results", []))
    existing_results.extend(subagent_summaries)

    # Store coordinator artifacts in metadata for traceability
    updated_meta = {
        **metadata,
        "coordinator_status": status,
        "coordinator_steps_completed": len(completed_results),
        "coordinator_completed_results": completed_results,
        "coordinator_context_log": context_log,
    }

    # ── 6. Return updated state ────────────────────────────────────────
    result_state = {
        **state,
        "next": AGENT_END,
        "final_response": response_text,
        "subagent_results": existing_results,
        "metadata": updated_meta,
    }
    if _has_messages:
        result_state["messages"] = [AIMessage(content=response_text)]

    return result_state
