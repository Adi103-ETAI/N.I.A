"""invoke_tara — Async tool wrapper for TARA subagent (Sprint 3).

Wraps TARA's compiled LangGraph subgraph as a callable async function
that returns a structured SubagentResult. The Coordinator (Sprint 4)
calls this via asyncio.gather() for parallel agent dispatch.

This does NOT modify any existing TARA code. It is a thin adapter layer.
"""
from __future__ import annotations

import logging
import uuid
import traceback

from src.core.schema.mission import MissionManifest, SubagentResult
from src.core.policy.scopes import CapabilityScope
from src.core.policy.engine import enforce_at_runtime, ScopeViolation

logger = logging.getLogger("Capabilities.InvokeTARA")


async def invoke_tara(
    objective: str,
    manifest: MissionManifest,
) -> SubagentResult:
    """Invoke TARA as a subagent to execute a technical/automation task.

    Args:
        objective: The task description for TARA to execute.
        manifest: The approved MissionManifest governing this swarm run.

    Returns:
        SubagentResult with the execution outcome.
    """
    agent_id = f"tara-{uuid.uuid4().hex[:8]}"
    logger.info(f"🛠️ [{agent_id}] Invoking TARA for: '{objective[:80]}...'")

    # ── 1. Policy gate ────────────────────────────────────────────────────
    try:
        enforce_at_runtime(CapabilityScope.EXECUTE, manifest)
    except ScopeViolation as e:
        logger.warning(f"[{agent_id}] Scope violation: {e}")
        return SubagentResult(
            agent_id=agent_id,
            status="scope_violation",
            output=str(e),
            failure_trace=str(e),
        )

    # ── 2. Lazy-load compiled TARA subgraph ──────────────────────────────
    try:
        from src.agents.tara.graph.workflow import get_tara_subgraph
        tara_app = get_tara_subgraph()
    except ImportError as e:
        logger.error(f"[{agent_id}] TARA not available: {e}")
        return SubagentResult(
            agent_id=agent_id,
            status="failed",
            output="TARA subgraph is not available.",
            failure_trace=traceback.format_exc(),
        )

    # ── 3. Build TARA input state ────────────────────────────────────────
    from langchain_core.messages import HumanMessage

    tara_input = {
        "messages": [HumanMessage(content=objective)],
        "user_goal": objective,
        "screen_context": None,
        "active_app": None,
        "clipboard": None,
        "last_error": None,
        "tool_calls_pending": False,
        "iteration_count": 0,
        "final_response": None,
        "metadata": {
            "mission_id": manifest.mission_id,
            "parent_agent": "coordinator",
        },
    }

    # ── 4. Execute ───────────────────────────────────────────────────────
    try:
        result = await tara_app.ainvoke(tara_input)

        final_response = result.get("final_response", "")

        # Fallback: extract from last AI message
        if not final_response:
            for msg in reversed(result.get("messages", [])):
                if getattr(msg, "type", None) == "ai":
                    final_response = msg.content if hasattr(msg, "content") else str(msg)
                    break

        final_response = final_response or "Task completed (no explicit response)."

        logger.info(f"✅ [{agent_id}] TARA completed: {final_response[:100]}...")

        return SubagentResult(
            agent_id=agent_id,
            status="success",
            output=final_response,
            scopes_used=[CapabilityScope.EXECUTE],
        )

    except Exception as e:
        logger.error(f"❌ [{agent_id}] TARA execution failed: {e}")
        return SubagentResult(
            agent_id=agent_id,
            status="failed",
            output=f"TARA execution error: {e}",
            failure_trace=traceback.format_exc(),
        )
