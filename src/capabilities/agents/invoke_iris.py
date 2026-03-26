"""invoke_iris — Async tool wrapper for IRIS vision subagent (Sprint 3).

Wraps IRIS's IrisAgent.aprocess() as a callable async function
that returns a structured SubagentResult. The Coordinator (Sprint 4)
calls this via asyncio.gather() for parallel agent dispatch.

This does NOT modify any existing IRIS code. It is a thin adapter layer.
"""
from __future__ import annotations

import logging
import uuid
import traceback

from src.core.schema.mission import MissionManifest, SubagentResult
from src.core.policy.scopes import CapabilityScope
from src.core.policy.engine import enforce_at_runtime, ScopeViolation

logger = logging.getLogger("Capabilities.InvokeIRIS")


async def invoke_iris(
    objective: str,
    manifest: MissionManifest,
    image_path: str | None = None,
) -> SubagentResult:
    """Invoke IRIS as a subagent for a vision/image analysis task.

    Args:
        objective: The vision query for IRIS (e.g. "What's on my screen?").
        manifest: The approved MissionManifest governing this swarm run.
        image_path: Optional path to a specific image file.

    Returns:
        SubagentResult with the vision analysis outcome.
    """
    agent_id = f"iris-{uuid.uuid4().hex[:8]}"
    logger.info(f"👁️ [{agent_id}] Invoking IRIS for: '{objective[:80]}...'")

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

    # ── 2. Instantiate IRIS ──────────────────────────────────────────────
    try:
        from src.agents.iris.agent import IrisAgent
        iris = IrisAgent()
    except ImportError as e:
        logger.error(f"[{agent_id}] IRIS not available: {e}")
        return SubagentResult(
            agent_id=agent_id,
            status="failed",
            output="IRIS agent is not available.",
            failure_trace=traceback.format_exc(),
        )

    if not iris.is_ready:
        logger.warning(f"[{agent_id}] IRIS not initialized (missing API key?)")
        return SubagentResult(
            agent_id=agent_id,
            status="failed",
            output="IRIS agent failed to initialize. Check NVIDIA_API_KEY.",
        )

    # ── 3. Execute via aprocess (non-blocking) ───────────────────────────
    try:
        result = await iris.aprocess(objective, image_path=image_path)

        # aprocess returns either a string or a state dict
        if isinstance(result, str):
            output = result
        elif isinstance(result, dict):
            # Extract from state
            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                output = last.content if hasattr(last, "content") else str(last)
            else:
                output = "Vision analysis completed (no output)."
        else:
            output = str(result)

        logger.info(f"✅ [{agent_id}] IRIS completed: {output[:100]}...")

        return SubagentResult(
            agent_id=agent_id,
            status="success",
            output=output,
            scopes_used=[CapabilityScope.EXECUTE],
        )

    except Exception as e:
        logger.error(f"❌ [{agent_id}] IRIS execution failed: {e}")
        return SubagentResult(
            agent_id=agent_id,
            status="failed",
            output=f"IRIS execution error: {e}",
            failure_trace=traceback.format_exc(),
        )
