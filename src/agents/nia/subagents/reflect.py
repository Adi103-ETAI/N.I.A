"""Reflect/Reformulate Engine — Sprint 4.

When a subagent fails the Coordinator calls this engine BEFORE retrying.
It uses an LLM to analyse the failure and produce a reformulated objective
that avoids the failure mode.  The subagent then receives this better
objective and retries with a clean state.

Public API:
    reflect_and_reformulate   — LLM-driven failure analysis + objective rewrite.
    evaluate_budget_extension — Heuristic gate for extra execution steps.
    build_escalation_message  — Human-readable escalation when the system is stuck.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from src.core.config.prompts import load_prompt

if TYPE_CHECKING:
    from src.core.schema.coordinator import BudgetExtensionRequest

logger = logging.getLogger("NIA.Coordinator.Reflect")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_REFLECT_SYSTEM = load_prompt(
    "reflect_system",
    fallback=(
        "You are a failure-analysis specialist inside an autonomous AI framework.\n\n"
        "A subagent was given an objective and failed. Your job:\n"
        "1. Analyse the failure trace and identify the specific failure mode.\n"
        "2. Reformulate the original objective so the next attempt avoids that failure mode.\n"
        "3. Be more specific and concrete than the original objective.\n"
        "4. Do NOT change the goal — only change the approach.\n"
        "5. Return ONLY the reformulated objective as a single plain-text string — no explanation, no markdown, no JSON, no preamble."
    ),
)
_HIGH_ATTEMPT_ADDENDUM = load_prompt(
    "reflect_high_attempt",
    fallback=(
        "IMPORTANT: This is already attempt {attempt}. Previous reformulations did not resolve the problem. "
        "You MUST suggest a fundamentally different approach — change the strategy, use alternative tools, "
        "or decompose the task into smaller pieces. Do NOT simply rephrase the same plan."
    ),
)
_REFLECT_USER = load_prompt(
    "reflect_user",
    fallback=(
        "Original objective:\n{objective}\n\n"
        "Failure trace:\n{trace}\n\n"
        "Attempt number: {attempt}\n"
        "{context_section}\n"
        "Produce the reformulated objective now."
    ),
)
_CONTEXT_SECTION = load_prompt(
    "reflect_context_section",
    fallback="Relevant context observations from sibling agents:\n{observations}",
)

# ---------------------------------------------------------------------------
# LLM accessor (lazy, mirrors MissionPlanner pattern)
# ---------------------------------------------------------------------------

_llm_instance = None


def _get_llm():
    """Lazy-load an LLM for reformulation (slightly creative temperature)."""
    global _llm_instance
    if _llm_instance is None:
        from src.models.manager import get_smart_model
        _llm_instance = get_smart_model(temperature=0.2)
    return _llm_instance


# ---------------------------------------------------------------------------
# reflect_and_reformulate
# ---------------------------------------------------------------------------

async def reflect_and_reformulate(
    original_objective: str,
    failure_trace: str,
    attempt_number: int,
    context_log: list[dict] | None = None,
) -> str:
    """Analyse a subagent failure and produce a reformulated objective.

    Uses an LLM to inspect the failure trace, identify the root cause, and
    rewrite the objective so the next retry avoids repeating the same failure
    mode.  On attempt 3+ the prompt pushes for a fundamentally different
    strategy.

    Args:
        original_objective: The objective string that was given to the
            subagent before it failed.
        failure_trace: The error / stack-trace returned by the subagent.
        attempt_number: How many times execution has been attempted so far
            (including the one that just failed).
        context_log: Optional list of ``ContextObservation``-style dicts
            from the shared coordinator state.  When provided the most
            recent observations are included in the LLM prompt so the
            reformulation can leverage cross-agent insights.

    Returns:
        A reformulated objective string ready for the next retry.  If the
        LLM call itself fails, the original objective is returned unchanged
        (graceful fallback).
    """
    logger.info(
        "Reflecting on failure for attempt %d — objective: %.100s...",
        attempt_number,
        original_objective,
    )

    # -- Build system prompt --------------------------------------------------
    system_text = _REFLECT_SYSTEM
    if attempt_number >= 3:
        system_text += _HIGH_ATTEMPT_ADDENDUM.format(attempt=attempt_number)

    # -- Build context section from context_log if available ------------------
    context_section = ""
    if context_log:
        # Take the last 5 observations to keep the prompt focused.
        recent = context_log[-5:]
        observations = "\n".join(
            f"- [{obs.get('agent_id', '?')}] {obs.get('observation', '')}"
            for obs in recent
        )
        context_section = _CONTEXT_SECTION.format(observations=observations)

    user_text = _REFLECT_USER.format(
        objective=original_objective,
        trace=failure_trace,
        attempt=attempt_number,
        context_section=context_section,
    )

    # -- Call LLM -------------------------------------------------------------
    try:
        llm = _get_llm()
        response = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        reformulated = response.content.strip()

        if reformulated:
            logger.info(
                "Reformulated objective (attempt %d): %.120s...",
                attempt_number,
                reformulated,
            )
            return reformulated

        logger.warning(
            "LLM returned empty reformulation on attempt %d — returning original.",
            attempt_number,
        )
    except Exception:
        logger.warning(
            "LLM reflection failed on attempt %d — returning original objective.",
            attempt_number,
            exc_info=True,
        )

    # Graceful fallback: return the original objective unchanged.
    return original_objective


# ---------------------------------------------------------------------------
# evaluate_budget_extension
# ---------------------------------------------------------------------------

async def evaluate_budget_extension(
    request: BudgetExtensionRequest,
    max_extra_steps: int = 3,
) -> bool:
    """Decide whether to grant a subagent's request for more execution steps.

    Pure heuristic logic -- no LLM call required.

    Decision rules (evaluated in order):
        1. If no artifacts produced and ``current_step >= 3`` -> deny
           (no meaningful progress).
        2. If ``steps_requested > max_extra_steps`` -> deny (too greedy).
        3. If artifacts have been produced -> grant (making progress).
        4. Otherwise -> grant (benefit of the doubt).

    Args:
        request: The budget extension request from the subagent.
        max_extra_steps: Maximum additional steps allowed per extension.
            Defaults to 3.

    Returns:
        ``True`` to grant the extension, ``False`` to deny.
    """
    # Rule 1: no progress after several steps -> deny
    if not request.artifacts_produced_so_far and request.current_step >= 3:
        logger.info(
            "Budget extension DENIED for %s: no artifacts produced by step %d "
            "(justification: %s).",
            request.agent_id,
            request.current_step,
            request.justification,
        )
        return False

    # Rule 2: requesting too many steps -> deny
    if request.steps_requested > max_extra_steps:
        logger.info(
            "Budget extension DENIED for %s: requested %d steps exceeds "
            "max_extra_steps=%d (justification: %s).",
            request.agent_id,
            request.steps_requested,
            max_extra_steps,
            request.justification,
        )
        return False

    # Rule 3: has artifacts -> grant (making progress)
    if request.artifacts_produced_so_far:
        logger.info(
            "Budget extension GRANTED for %s: +%d steps — %d artifact(s) "
            "produced so far (justification: %s).",
            request.agent_id,
            request.steps_requested,
            len(request.artifacts_produced_so_far),
            request.justification,
        )
        return True

    # Rule 4: default -> grant (benefit of the doubt)
    logger.info(
        "Budget extension GRANTED for %s: +%d steps — early stage, giving "
        "benefit of the doubt (justification: %s).",
        request.agent_id,
        request.steps_requested,
        request.justification,
    )
    return True


# ---------------------------------------------------------------------------
# build_escalation_message
# ---------------------------------------------------------------------------

def build_escalation_message(
    objective: str,
    results: list[dict],
    max_retries_hit: bool = False,
) -> str:
    """Format a clear, actionable escalation message for the human.

    Instead of a generic "something went wrong" this produces a message
    that explains what was attempted, what failed, and what information
    or action from the human would unblock progress.

    Args:
        objective: The original mission objective.
        results: List of ``SubagentResult``-style dicts from all attempts.
        max_retries_hit: Whether the retry ceiling was reached.

    Returns:
        A human-readable escalation string.
    """
    total_attempts = len(results)

    # -- Summarise each attempt -----------------------------------------------
    attempt_lines: list[str] = []
    for idx, result in enumerate(results, start=1):
        status = result.get("status", "unknown")
        agent = result.get("agent_id", "unknown-agent")
        trace = result.get("failure_trace") or result.get("output") or "(no detail)"
        # Truncate very long traces for readability.
        if len(trace) > 300:
            trace = trace[:297] + "..."
        attempt_lines.append(f"  {idx}. [{agent}] status={status} -- {trace}")

    attempts_block = "\n".join(attempt_lines) if attempt_lines else "  (none recorded)"

    # -- Determine what help is needed ----------------------------------------
    statuses = [r.get("status", "") for r in results]

    if "scope_violation" in statuses:
        help_needed = (
            "One or more attempts were blocked by a scope/permission restriction. "
            "Please confirm whether the required permissions should be granted."
        )
    elif "needs_clarification" in statuses:
        # Pull the most recent clarification request.
        clarifications = [
            r.get("output", "")
            for r in results
            if r.get("status") == "needs_clarification" and r.get("output")
        ]
        detail = clarifications[-1] if clarifications else "Additional context about the task."
        help_needed = f"I need clarification to proceed: {detail}"
    elif all(s == "failed" for s in statuses):
        help_needed = (
            "Every attempt failed. I may need a different approach, additional "
            "context about the environment, or confirmation that the task is "
            "feasible under the current constraints."
        )
    else:
        help_needed = (
            "I was unable to make sufficient progress. Additional guidance or "
            "a revised objective would help me continue."
        )

    # -- Compose the full message ---------------------------------------------
    retry_note = ""
    if max_retries_hit:
        retry_note = (
            f"\n\nI have exhausted the maximum retry budget ({total_attempts} "
            f"attempt(s)) without success."
        )

    message = (
        f"I need your help with the following objective:\n"
        f"  \"{objective}\"\n"
        f"\n"
        f"Here is what I tried ({total_attempts} attempt(s)):\n"
        f"{attempts_block}"
        f"{retry_note}\n"
        f"\n"
        f"What I need from you:\n"
        f"  {help_needed}"
    )

    logger.info(
        "Escalation message built for objective: %.80s... (%d attempt(s), "
        "max_retries_hit=%s).",
        objective,
        total_attempts,
        max_retries_hit,
    )

    return message


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "reflect_and_reformulate",
    "evaluate_budget_extension",
    "build_escalation_message",
]
