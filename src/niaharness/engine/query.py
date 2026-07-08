"""Core tool-aware query loop.

Ported from OpenClaude's query.ts with auto-compaction integration, recovery
paths, continuation nudge detection, tool failure loop guard, and budget
enforcement.  Maintains backward compatibility with the existing niaharness
interface while adding idiomatic Python async generators and type hints.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

from niaharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    SupportsStreamingMessages,
)
from niaharness.api.usage import UsageSnapshot
from niaharness.engine.messages import ConversationMessage, ToolResultBlock
from niaharness.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    BudgetExceeded,
    ContinuationNudge,
    MaxOutputTokensRecovery,
    MaxTurnsReached,
    QueryResult,
    StreamEvent,
    TerminationReason,
    ToolExecutionCompleted,
    ToolExecutionStarted,
    ToolFailureLoopDetected,
    ToolUseSummary,
    UserInterrupted,
)
from niaharness.hooks import HookEvent, HookExecutor
from niaharness.permissions.checker import PermissionChecker
from niaharness.tools.base import ToolExecutionContext, ToolRegistry

logger = logging.getLogger(__name__)

PermissionPrompt = Callable[[str, str], Awaitable[bool]]
AskUserPrompt = Callable[[str], Awaitable[str]]

# ---------------------------------------------------------------------------
# Constants (ported from OpenClaude query.ts)
# ---------------------------------------------------------------------------

MAX_OUTPUT_TOKENS_RECOVERY_LIMIT = 3
MAX_CONTINUATION_NUDGES = 3
DEFAULT_TOOL_FAILURE_LOOP_THRESHOLD = 3
MAX_FALLBACK_CATEGORY_LENGTH = 120
COMPLETION_THRESHOLD = 0.9
DIMINISHING_THRESHOLD = 500


# ---------------------------------------------------------------------------
# Transition types (ported from OpenClaude transitions.ts)
# ---------------------------------------------------------------------------


class TerminalReason(str):
    """Terminal reasons for query loop exit."""

    BLOCKING_LIMIT = "blocking_limit"
    IMAGE_ERROR = "image_error"
    MODEL_ERROR = "model_error"
    ABORTED_STREAMING = "aborted_streaming"
    PROMPT_TOO_LONG = "prompt_too_long"
    COMPLETED = "completed"
    STOP_HOOK_PREVENTED = "stop_hook_prevented"
    ABORTED_TOOLS = "aborted_tools"
    HOOK_STOPPED = "hook_stopped"
    MAX_TURNS = "max_turns"
    TOOL_FAILURE_LOOP = "tool_failure_loop"


class ContinueReason(str):
    """Continue reasons for query loop iteration."""

    MAX_OUTPUT_TOKENS_RECOVERY = "max_output_tokens_recovery"
    CONTINUATION_NUDGE = "continuation_nudge"
    NEXT_TURN = "next_turn"
    TOOL_FAILURE_RETRY = "tool_failure_retry"


# ---------------------------------------------------------------------------
# Tool failure loop guard (ported from OpenClaude toolFailureLoopGuard.ts)
# ---------------------------------------------------------------------------


@dataclass
class ToolFailureLoopGuardState:
    """Mutable state for detecting repeated tool failure loops."""

    signature_counts: dict[str, int] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    path_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolFailureLoopDecision:
    """Decision from the tool failure loop guard."""

    tripped: bool
    kind: str = ""
    threshold: int = 0
    message: str = ""
    tool_name: str | None = None
    error_category: str | None = None
    path: str | None = None


def get_tool_failure_loop_threshold(
    env_value: str | None = None,
) -> int:
    """Return the configurable failure-loop threshold."""
    if env_value is None:
        return DEFAULT_TOOL_FAILURE_LOOP_THRESHOLD
    try:
        return int(env_value)
    except (ValueError, TypeError):
        return DEFAULT_TOOL_FAILURE_LOOP_THRESHOLD


def _normalize_error_category(content: str) -> str:
    """Categorise a tool error message into a short label."""
    normalized = re.sub(r"</?tool_use_error[^>]*>", " ", content)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if re.search(r"InputValidationError", normalized, re.IGNORECASE):
        return "InputValidationError"
    if re.search(r"Invalid tool parameters", normalized, re.IGNORECASE):
        return "InputValidationError"
    if re.search(r"No such tool available", normalized, re.IGNORECASE):
        return "NoSuchTool"
    if re.search(r"\b(EACCES|EPERM)\b", normalized, re.IGNORECASE):
        return "PermissionError"
    if re.search(r"permission denied", normalized, re.IGNORECASE):
        return "PermissionError"
    if re.search(r"\bENOENT\b", normalized) or re.search(r"not found", normalized, re.IGNORECASE):
        return "NotFound"
    if re.search(r"Error writing file", normalized, re.IGNORECASE):
        return "FileWriteError"

    return normalized.lower()[:MAX_FALLBACK_CATEGORY_LENGTH] or "unknown error"


def _is_ignored_synthetic(content: str) -> bool:
    """Return True for synthetic tool results that should not count as failures."""
    normalized = content.lower().strip()
    unbracketed = re.sub(r"^\[(.*)\]$", r"\1", normalized).strip()
    without_prefix = re.sub(r"^error:\s*", "", unbracketed).strip()

    return (
        without_prefix == "interrupted by user"
        or without_prefix.startswith("request interrupted by user")
        or without_prefix == "user rejected tool use"
        or without_prefix.startswith("the user doesn't want to proceed with this tool use")
        or without_prefix.startswith("the user doesn't want to take this action right now")
        or without_prefix.startswith("streaming fallback - tool execution discarded")
        or without_prefix.startswith("cancelled: parallel tool call")
    )


def _extract_normalized_path(tool_input: dict[str, Any] | None) -> str | None:
    """Extract and normalise a file path from tool input."""
    if not isinstance(tool_input, dict):
        return None
    for key in ("file_path", "path", "notebook_path"):
        value = tool_input.get(key)
        if isinstance(value, str):
            normalized = value.strip().replace("\\", "/")
            normalized = re.sub(r"/{2,}", "/", normalized)
            normalized = normalized.rstrip("/")
            if normalized == "" and value.strip().startswith("/"):
                return "/"
            if normalized:
                return normalized
    return None


def update_tool_failure_loop_guard(
    state: ToolFailureLoopGuardState,
    tool_use_blocks: list[dict[str, Any]],
    tool_results: list[ToolResultBlock],
    threshold: int | None = None,
) -> ToolFailureLoopDecision:
    """Check tool results for repeated failure patterns.

    Ported from OpenClaude's updateToolFailureLoopGuard.  Returns a decision
    indicating whether the guard has tripped.
    """
    actual_threshold = threshold if threshold is not None else get_tool_failure_loop_threshold()
    if actual_threshold == 0:
        return ToolFailureLoopDecision(tripped=False)

    tool_use_by_id: dict[str, dict[str, Any]] = {}
    for block in tool_use_blocks:
        block_id = block.get("id", "")
        if isinstance(block_id, str):
            tool_use_by_id[block_id] = block

    failures: list[dict[str, str | None]] = []
    has_success = False

    for result in tool_results:
        if result.is_error:
            if _is_ignored_synthetic(result.content):
                continue
            tool_use = tool_use_by_id.get(result.tool_use_id, {})
            tool_name = tool_use.get("name", "unknown")
            if isinstance(tool_name, str):
                pass
            else:
                tool_name = "unknown"
            error_category = _normalize_error_category(result.content)
            failures.append(
                {
                    "tool_name": tool_name,
                    "error_category": error_category,
                    "path": _extract_normalized_path(tool_use.get("input")),
                }
            )
        else:
            has_success = True

    if has_success:
        state.signature_counts.clear()
        state.category_counts.clear()
        state.path_counts.clear()
        return ToolFailureLoopDecision(tripped=False)

    for failure in failures:
        tool_name = str(failure["tool_name"])
        error_category = str(failure["error_category"])
        path = failure["path"]

        sig_key = f"{tool_name}\0{error_category}"
        sig_count = state.signature_counts.get(sig_key, 0) + 1
        state.signature_counts[sig_key] = sig_count

        cat_count = state.category_counts.get(error_category, 0) + 1
        state.category_counts[error_category] = cat_count

        path_count = 0
        if path is not None:
            path_count = state.path_counts.get(path, 0) + 1
            state.path_counts[path] = path_count

        if path_count >= actual_threshold and path:
            return ToolFailureLoopDecision(
                tripped=True,
                kind="path",
                threshold=actual_threshold,
                path=path,
                message=(
                    f"Stopped: repeated tool failures detected.\n\n"
                    f"The path `{path}` failed {actual_threshold} times. "
                    f"Please inspect permissions, path, or tool schema before retrying."
                ),
            )

        if sig_count >= actual_threshold:
            return ToolFailureLoopDecision(
                tripped=True,
                kind="signature",
                threshold=actual_threshold,
                tool_name=tool_name,
                error_category=error_category,
                message=(
                    f"Stopped: repeated tool failures detected.\n\n"
                    f"`{tool_name}` failed {actual_threshold} times with "
                    f"`{error_category}`. Please inspect permissions, path, "
                    f"or tool schema before retrying."
                ),
            )

        if cat_count >= actual_threshold:
            return ToolFailureLoopDecision(
                tripped=True,
                kind="category",
                threshold=actual_threshold,
                error_category=error_category,
                message=(
                    f"Stopped: repeated tool failures detected.\n\n"
                    f"Tool calls failed {actual_threshold} times with "
                    f"`{error_category}`. Please inspect permissions, path, "
                    f"or tool schema before retrying."
                ),
            )

    return ToolFailureLoopDecision(tripped=False)


# ---------------------------------------------------------------------------
# Continuation nudge detection (ported from OpenClaude query.ts)
# ---------------------------------------------------------------------------

_CONTINUATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bso now i (?:have to|need to|must)\b"),
    re.compile(r"\blet me (?:now |go ahead and )?\b"),
    re.compile(r"\bi(?:'ll| will) (?:now |go ahead and )?\b"),
    re.compile(r"\bnext,?\s+(?:i(?:'ll| will) |we need to )\b"),
    re.compile(r"\btime to\b"),
    re.compile(r"\bnow (?:i (?:need to|have to|must)|let me|we)\b"),
    re.compile(r"\bfirst,?\s+(?:i(?:'ll| will) |we need to )\b"),
    re.compile(r"\bmy (?:next|immediate) (?:step|action|task)\b"),
    re.compile(r"\bproceeding with\b"),
    re.compile(r"\bmoving on to\b"),
    re.compile(r"\bpicking up (?:where|from)\b"),
]


def _analyze_continuation_intent(text: str) -> tuple[bool, str]:
    """Detect whether the model intends to continue but forgot tool calls.

    Returns (should_nudge, reason_description).
    """
    lower = text.lower()
    for pattern in _CONTINUATION_PATTERNS:
        match = pattern.search(lower)
        if match:
            return True, f"continuation_signal: {match.group()!r}"
    return False, ""


# ---------------------------------------------------------------------------
# Token budget tracker (ported from OpenClaude tokenBudget.ts)
# ---------------------------------------------------------------------------


@dataclass
class BudgetTracker:
    """Tracks token budget continuation state."""

    continuation_count: int = 0
    last_delta_tokens: int = 0
    last_global_turn_tokens: int = 0
    started_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TokenBudgetDecision:
    """Decision from the token budget check."""

    action: str  # 'continue' | 'stop'
    nudge_message: str = ""
    continuation_count: int = 0
    pct: int = 0
    turn_tokens: int = 0
    budget: int = 0
    diminishing_returns: bool = False


def check_token_budget(
    tracker: BudgetTracker,
    budget: int | None,
    global_turn_tokens: int,
) -> TokenBudgetDecision:
    """Check whether the token budget allows continuation.

    Ported from OpenClaude's checkTokenBudget.
    """
    if budget is None or budget <= 0:
        return TokenBudgetDecision(action="stop")

    turn_tokens = global_turn_tokens
    pct = round((turn_tokens / budget) * 100)
    delta = global_turn_tokens - tracker.last_global_turn_tokens

    is_diminishing = (
        tracker.continuation_count >= 3
        and delta < DIMINISHING_THRESHOLD
        and tracker.last_delta_tokens < DIMINISHING_THRESHOLD
    )

    if not is_diminishing and turn_tokens < budget * COMPLETION_THRESHOLD:
        tracker.continuation_count += 1
        tracker.last_delta_tokens = delta
        tracker.last_global_turn_tokens = global_turn_tokens
        return TokenBudgetDecision(
            action="continue",
            nudge_message=(
                f"Token budget: {pct}% used ({turn_tokens:,} / {budget:,} tokens). "
                f"Continue working to complete the task."
            ),
            continuation_count=tracker.continuation_count,
            pct=pct,
            turn_tokens=turn_tokens,
            budget=budget,
        )

    if is_diminishing or tracker.continuation_count > 0:
        return TokenBudgetDecision(
            action="stop",
            diminishing_returns=is_diminishing,
            continuation_count=tracker.continuation_count,
            pct=pct,
            turn_tokens=turn_tokens,
            budget=budget,
        )

    return TokenBudgetDecision(action="stop")


# ---------------------------------------------------------------------------
# Query context
# ---------------------------------------------------------------------------


@dataclass
class QueryContext:
    """Context shared across a query run."""

    api_client: SupportsStreamingMessages
    tool_registry: ToolRegistry
    permission_checker: PermissionChecker
    cwd: Path
    model: str
    system_prompt: str
    max_tokens: int
    permission_prompt: PermissionPrompt | None = None
    ask_user_prompt: AskUserPrompt | None = None
    max_turns: int = 200
    max_budget_usd: float | None = None
    hook_executor: HookExecutor | None = None
    tool_metadata: dict[str, object] | None = None
    abort_event: asyncio.Event | None = None
    token_budget: int | None = None


# ---------------------------------------------------------------------------
# Main query loop
# ---------------------------------------------------------------------------


async def run_query(
    context: QueryContext,
    messages: list[ConversationMessage],
    *,
    cost_usd_fn: Callable[[], float] | None = None,
) -> AsyncIterator[tuple[StreamEvent, UsageSnapshot | None]]:
    """Run the conversation loop until the model stops requesting tools.

    Auto-compaction is checked at the start of each turn.  When the
    estimated token count exceeds the model's auto-compact threshold,
    the engine first tries a cheap microcompact (clearing old tool result
    content) and, if that is not enough, performs a full LLM-based
    summarization of older messages.

    Recovery paths:
    - prompt-too-long: surfaces error and returns
    - max-output-tokens: injects recovery nudge up to 3 times
    - continuation nudge: detects model intent to continue without tool calls
    - tool failure loop: stops when the same error repeats
    - budget enforcement: max turns and max USD cap
    """
    from niaharness.services.compact import (
        AutoCompactState,
        auto_compact_if_needed,
    )

    compact_state = AutoCompactState()
    tool_failure_guard = ToolFailureLoopGuardState()
    budget_tracker = BudgetTracker()
    max_output_recovery_count = 0
    continuation_nudge_count = 0
    turn_count = 0

    _start_time = time.monotonic()

    for turn in range(context.max_turns):
        turn_count = turn + 1

        # --- abort check --------------------------------------------------
        if context.abort_event and context.abort_event.is_set():
            yield UserInterrupted(), None
            return

        # --- budget enforcement (USD) -------------------------------------
        if context.max_budget_usd is not None and cost_usd_fn is not None:
            current_cost = cost_usd_fn()
            if current_cost >= context.max_budget_usd:
                yield BudgetExceeded(
                    max_budget_usd=context.max_budget_usd,
                    total_cost_usd=current_cost,
                ), None
                yield QueryResult(
                    reason=TerminationReason.MAX_BUDGET_USD,
                    is_error=True,
                    duration_ms=(time.monotonic() - _start_time) * 1000,
                    num_turns=turn_count,
                    total_cost_usd=current_cost,
                    errors=[f"Reached maximum budget (${context.max_budget_usd})"],
                ), None
                return

        # --- auto-compact check before calling the model -------------------
        # auto_compact_if_needed returns a (possibly new) list.  When it
        # mutates the conversation we need to propagate that back to the
        # caller's list reference so subsequent appends are visible to them.
        compacted_messages, was_compacted, compact_state = await auto_compact_if_needed(
            messages,
            model=context.model,
            state=compact_state,
        )
        if was_compacted:
            messages[:] = compacted_messages

        # --- token budget continuation check -------------------------------
        if context.token_budget is not None:
            budget_decision = check_token_budget(
                budget_tracker,
                context.token_budget,
                sum(
                    len(m.text) // 4 for m in messages
                ),  # rough token estimate
            )
            if budget_decision.action == "continue":
                nudge_msg = ConversationMessage.from_user_text(budget_decision.nudge_message)
                messages.append(nudge_msg)
                yield ContinuationNudge(
                    nudge_count=budget_decision.continuation_count,
                    max_nudges=10,
                    reason=budget_decision.nudge_message,
                ), None
                continue

        # --- call the model ------------------------------------------------
        final_message: ConversationMessage | None = None
        usage = UsageSnapshot()
        is_max_output_tokens = False

        try:
            async for event in context.api_client.stream_message(
                ApiMessageRequest(
                    model=context.model,
                    messages=messages,
                    system_prompt=context.system_prompt,
                    max_tokens=context.max_tokens,
                    tools=context.tool_registry.to_api_schema(),
                )
            ):
                if isinstance(event, ApiTextDeltaEvent):
                    yield AssistantTextDelta(text=event.text), None
                    continue

                if isinstance(event, ApiMessageCompleteEvent):
                    final_message = event.message
                    usage = event.usage

        except Exception as exc:
            error_msg = str(exc)

            # P1 fix: consult the recovery registry before giving up.
            # The registry has 16 one-shot guards (prompt_too_long_compress,
            # rate_limit_429_backoff, auth_401_rotate_credential, etc.)
            # that can potentially recover from transient failures.
            try:
                from niaharness.engine.recovery import (
                    get_default_registry,
                    ErrorContext,
                    ActionType,
                )

                registry = get_default_registry()
                ctx = ErrorContext(
                    exc=exc,
                    attempt=turn_count,
                    max_retries=context.max_turns,
                    provider=getattr(context.api_client, "provider", ""),
                    model=context.model,
                )
                action = registry.match(exc, ctx)
            except Exception:
                action = None

            if action is not None:
                if action.type == ActionType.COMPRESS and action.should_retry:
                    # Trigger compaction and retry this turn.
                    from niaharness.services.compact import auto_compact_if_needed
                    messages[:] = auto_compact_if_needed(
                        messages=messages,
                        model=context.model,
                        threshold=4000,
                        state=compact_state,
                    )
                    yield MaxOutputTokensRecovery(
                        message=f"Recovery: {action.description} — compacted and retrying",
                    ), None
                    continue  # retry the same turn

                if action.type == ActionType.RETRY and action.should_retry:
                    import asyncio as _asyncio
                    yield MaxOutputTokensRecovery(
                        message=f"Recovery: {action.description} — retrying in {action.delay_seconds:.1f}s",
                    ), None
                    await _asyncio.sleep(action.delay_seconds)
                    continue  # retry the same turn

                # For ABORT or unhandled action types, fall through to error.
                if action.type == ActionType.ABORT:
                    yield QueryResult(
                        reason=TerminationReason.MODEL_ERROR,
                        is_error=True,
                        duration_ms=(time.monotonic() - _start_time) * 1000,
                        num_turns=turn_count,
                        errors=[f"{error_msg} (recovery: {action.description})"],
                    ), None
                    return

            # Detect prompt-too-long errors (legacy path — recovery didn't handle it)
            if "prompt is too long" in error_msg.lower() or "invalid_request" in error_msg.lower():
                yield QueryResult(
                    reason=TerminationReason.PROMPT_TOO_LONG,
                    is_error=True,
                    duration_ms=(time.monotonic() - _start_time) * 1000,
                    num_turns=turn_count,
                    errors=[error_msg],
                ), None
                return
            # Other API errors
            yield QueryResult(
                reason=TerminationReason.MODEL_ERROR,
                is_error=True,
                duration_ms=(time.monotonic() - _start_time) * 1000,
                num_turns=turn_count,
                errors=[error_msg],
            ), None
            return

        if final_message is None:
            raise RuntimeError("Model stream finished without a final message")

        # --- detect max_output_tokens in the response content ---------------
        last_text = final_message.text
        if "output token limit" in last_text.lower() or "max_output_tokens" in last_text.lower():
            is_max_output_tokens = True

        messages.append(final_message)
        yield AssistantTurnComplete(message=final_message, usage=usage), usage

        # --- no tool calls: check continuation nudge -----------------------
        if not final_message.tool_uses:
            # Max output tokens recovery
            if is_max_output_tokens and max_output_recovery_count < MAX_OUTPUT_TOKENS_RECOVERY_LIMIT:
                recovery_msg = ConversationMessage.from_user_text(
                    "Output token limit hit. Resume directly — no apology, no recap of what "
                    "you were doing. Pick up mid-thought if that is where the cut happened. "
                    "Break remaining work into smaller pieces."
                )
                messages.append(recovery_msg)
                max_output_recovery_count += 1
                yield MaxOutputTokensRecovery(
                    attempt=max_output_recovery_count,
                    max_attempts=MAX_OUTPUT_TOKENS_RECOVERY_LIMIT,
                ), None
                continue

            # Continuation nudge detection
            if (
                last_text
                and turn_count < context.max_turns
                and continuation_nudge_count < MAX_CONTINUATION_NUDGES
            ):
                should_nudge, nudge_reason = _analyze_continuation_intent(last_text)
                if should_nudge:
                    nudge_msg = ConversationMessage.from_user_text(
                        "Continue with the task. If you were interrupted, resume your thought. "
                        "Otherwise, use the appropriate tools to proceed to the next step."
                    )
                    messages.append(nudge_msg)
                    continuation_nudge_count += 1
                    yield ContinuationNudge(
                        nudge_count=continuation_nudge_count,
                        max_nudges=MAX_CONTINUATION_NUDGES,
                        reason=nudge_reason,
                    ), None
                    continue

            # Normal completion
            yield QueryResult(
                reason=TerminationReason.COMPLETED,
                is_error=False,
                duration_ms=(time.monotonic() - _start_time) * 1000,
                num_turns=turn_count,
                result_text=final_message.text,
            ), None
            return

        # --- execute tool calls --------------------------------------------
        tool_calls = final_message.tool_uses

        if len(tool_calls) == 1:
            tc = tool_calls[0]
            yield ToolExecutionStarted(tool_name=tc.name, tool_input=tc.input), None
            result = await _execute_tool_call(context, tc.name, tc.id, tc.input)
            yield ToolExecutionCompleted(
                tool_name=tc.name,
                output=result.content,
                is_error=result.is_error,
            ), None
            tool_results = [result]
        else:
            for tc in tool_calls:
                yield ToolExecutionStarted(tool_name=tc.name, tool_input=tc.input), None

            async def _run(tc: Any) -> ToolResultBlock:
                return await _execute_tool_call(context, tc.name, tc.id, tc.input)

            results = await asyncio.gather(*[_run(tc) for tc in tool_calls])
            tool_results = list(results)

            for tc, result in zip(tool_calls, tool_results):
                yield ToolExecutionCompleted(
                    tool_name=tc.name,
                    output=result.content,
                    is_error=result.is_error,
                ), None

        # --- tool failure loop guard ---------------------------------------
        tool_use_block_dicts = [
            {"id": tc.id, "name": tc.name, "input": tc.input}
            for tc in tool_calls
        ]
        failure_decision = update_tool_failure_loop_guard(
            tool_failure_guard,
            tool_use_block_dicts,
            tool_results,
        )
        if failure_decision.tripped:
            yield ToolFailureLoopDetected(
                kind=failure_decision.kind,
                threshold=failure_decision.threshold,
                message=failure_decision.message,
                tool_name=failure_decision.tool_name,
                error_category=failure_decision.error_category,
                path=failure_decision.path,
            ), None
            yield QueryResult(
                reason=TerminationReason.TOOL_FAILURE_LOOP,
                is_error=True,
                duration_ms=(time.monotonic() - _start_time) * 1000,
                num_turns=turn_count,
                errors=[failure_decision.message],
            ), None
            return

        # --- tool use summary (async, non-blocking) ------------------------
        tool_ids = [tc.id for tc in tool_calls]
        yield ToolUseSummary(
            summary=f"Executed {len(tool_calls)} tool(s): {', '.join(tc.name for tc in tool_calls)}",
            preceding_tool_use_ids=tool_ids,
        ), None

        # --- reset per-turn counters on success ----------------------------
        max_output_recovery_count = 0
        continuation_nudge_count = 0

        messages.append(ConversationMessage(role="user", content=tool_results))

    yield QueryResult(
        reason=TerminationReason.MAX_TURNS,
        is_error=True,
        duration_ms=(time.monotonic() - _start_time) * 1000,
        num_turns=turn_count,
        errors=[f"Exceeded maximum turn limit ({context.max_turns})"],
    ), None


# ---------------------------------------------------------------------------
# Tool execution helper
# ---------------------------------------------------------------------------


async def _execute_tool_call(
    context: QueryContext,
    tool_name: str,
    tool_use_id: str,
    tool_input: dict[str, object],
) -> ToolResultBlock:
    """Execute a single tool call with hook and permission checks."""
    if context.hook_executor is not None:
        pre_hooks = await context.hook_executor.execute(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "event": HookEvent.PRE_TOOL_USE.value,
            },
        )
        if pre_hooks.blocked:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=pre_hooks.reason or f"pre_tool_use hook blocked {tool_name}",
                is_error=True,
            )

    tool = context.tool_registry.get(tool_name)
    if tool is None:
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"Unknown tool: {tool_name}",
            is_error=True,
        )

    try:
        parsed_input = tool.input_model.model_validate(tool_input)
    except Exception as exc:
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"Invalid input for {tool_name}: {exc}",
            is_error=True,
        )

    _file_path = str(tool_input.get("file_path", "")) or None
    _command = str(tool_input.get("command", "")) or None
    decision = context.permission_checker.evaluate(
        tool_name,
        is_read_only=tool.is_read_only(parsed_input),
        file_path=_file_path,
        command=_command,
    )
    if not decision.allowed:
        if decision.requires_confirmation and context.permission_prompt is not None:
            confirmed = await context.permission_prompt(tool_name, decision.reason)
            if not confirmed:
                return ToolResultBlock(
                    tool_use_id=tool_use_id,
                    content=f"Permission denied for {tool_name}",
                    is_error=True,
                )
        else:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=decision.reason or f"Permission denied for {tool_name}",
                is_error=True,
            )

    result = await tool.execute(
        parsed_input,
        ToolExecutionContext(
            cwd=context.cwd,
            metadata={
                "tool_registry": context.tool_registry,
                "ask_user_prompt": context.ask_user_prompt,
                **(context.tool_metadata or {}),
            },
        ),
    )
    tool_result = ToolResultBlock(
        tool_use_id=tool_use_id,
        content=result.output,
        is_error=result.is_error,
    )
    if context.hook_executor is not None:
        await context.hook_executor.execute(
            HookEvent.POST_TOOL_USE,
            {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_result.content,
                "tool_is_error": tool_result.is_error,
                "event": HookEvent.POST_TOOL_USE.value,
            },
        )
    return tool_result
