"""Events yielded by the query engine.

Ported from OpenClaude's SDKMessage types with additional system-level events
for compaction boundaries, API retries, tool use summaries, and recovery paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from niaharness.api.usage import UsageSnapshot
from niaharness.engine.messages import ConversationMessage


# ---------------------------------------------------------------------------
# Streaming events (per-token)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssistantTextDelta:
    """Incremental assistant text."""

    text: str


@dataclass(frozen=True)
class AssistantThinkingDelta:
    """Incremental thinking block text."""

    text: str


# ---------------------------------------------------------------------------
# Turn-level events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssistantTurnComplete:
    """Completed assistant turn."""

    message: ConversationMessage
    usage: UsageSnapshot


@dataclass(frozen=True)
class ToolExecutionStarted:
    """The engine is about to execute a tool."""

    tool_name: str
    tool_input: dict[str, Any]


@dataclass(frozen=True)
class ToolExecutionCompleted:
    """A tool has finished executing."""

    tool_name: str
    output: str
    is_error: bool = False


# ---------------------------------------------------------------------------
# Tool use summary (ported from OpenClaude tool_use_summary)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolUseSummary:
    """Summary of a batch of tool uses, emitted after tools complete.

    Matches OpenClaude's ToolUseSummaryMessage for SDK consumers.
    """

    summary: str
    preceding_tool_use_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# System messages (compact boundary, API retry, warnings)
# ---------------------------------------------------------------------------


class SystemMessageSubtype(str, Enum):
    COMPACT_BOUNDARY = "compact_boundary"
    API_RETRY = "api_retry"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class CompactBoundary:
    """Signals that auto-compaction occurred and older messages were summarized.

    Matches OpenClaude's SDKCompactBoundaryMessage.
    """

    pre_compact_token_count: int = 0
    post_compact_token_count: int = 0
    summary_preview: str = ""


@dataclass(frozen=True)
class ApiRetryNotification:
    """API call failed and will be retried.

    Matches OpenClaude's system/api_retry event.
    """

    attempt: int
    max_retries: int
    retry_delay_ms: float
    error_status: str | None = None
    error_message: str = ""


@dataclass(frozen=True)
class SystemMessage:
    """Generic system-level informational message."""

    subtype: SystemMessageSubtype
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Recovery / termination events
# ---------------------------------------------------------------------------


class TerminationReason(str, Enum):
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
    MAX_BUDGET_USD = "max_budget_usd"
    MAX_OUTPUT_TOKENS = "max_output_tokens"


@dataclass(frozen=True)
class MaxTurnsReached:
    """Emitted when the engine hits the max-turns limit."""

    max_turns: int
    turn_count: int


@dataclass(frozen=True)
class BudgetExceeded:
    """Emitted when the USD budget cap is hit."""

    max_budget_usd: float
    total_cost_usd: float


@dataclass(frozen=True)
class ToolFailureLoopDetected:
    """Emitted when the tool failure loop guard trips.

    Matches OpenClaude's toolFailureLoopGuard tripped decision.
    """

    kind: str  # 'signature' | 'category' | 'path'
    threshold: int
    message: str
    tool_name: str | None = None
    error_category: str | None = None
    path: str | None = None


@dataclass(frozen=True)
class UserInterrupted:
    """Emitted when the user aborts the query mid-stream."""

    during_tool_execution: bool = False


@dataclass(frozen=True)
class MaxOutputTokensRecovery:
    """Emitted when max_output_tokens is hit and recovery is attempted."""

    attempt: int
    max_attempts: int
    message: str = ""


@dataclass(frozen=True)
class ContinuationNudge:
    """Emitted when a continuation nudge is sent to the model."""

    nudge_count: int
    max_nudges: int
    reason: str = ""


@dataclass(frozen=True)
class QueryResult:
    """Terminal result of a query run.

    Matches OpenClaude's SDK result message.
    """

    reason: TerminationReason
    is_error: bool = False
    duration_ms: float = 0.0
    num_turns: int = 0
    tool_call_count: int = 0
    result_text: str = ""
    total_cost_usd: float = 0.0
    usage: UsageSnapshot | None = None
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Union type for all stream events
# ---------------------------------------------------------------------------

StreamEvent = (
    AssistantTextDelta
    | AssistantThinkingDelta
    | AssistantTurnComplete
    | ToolExecutionStarted
    | ToolExecutionCompleted
    | ToolUseSummary
    | CompactBoundary
    | ApiRetryNotification
    | SystemMessage
    | MaxTurnsReached
    | BudgetExceeded
    | ToolFailureLoopDetected
    | UserInterrupted
    | MaxOutputTokensRecovery
    | ContinuationNudge
    | QueryResult
)
