"""Core engine exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from niaharness.engine.messages import (
        ConversationMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
    )
    from niaharness.engine.query_engine import (
        AbortController,
        FileStateCache,
        PermissionDenialTracker,
        QueryEngine,
    )
    from niaharness.engine.stream_events import (
        AssistantTextDelta,
        AssistantThinkingDelta,
        AssistantTurnComplete,
        BudgetExceeded,
        CompactBoundary,
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

__all__ = [
    "AbortController",
    "AssistantTextDelta",
    "AssistantThinkingDelta",
    "AssistantTurnComplete",
    "BudgetExceeded",
    "CompactBoundary",
    "ContinuationNudge",
    "ConversationMessage",
    "FileStateCache",
    "MaxOutputTokensRecovery",
    "MaxTurnsReached",
    "PermissionDenialTracker",
    "QueryEngine",
    "QueryResult",
    "StreamEvent",
    "TerminationReason",
    "TextBlock",
    "ToolExecutionCompleted",
    "ToolExecutionStarted",
    "ToolFailureLoopDetected",
    "ToolResultBlock",
    "ToolUseBlock",
    "ToolUseSummary",
    "UserInterrupted",
]


def __getattr__(name: str):
    if name in {"ConversationMessage", "TextBlock", "ToolResultBlock", "ToolUseBlock"}:
        from niaharness.engine.messages import (
            ConversationMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
        )

        return {
            "ConversationMessage": ConversationMessage,
            "TextBlock": TextBlock,
            "ToolResultBlock": ToolResultBlock,
            "ToolUseBlock": ToolUseBlock,
        }[name]

    if name in {
        "AbortController",
        "FileStateCache",
        "PermissionDenialTracker",
        "QueryEngine",
    }:
        from niaharness.engine.query_engine import (
            AbortController,
            FileStateCache,
            PermissionDenialTracker,
            QueryEngine,
        )

        return {
            "AbortController": AbortController,
            "FileStateCache": FileStateCache,
            "PermissionDenialTracker": PermissionDenialTracker,
            "QueryEngine": QueryEngine,
        }[name]

    if name in {
        "AssistantTextDelta",
        "AssistantThinkingDelta",
        "AssistantTurnComplete",
        "BudgetExceeded",
        "CompactBoundary",
        "ContinuationNudge",
        "MaxOutputTokensRecovery",
        "MaxTurnsReached",
        "QueryResult",
        "StreamEvent",
        "TerminationReason",
        "ToolExecutionCompleted",
        "ToolExecutionStarted",
        "ToolFailureLoopDetected",
        "ToolUseSummary",
        "UserInterrupted",
    }:
        from niaharness.engine.stream_events import (
            AssistantTextDelta,
            AssistantThinkingDelta,
            AssistantTurnComplete,
            BudgetExceeded,
            CompactBoundary,
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

        return {
            "AssistantTextDelta": AssistantTextDelta,
            "AssistantThinkingDelta": AssistantThinkingDelta,
            "AssistantTurnComplete": AssistantTurnComplete,
            "BudgetExceeded": BudgetExceeded,
            "CompactBoundary": CompactBoundary,
            "ContinuationNudge": ContinuationNudge,
            "MaxOutputTokensRecovery": MaxOutputTokensRecovery,
            "MaxTurnsReached": MaxTurnsReached,
            "QueryResult": QueryResult,
            "StreamEvent": StreamEvent,
            "TerminationReason": TerminationReason,
            "ToolExecutionCompleted": ToolExecutionCompleted,
            "ToolExecutionStarted": ToolExecutionStarted,
            "ToolFailureLoopDetected": ToolFailureLoopDetected,
            "ToolUseSummary": ToolUseSummary,
            "UserInterrupted": UserInterrupted,
        }[name]

    raise AttributeError(name)
