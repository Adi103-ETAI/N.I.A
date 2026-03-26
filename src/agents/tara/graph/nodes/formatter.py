"""TARA Graph Nodes — Response Formatter & Routing.

Contains the final-stage nodes and the conditional routing function
that drives the TARA graph's cyclic execution loop.

Node 3: response_formatter
    Scans message history for the last meaningful AI text response
    and writes it to state["final_response"].

Routing: should_continue
    Decides the next graph step after each reasoner iteration:
        • final_response set → __end__
        • tool_calls_pending → tool_executor
        • iteration limit hit → __end__
        • last msg has tool_calls → tool_executor
        • default → reasoner (continue loop)
"""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage

from src.core.logger import setup_logger
from src.core.config import get_settings
from src.core.schema.states import safe_get_content
from src.agents.tara.graph.state import TaraState, TaraNextStep

logger = setup_logger("TARA.Nodes.Formatter")
settings = get_settings()


# =============================================================================
# Node 3: Response Formatter
# =============================================================================

def response_formatter(state: TaraState) -> Dict[str, Any]:
    """Format the final response for handoff back to NIA.

    Walks the message history in reverse to find the last AIMessage
    that contains text (no pending tool calls). Falls back to
    "Task completed." if nothing is found.

    Args:
        state: Current TaraState after tool execution.

    Returns:
        Partial state update with ``final_response`` populated.
    """
    messages = state.get("messages", [])

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            # Skip intermediate AI messages that only contain tool calls
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                content = safe_get_content(msg)
                if content:
                    return {"final_response": content}

    return {"final_response": "Task completed."}


# =============================================================================
# Routing
# =============================================================================

def should_continue(state: TaraState) -> TaraNextStep:
    """Determine the next graph step after a reasoner iteration.

    Decision priority:
        1. ``final_response`` set → ``__end__``
        2. ``tool_calls_pending`` flag → ``tool_executor``
        3. Iteration limit reached → ``__end__``
        4. Last message has tool calls → ``tool_executor``
        5. Last message is plain AI text → ``__end__``
        6. Default fallback → ``reasoner``

    Args:
        state: Current TaraState.

    Returns:
        One of: ``"tool_executor"``, ``"reasoner"``, ``"__end__"``.
    """
    if state.get("final_response"):
        return "__end__"

    if state.get("tool_calls_pending"):
        return "tool_executor"

    if state.get("iteration_count", 0) >= settings.MAX_ITERATIONS:
        return "__end__"

    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage):
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                return "tool_executor"
            elif safe_get_content(last_msg) and not state.get("tool_calls_pending"):
                return "__end__"

    return "reasoner"


__all__ = ["response_formatter", "should_continue"]
