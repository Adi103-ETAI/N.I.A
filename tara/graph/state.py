"""
TARA 2.0 State Definitions.

Defines the TypedDict state schema for the TARA SubGraph.
This state flows through all graph nodes and accumulates context.

Architecture:
    NIA Master Graph → TaraState → TARA Nodes → Tool Results → Updated State
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict

# LangChain message types
try:
    from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
    _HAS_LANGCHAIN = True
except ImportError:
    BaseMessage = Any  # type: ignore
    AIMessage = dict  # type: ignore
    HumanMessage = dict  # type: ignore
    ToolMessage = dict  # type: ignore
    _HAS_LANGCHAIN = False


# =============================================================================
# State Definition
# =============================================================================

class TaraState(TypedDict, total=False):
    """
    State for the TARA SubGraph.
    
    This TypedDict defines all data that flows through TARA's reasoning loop.
    Uses operator.add for message accumulation (LangGraph pattern).
    
    Attributes:
        messages: Conversation history with tool calls/results.
        user_goal: The original user request being processed.
        
        # Dynamic Context (Updated by sensors)
        screen_context: UI tree dump or screenshot description.
        active_app: Currently focused application alias.
        clipboard: Current clipboard content preview.
        last_error: Most recent error message.
        
        # Control Flow
        tool_calls_pending: Whether tools need execution.
        iteration_count: Loop counter for safety limits.
        final_response: Response to return to NIA.
    """
    # Core message flow (accumulates via add operator)
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # User intent
    user_goal: str
    
    # Dynamic context (sensors update these)
    screen_context: Optional[str]
    active_app: Optional[str]
    clipboard: Optional[str]
    last_error: Optional[str]
    
    # Control
    tool_calls_pending: bool
    iteration_count: int
    final_response: Optional[str]
    
    # Metadata
    metadata: Dict[str, Any]


# =============================================================================
# State Factory
# =============================================================================

def create_initial_tara_state(user_goal: str, messages: List[BaseMessage] = None) -> TaraState:
    """
    Create initial state for a TARA reasoning session.
    
    Args:
        user_goal: The user's request to fulfill.
        messages: Optional existing messages from NIA.
        
    Returns:
        Initialized TaraState ready for graph execution.
    """
    if messages is None:
        messages = []
    
    return TaraState(
        messages=messages,
        user_goal=user_goal,
        screen_context=None,
        active_app=None,
        clipboard=None,
        last_error=None,
        tool_calls_pending=False,
        iteration_count=0,
        final_response=None,
        metadata={},
    )


# =============================================================================
# Routing Types
# =============================================================================

# Valid next steps in the graph
TaraNextStep = Literal["reasoner", "tool_executor", "context_updater", "__end__"]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TaraState",
    "TaraNextStep",
    "create_initial_tara_state",
    "BaseMessage",
    "AIMessage",
    "HumanMessage",
    "ToolMessage",
]
