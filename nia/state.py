"""N.I.A. State Module - Agent State Definitions.

This module defines the state structures used by the LangGraph-based
supervisor architecture for NIA (Neural Intelligence Assistant).

The state tracks:
- Conversation messages (using LangChain's BaseMessage format)
- Routing decisions (which agent should act next)
- Metadata for tracing and debugging
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict
from datetime import datetime

# LangChain message types
try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    _HAS_LANGCHAIN = True
except ImportError:
    # Fallback for environments without langchain
    BaseMessage = Any  # type: ignore
    HumanMessage = dict  # type: ignore
    AIMessage = dict  # type: ignore
    SystemMessage = dict  # type: ignore
    _HAS_LANGCHAIN = False

# LangGraph operator for message accumulation
try:
    from langgraph.graph import add_messages
    _HAS_LANGGRAPH = True
except ImportError:
    # Fallback: simple list append
    def add_messages(left: List, right: List) -> List:
        return left + right
    _HAS_LANGGRAPH = False


# =============================================================================
# Agent Names (for routing)
# =============================================================================

# Available agents in the system
AGENT_SUPERVISOR = "supervisor"
AGENT_IRIS = "iris"       # Vision specialist
AGENT_TARA = "tara"       # Logic/reasoning specialist
AGENT_END = "__end__"     # Terminal state

# Valid routing destinations
AgentName = Literal["supervisor", "iris", "tara", "__end__"]


# =============================================================================
# Agent State Definition
# =============================================================================

class AgentState(TypedDict, total=False):
    """State shared across all agents in the NIA supervisor graph.
    
    This TypedDict defines the structure of state passed between nodes
    in the LangGraph execution.
    
    Attributes:
        messages: Conversation history (accumulated across turns).
                  Uses LangGraph's add_messages reducer for proper merging.
        next: Name of the next agent to execute. Set by supervisor routing.
        user_input: Original user input for the current turn.
        final_response: The response to return to the user.
        route_reason: Why the supervisor chose this route (for debugging).
        metadata: Additional context (timestamps, turn counts, etc.).
    """
    # Core conversation state (uses add_messages reducer)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Routing control
    next: AgentName
    
    # Current turn data
    user_input: str
    final_response: Optional[str]
    
    # Debugging/tracing
    route_reason: Optional[str]
    metadata: Dict[str, Any]


# =============================================================================
# State Factory Functions
# =============================================================================

def create_initial_state(user_input: str) -> AgentState:
    """Create an initial state for a new conversation turn.
    
    Args:
        user_input: The user's message to process.
        
    Returns:
        AgentState ready for graph execution.
    """
    if _HAS_LANGCHAIN:
        messages = [HumanMessage(content=user_input)]
    else:
        messages = [{"role": "user", "content": user_input}]
    
    return AgentState(
        messages=messages,
        next=AGENT_SUPERVISOR,
        user_input=user_input,
        final_response=None,
        route_reason=None,
        metadata={
            "timestamp": datetime.now().isoformat(),
            "turn_id": 0,
        },
    )


def extract_response(state: AgentState) -> str:
    """Extract the final response string from agent state.
    
    Args:
        state: Completed agent state.
        
    Returns:
        The response string to return to the user.
    """
    # Prefer explicit final_response if set
    if state.get("final_response"):
        return state["final_response"]
    
    # Otherwise, extract from last AI message
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if _HAS_LANGCHAIN and hasattr(last_msg, "content"):
            return last_msg.content
        elif isinstance(last_msg, dict):
            return last_msg.get("content", "")
    
    return "I'm sorry, I couldn't generate a response."


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # State
    "AgentState",
    "AgentName",
    
    # Constants
    "AGENT_SUPERVISOR",
    "AGENT_IRIS",
    "AGENT_TARA",
    "AGENT_END",
    
    # Helpers
    "create_initial_state",
    "extract_response",
    
    # Re-exports for convenience
    "BaseMessage",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
]
