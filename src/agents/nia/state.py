"""N.I.A. State Module - Agent State Definitions.

This module defines the state structures used by the LangGraph-based
supervisor architecture for NIA (Neural Intelligence Assistant).

The state tracks:
- Conversation messages (using LangChain's BaseMessage format)
- Routing decisions (which agent should act next)
- Metadata for tracing and debugging

LAZY LOADING:
    This module defers LangChain/LangGraph imports to avoid slow boot.
    Imports happen on first USE, not at module load time.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)
from datetime import datetime


# =============================================================================
# Agent Names (for routing) - Pure Python, no deps
# =============================================================================

AGENT_SUPERVISOR = "supervisor"
AGENT_IRIS = "iris"       # Vision specialist
AGENT_TARA = "tara"       # Logic/reasoning specialist
AGENT_END = "__end__"     # Terminal state

# Valid routing destinations
AgentName = Literal["supervisor", "iris", "tara", "__end__"]


# =============================================================================
# TYPE_CHECKING Block: IDE-only imports (no runtime cost)
# =============================================================================

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


# =============================================================================
# Lazy Import Helpers
# =============================================================================

_langchain_cache: Dict[str, Any] = {}
_langgraph_cache: Dict[str, Any] = {}


def _get_langchain_messages():
    """Lazy-load LangChain message classes."""
    if "loaded" not in _langchain_cache:
        try:
            from langchain_core.messages import (
                BaseMessage,
                HumanMessage,
                AIMessage,
                SystemMessage,
            )
            _langchain_cache["BaseMessage"] = BaseMessage
            _langchain_cache["HumanMessage"] = HumanMessage
            _langchain_cache["AIMessage"] = AIMessage
            _langchain_cache["SystemMessage"] = SystemMessage
            _langchain_cache["loaded"] = True
        except ImportError:
            _langchain_cache["BaseMessage"] = Any
            _langchain_cache["HumanMessage"] = dict
            _langchain_cache["AIMessage"] = dict
            _langchain_cache["SystemMessage"] = dict
            _langchain_cache["loaded"] = False
    return _langchain_cache


def _get_add_messages() -> Callable[[List, List], List]:
    """Lazy-load LangGraph's add_messages reducer."""
    if "add_messages" not in _langgraph_cache:
        try:
            from langgraph.graph import add_messages
            _langgraph_cache["add_messages"] = add_messages
        except ImportError:
            # Fallback: simple list append
            def add_messages(left: List, right: List) -> List:
                return left + right
            _langgraph_cache["add_messages"] = add_messages
    return _langgraph_cache["add_messages"]


# =============================================================================
# Agent State Definition (Uses Any as placeholder for lazy BaseMessage)
# =============================================================================

class AgentState(Dict[str, Any]):
    """State shared across all agents in the NIA supervisor graph.
    
    This is a TypedDict-like class that defines the structure of state 
    passed between nodes in the LangGraph execution.
    
    Attributes:
        messages: Conversation history (accumulated across turns).
                  Uses LangGraph's add_messages reducer for proper merging.
        next: Name of the next agent to execute. Set by supervisor routing.
        user_input: Original user input for the current turn.
        final_response: The response to return to the user.
        route_reason: Why the supervisor chose this route (for debugging).
        metadata: Additional context (timestamps, turn counts, etc.).
    
    Note:
        We use a Dict subclass instead of TypedDict to allow dynamic
        attribute access while maintaining type hints via TYPE_CHECKING.
    """
    messages: Sequence[Any]  # BaseMessage at runtime
    next: AgentName
    user_input: str
    final_response: Optional[str]
    route_reason: Optional[str]
    metadata: Dict[str, Any]


# Also export as TypedDict for LangGraph compatibility
try:
    from typing import TypedDict
    
    class _AgentStateTypedDict(TypedDict, total=False):
        """TypedDict version for LangGraph state channel definitions."""
        messages: Sequence[Any]
        next: str
        user_input: str
        final_response: Optional[str]
        route_reason: Optional[str]
        metadata: Dict[str, Any]
    
    # Use TypedDict for actual usage (LangGraph expects this)
    AgentState = _AgentStateTypedDict  # type: ignore
except ImportError:
    pass  # Keep the Dict subclass as fallback


# =============================================================================
# State Factory Functions
# =============================================================================

def create_initial_state(user_input: str) -> dict:
    """Create an initial state for a new conversation turn.
    
    Args:
        user_input: The user's message to process.
        
    Returns:
        AgentState ready for graph execution.
    """
    lc = _get_langchain_messages()
    
    if lc.get("loaded"):
        HumanMessage = lc["HumanMessage"]
        messages = [HumanMessage(content=user_input)]
    else:
        messages = [{"role": "user", "content": user_input}]
    
    return {
        "messages": messages,
        "next": AGENT_SUPERVISOR,
        "user_input": user_input,
        "final_response": None,
        "route_reason": None,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "turn_id": 0,
        },
    }


def extract_response(state: dict) -> str:
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
        # Check if it's a LangChain message with content attribute
        if hasattr(last_msg, "content"):
            return last_msg.content
        elif isinstance(last_msg, dict):
            return last_msg.get("content", "")
    
    return "I'm sorry, I couldn't generate a response."


# =============================================================================
# Backward Compatibility: Module-level exports for existing code
# =============================================================================

def __getattr__(name: str):
    """Lazy-load LangChain message classes on demand."""
    if name in ("BaseMessage", "HumanMessage", "AIMessage", "SystemMessage"):
        lc = _get_langchain_messages()
        return lc.get(name)
    
    if name == "add_messages":
        return _get_add_messages()
    
    raise AttributeError(f"module 'nia.state' has no attribute '{name}'")


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
    
    # Lazy-loaded re-exports for convenience
    "BaseMessage",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
]
