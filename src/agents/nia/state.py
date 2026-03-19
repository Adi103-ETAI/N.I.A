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
    Union,
)
from datetime import datetime


# =============================================================================
# Agent Names (for routing) - Pure Python, no deps
# =============================================================================

AGENT_SUPERVISOR = "supervisor"
AGENT_IRIS = "iris"       # Vision specialist
AGENT_TARA = "tara"       # Logic/reasoning specialist
AGENT_DOCKER = "docker"   # Docker execution node (skills)
AGENT_SANDBOX = "sandbox" # Static sandbox execution (Phase 3)
AGENT_COORDINATOR = "coordinator"  # Sprint 4: Multi-step coordinator
AGENT_END = "__end__"     # Terminal state

# Valid routing destinations
AgentName = Literal["supervisor", "iris", "tara", "docker", "sandbox", "coordinator", "__end__"]


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
            from langgraph.graph.message import add_messages
            _langgraph_cache["add_messages"] = add_messages
        except ImportError:
            try:
                from langgraph.graph import add_messages
                _langgraph_cache["add_messages"] = add_messages
            except ImportError:
                # Fallback: simple list append
                def add_messages(left: List, right: List) -> List:
                    return left + right
                _langgraph_cache["add_messages"] = add_messages
    return _langgraph_cache["add_messages"]


# Eagerly resolve add_messages at module load for Annotated[] usage
try:
    from langgraph.graph.message import add_messages as _add_messages_reducer
except ImportError:
    try:
        from langgraph.graph import add_messages as _add_messages_reducer
    except ImportError:
        def _add_messages_reducer(left: List, right: List) -> List:  # type: ignore
            return left + right


# =============================================================================
# Agent State Definition (Uses Any as placeholder for lazy BaseMessage)
# =============================================================================

# =============================================================================
# Agent State Definition (TypedDict for LangGraph v1)
# =============================================================================

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

class AgentState(TypedDict, total=False):
    """
    State shared across all agents in the NIA supervisor graph.

    Attributes:
        messages: Conversation history. Uses add_messages reducer for safe
                  parallel writes — LangGraph merges lists automatically.
        next: Name of the next agent to execute.
        user_input: Original user input for the current turn.
        final_response: The response to return to the user.
        route_reason: Why the supervisor chose this route.
        metadata: Additional context and routing metadata.
        session_id: Persistent session ID for checkpointing.
        sandbox_result: Output from the last container execution.
        subagent_results: Accumulated results from spawned subagents.
    """
    messages: Annotated[List[Any], _add_messages_reducer]  # safe parallel writes
    next: AgentName
    user_input: str
    final_response: Optional[str]
    route_reason: Optional[str]
    metadata: Dict[str, Any]
    session_id: str
    sandbox_result: Optional[str]   # output from static Docker sandbox
    subagent_results: List[str]     # summaries from spawned subagents


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
    
    import uuid
    session_id = str(uuid.uuid4())
    
    return {
        "messages": messages,
        "next": AGENT_SUPERVISOR,
        "user_input": user_input,
        "final_response": None,
        "route_reason": None,
        "session_id": session_id,
        "sandbox_result": None,
        "subagent_results": [],
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "turn_id": 0,
        },
    }


def safe_get_content(msg: Any) -> str:
    """
    Robustly extract string content from a message.
    
    Handles LangChain v1 content blocks where msg.content 
    can be a list of dicts (e.g. tool calls or multimodal).
    
    Args:
        msg: Message object (BaseMessage, dict, or str)
        
    Returns:
        Extracted string content or empty string.
    """
    if isinstance(msg, str):
        return msg
        
    # Get raw content
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
        
    if content is None:
        return ""
        
    # Case 1: Simple string
    if isinstance(content, str):
        return content
        
    # Case 2: List of content blocks (LangChain v1 / Anthropic / OpenAI)
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                # Extract 'text' from block types
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                # Ignore 'tool_use' or 'image' blocks for text extraction
        return "".join(text_parts)
        
    return str(content)


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
        return safe_get_content(last_msg)
    
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
    "AGENT_COORDINATOR",
    "AGENT_END",

    # Helpers
    "create_initial_state",
    "extract_response",
    "safe_get_content",

    # Lazy-loaded re-exports for convenience
    "BaseMessage",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
]
