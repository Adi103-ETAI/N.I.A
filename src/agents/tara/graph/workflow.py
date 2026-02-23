"""TARA Graph Workflow — LangGraph StateGraph Assembly & Compilation.

Assembles all TARA nodes into a compiled ``StateGraph`` application and
provides convenience wrappers for invoking it.

Graph topology::

    ┌─────────────────────────────────────────────────────────────┐
    │                     TARA SubGraph                           │
    │                                                             │
    │   ┌─────────┐    tool_calls    ┌─────────┐                  │
    │   │         │ ──────────────►  │         │                  │
    │   │ REASON  │                  │  ACTION │                  │
    │   │         │ ◄────────────────│         │                  │
    │   └────┬────┘   loop back      └─────────┘                  │
    │        │                                                    │
    │        │ no_tools                                           │
    │        ▼                                                    │
    │      [END]                                                  │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from src.agents.tara.graph import tara_app, create_initial_tara_state
    
    state = create_initial_tara_state("Open Notepad and type hello")
    result = tara_app.invoke(state)
    print(result["final_response"])
"""
from __future__ import annotations

from typing import Optional

from src.core.logger import setup_logger

logger = setup_logger("TARA.Workflow")

# =============================================================================
# LangGraph Imports
# =============================================================================

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False
    StateGraph = None  # type: ignore
    END = "__end__"
    MemorySaver = None  # type: ignore
    logger.warning("langgraph not installed - workflow disabled")

# Internal imports
from .state import TaraState
from .nodes import reasoner, tool_executor, should_continue


# =============================================================================
# Graph Builder
# =============================================================================

def build_tara_graph(with_memory: bool = False):
    """
    Build and compile the TARA StateGraph.
    
    Args:
        with_memory: Enable MemorySaver for conversation persistence.
        
    Returns:
        Compiled LangGraph application.
    """
    if not _HAS_LANGGRAPH:
        raise RuntimeError("langgraph not installed. Run: uv add langgraph")
    
    logger.debug("Building TARA SubGraph...")
    
    # Initialize StateGraph with TaraState schema
    workflow = StateGraph(TaraState)
    
    # =========================================================================
    # Add Nodes
    # =========================================================================
    
    # Reasoner: Generates tool calls using dynamic context
    workflow.add_node("reasoner", reasoner)
    
    # Action: Executes tools and updates state
    workflow.add_node("action", tool_executor)
    
    # =========================================================================
    # Set Entry Point
    # =========================================================================
    
    workflow.set_entry_point("reasoner")
    
    # =========================================================================
    # Add Edges
    # =========================================================================
    
    # Conditional edge from reasoner
    # Based on should_continue(), goes to "action" or END
    workflow.add_conditional_edges(
        "reasoner",
        should_continue,
        {
            "tool_executor": "action",  # If tools pending
            "reasoner": "reasoner",      # If need more reasoning
            "__end__": END,              # If done
        }
    )
    
    # After action, always loop back to reasoner
    workflow.add_edge("action", "reasoner")
    
    # =========================================================================
    # Compile
    # =========================================================================
    
    # Optional memory saver for persistence
    checkpointer = None
    if with_memory and MemorySaver:
        checkpointer = MemorySaver()
        logger.info("Memory saver enabled")
    
    # Compile the graph
    tara_app = workflow.compile(checkpointer=checkpointer)
    
    logger.debug("TARA SubGraph compiled successfully")
    return tara_app


# =============================================================================
# Cached Instance
# =============================================================================

_cached_app = None


def get_tara_graph(with_memory: bool = False):
    """
    Get or create the TARA graph instance.
    
    Uses caching for performance; first call builds the graph.
    
    Args:
        with_memory: Enable memory persistence.
        
    Returns:
        Compiled LangGraph application.
    """
    global _cached_app
    
    if _cached_app is None:
        _cached_app = build_tara_graph(with_memory=with_memory)
    
    return _cached_app


# =============================================================================
# Convenience Runner
# =============================================================================

def run_tara(user_goal: str, initial_context: dict = None) -> str:
    """
    High-level function to run TARA on a user goal.
    
    Args:
        user_goal: What the user wants to achieve.
        initial_context: Optional initial state values.
        
    Returns:
        TARA's final response string.
    """
    from .state import create_initial_tara_state
    
    # Build initial state
    state = create_initial_tara_state(user_goal)
    
    # Add any initial context
    if initial_context:
        state.update(initial_context)
    
    # Get graph and invoke
    app = get_tara_graph()
    result = app.invoke(state)
    
    # Extract response
    return result.get("final_response", "Task completed.")


# =============================================================================
# Pre-compiled App (for direct import)
# =============================================================================

# Build on import if langgraph available
tara_app = None
if _HAS_LANGGRAPH:
    try:
        tara_app = build_tara_graph()
    except Exception as e:
        logger.error(f"Failed to pre-compile TARA graph: {e}")
        tara_app = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "build_tara_graph",
    "get_tara_graph",
    "run_tara",
    "tara_app",
]
