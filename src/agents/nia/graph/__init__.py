"""N.I.A. Graph Package - LangGraph State Machine.

This package provides the LangGraph-based state machine for NIA.

Public API:
    NIAGraph: Main graph class
    get_graph: Singleton factory
    process_input: Process user input
    aprocess_input: Async version

Example:
    from src.agents.nia.graph import process_input
    response = process_input("Hello!", thread_id="user_123")
"""
from .builder import (
    NIAGraph,
    get_graph,
    process_input,
    aprocess_input,
    get_conversation_history,
    clear_conversation,
)

__all__ = [
    "NIAGraph",
    "get_graph",
    "process_input",
    "aprocess_input",
    "get_conversation_history",
    "clear_conversation",
]
