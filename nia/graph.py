"""N.I.A. Graph Module - LangGraph State Machine with Persistence.

This module builds the LangGraph StateGraph that orchestrates the
NIA supervisor architecture. The graph defines:
- Nodes: Supervisor, IRIS, TARA
- Edges: Routing logic based on supervisor decisions
- Entry/exit points for graph execution
- Persistence: SQLite-based checkpointing for conversation memory

The main public interface is `process_input(text, thread_id) -> str`.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

# Import state and agents
from .state import (
    AgentState,
    AGENT_SUPERVISOR,
    AGENT_IRIS,
    AGENT_TARA,
    AGENT_END,
    create_initial_state,
    extract_response,
)
from .agent import SupervisorAgent, IrisAgent
from .agent import TaraAgent as TaraAgentPlaceholder

# Try to import real TaraAgent
try:
    from tara.agent import TaraAgent as RealTaraAgent
    _HAS_TARA = True
except ImportError:
    _HAS_TARA = False
    RealTaraAgent = None  # type: ignore
    logger.debug("Real TARA not available, using placeholder")

# Configure logger
logger = logging.getLogger(__name__)

# =============================================================================
# LangGraph Imports
# =============================================================================

# Try to import LangGraph
try:
    from langgraph.graph import StateGraph, END
    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False
    StateGraph = None  # type: ignore
    END = "__end__"
    logger.warning("langgraph not installed. Install with: pip install langgraph")

# Try to import SqliteSaver for checkpointing
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _HAS_CHECKPOINTER = True
except ImportError:
    _HAS_CHECKPOINTER = False
    SqliteSaver = None  # type: ignore
    logger.debug("langgraph-checkpoint-sqlite not available. Persistence disabled.")


# =============================================================================
# Default Paths
# =============================================================================

# Default path for state database
DEFAULT_STATE_DB = Path("data/state.db")


# =============================================================================
# Graph Builder
# =============================================================================

class NIAGraph:
    """LangGraph-based state machine for NIA supervisor architecture.
    
    The graph structure:
    ```
    [START] → [supervisor] → routing decision
                               ├── direct response → [END]
                               ├── IRIS → [iris] → [END]
                               └── TARA → [tara] → [END]
    ```
    
    Persistence:
        When checkpointing is enabled, conversation state is saved to SQLite.
        Each thread_id maintains its own conversation history.
    
    Example:
        graph = NIAGraph()
        response = graph.run("Hello!", thread_id="user_123")
    """
    
    def __init__(
        self,
        model_type: str = "smart",
        temperature: float = 0.7,
        state_db_path: Optional[str] = None,
        enable_persistence: bool = True,
    ) -> None:
        """Initialize the NIA graph.
        
        Args:
            model_type: Type of model to use ('smart' or 'fast').
            temperature: Sampling temperature.
            state_db_path: Path to SQLite database for state persistence.
            enable_persistence: Whether to enable conversation persistence.
        """
        self.model_type = model_type
        self.temperature = temperature
        self.enable_persistence = enable_persistence and _HAS_CHECKPOINTER
        
        # Initialize agents
        self.supervisor = SupervisorAgent(
            model_type=model_type,
            temperature=temperature,
        )
        self.iris = IrisAgent()
        
        # Use real TARA if available, otherwise placeholder
        if _HAS_TARA and RealTaraAgent:
            self.tara = RealTaraAgent(temperature=temperature)
            logger.info("Using real TARA agent with tools")
        else:
            self.tara = TaraAgentPlaceholder()
            logger.info("Using TARA placeholder (tools not available)")
        
        # Persistence setup
        self._db_path = state_db_path or str(DEFAULT_STATE_DB)
        self._conn: Optional[sqlite3.Connection] = None
        self._checkpointer = None
        
        # Build the graph
        self._graph = None
        self._compiled = None
        
        if _HAS_LANGGRAPH:
            self._build_graph()
        else:
            logger.warning("LangGraph not available. Using fallback execution.")
    
    def _init_checkpointer(self) -> Optional[Any]:
        """Initialize the SQLite checkpointer for persistence."""
        if not self.enable_persistence or not _HAS_CHECKPOINTER:
            return None
        
        try:
            # Ensure data directory exists
            db_path = Path(self._db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create connection with thread safety disabled (we handle it)
            self._conn = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
            )
            
            # Create checkpointer
            checkpointer = SqliteSaver(self._conn)
            
            logger.info("Persistence enabled: %s", self._db_path)
            return checkpointer
            
        except Exception as exc:
            logger.warning("Failed to initialize checkpointer: %s", exc)
            return None
    
    def _build_graph(self) -> None:
        """Build the LangGraph state machine."""
        # Create the graph with AgentState schema
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node(AGENT_SUPERVISOR, self._supervisor_node)
        graph.add_node(AGENT_IRIS, self._iris_node)
        graph.add_node(AGENT_TARA, self._tara_node)
        
        # Set entry point
        graph.set_entry_point(AGENT_SUPERVISOR)
        
        # Add conditional edges from supervisor
        graph.add_conditional_edges(
            AGENT_SUPERVISOR,
            self._route_from_supervisor,
            {
                AGENT_IRIS: AGENT_IRIS,
                AGENT_TARA: AGENT_TARA,
                AGENT_END: END,
            }
        )
        
        # Add edges from specialists to END
        graph.add_edge(AGENT_IRIS, END)
        graph.add_edge(AGENT_TARA, END)
        
        # Initialize checkpointer for persistence
        self._checkpointer = self._init_checkpointer()
        
        # Compile the graph with or without checkpointer
        self._graph = graph
        if self._checkpointer:
            self._compiled = graph.compile(checkpointer=self._checkpointer)
            logger.info("NIA graph compiled with persistence")
        else:
            self._compiled = graph.compile()
            logger.info("NIA graph compiled (no persistence)")
    
    def _supervisor_node(self, state: AgentState) -> AgentState:
        """Supervisor node function."""
        logger.debug("Executing supervisor node")
        return self.supervisor.process(state)
    
    def _iris_node(self, state: AgentState) -> AgentState:
        """IRIS node function."""
        logger.debug("Executing IRIS node")
        return self.iris.process(state)
    
    def _tara_node(self, state: AgentState) -> AgentState:
        """TARA node function."""
        logger.debug("Executing TARA node")
        return self.tara.process(state)
    
    def _route_from_supervisor(self, state: AgentState) -> str:
        """Determine next node based on supervisor's routing decision."""
        next_agent = state.get("next", AGENT_END)
        route_reason = state.get("route_reason", "No reason provided")
        
        logger.debug("Routing decision: %s (reason: %s)", next_agent, route_reason)
        
        if next_agent == AGENT_IRIS:
            return AGENT_IRIS
        elif next_agent == AGENT_TARA:
            return AGENT_TARA
        else:
            return AGENT_END
    
    def run(
        self,
        user_input: str,
        thread_id: str = "default",
    ) -> str:
        """Run the graph with user input and return response.
        
        Args:
            user_input: The user's message.
            thread_id: Conversation thread ID for persistence.
                       Each thread maintains its own conversation history.
            
        Returns:
            The assistant's response as a string.
        """
        # Create initial state
        initial_state = create_initial_state(user_input)
        
        # Build config with thread ID for checkpointing
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        
        if self._compiled:
            # Run through LangGraph
            try:
                final_state = self._compiled.invoke(initial_state, config)
                return extract_response(final_state)
            except Exception as exc:
                logger.exception("Graph execution failed: %s", exc)
                return f"I encountered an error processing your request: {exc}"
        else:
            # Fallback execution without LangGraph
            return self._fallback_run(initial_state)
    
    def _fallback_run(self, state: AgentState) -> str:
        """Fallback execution when LangGraph is not available."""
        # Run supervisor
        state = self.supervisor.process(state)
        
        # Check routing
        next_agent = state.get("next", AGENT_END)
        
        if next_agent == AGENT_IRIS:
            state = self.iris.process(state)
        elif next_agent == AGENT_TARA:
            state = self.tara.process(state)
        
        return extract_response(state)
    
    async def arun(
        self,
        user_input: str,
        thread_id: str = "default",
    ) -> str:
        """Async version of run.
        
        Args:
            user_input: The user's message.
            thread_id: Conversation thread ID for persistence.
            
        Returns:
            The assistant's response as a string.
        """
        # For now, just call sync version
        # TODO: Implement proper async execution with ainvoke
        return self.run(user_input, thread_id=thread_id)
    
    def get_thread_history(self, thread_id: str) -> list:
        """Get conversation history for a thread.
        
        Args:
            thread_id: The thread ID to retrieve history for.
            
        Returns:
            List of messages in the thread.
        """
        if not self._checkpointer or not self._compiled:
            return []
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self._compiled.get_state(config)
            if state and state.values:
                return state.values.get("messages", [])
        except Exception as exc:
            logger.debug("Failed to get thread history: %s", exc)
        
        return []
    
    def clear_thread(self, thread_id: str) -> bool:
        """Clear conversation history for a thread.
        
        Args:
            thread_id: The thread ID to clear.
            
        Returns:
            True if successful.
        """
        if not self._conn:
            return False
        
        try:
            # Delete from checkpoints table
            cursor = self._conn.cursor()
            cursor.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (thread_id,)
            )
            self._conn.commit()
            logger.info("Cleared thread: %s", thread_id)
            return True
        except Exception as exc:
            logger.debug("Failed to clear thread: %s", exc)
            return False
    
    def close(self) -> None:
        """Close database connections."""
        if self._conn:
            try:
                self._conn.close()
                self._conn = None
                logger.debug("Closed state database connection")
            except Exception:
                pass


# =============================================================================
# Module-level Singleton
# =============================================================================

_graph_instance: Optional[NIAGraph] = None


def get_graph(
    model_type: str = "smart",
    temperature: float = 0.7,
    state_db_path: Optional[str] = None,
    enable_persistence: bool = True,
    force_new: bool = False,
) -> NIAGraph:
    """Get or create the NIA graph singleton.
    
    Args:
        model_type: Type of model to use ('smart' or 'fast').
        temperature: Sampling temperature.
        state_db_path: Path to SQLite database for persistence.
        enable_persistence: Whether to enable conversation persistence.
        force_new: If True, create a new instance.
        
    Returns:
        NIAGraph instance.
    """
    global _graph_instance
    
    if _graph_instance is None or force_new:
        _graph_instance = NIAGraph(
            model_type=model_type,
            temperature=temperature,
            state_db_path=state_db_path,
            enable_persistence=enable_persistence,
        )
    
    return _graph_instance


# =============================================================================
# Public Interface
# =============================================================================

def process_input(
    text: str,
    thread_id: str = "default",
    model_type: str = "smart",
    temperature: float = 0.7,
) -> str:
    """Process user input through the NIA graph and return response.
    
    This is the main public interface for the NIA system. It:
    1. Creates/retrieves the graph singleton
    2. Runs the input through supervisor → specialist routing
    3. Persists conversation state to SQLite
    4. Returns the final response as a simple string
    
    Args:
        text: The user's input text.
        thread_id: Conversation thread ID. Each thread maintains separate
                   conversation history. Use different IDs for different users
                   or conversations. Default is "default".
        model_type: Type of model ('smart' or 'fast').
        temperature: Sampling temperature (default: 0.7).
        
    Returns:
        The assistant's response as a string.
        
    Example:
        # Single user, default thread
        response = process_input("What is 2 + 2?")
        
        # Multi-user with separate threads
        response = process_input("Hello!", thread_id="user_alice")
        response = process_input("Hi there!", thread_id="user_bob")
    """
    if not text or not text.strip():
        return "I didn't receive any input. How can I help you?"
    
    graph = get_graph(model_type=model_type, temperature=temperature)
    return graph.run(text.strip(), thread_id=thread_id)


async def aprocess_input(
    text: str,
    thread_id: str = "default",
    model_type: str = "smart",
    temperature: float = 0.7,
) -> str:
    """Async version of process_input.
    
    Args:
        text: The user's input text.
        thread_id: Conversation thread ID.
        model_type: Type of model ('smart' or 'fast').
        temperature: Sampling temperature.
        
    Returns:
        The assistant's response as a string.
    """
    if not text or not text.strip():
        return "I didn't receive any input. How can I help you?"
    
    graph = get_graph(model_type=model_type, temperature=temperature)
    return await graph.arun(text.strip(), thread_id=thread_id)


def get_conversation_history(thread_id: str = "default") -> list:
    """Get conversation history for a thread.
    
    Args:
        thread_id: The thread ID to retrieve.
        
    Returns:
        List of messages.
    """
    graph = get_graph()
    return graph.get_thread_history(thread_id)


def clear_conversation(thread_id: str = "default") -> bool:
    """Clear conversation history for a thread.
    
    Args:
        thread_id: The thread ID to clear.
        
    Returns:
        True if successful.
    """
    graph = get_graph()
    return graph.clear_thread(thread_id)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "NIAGraph",
    "get_graph",
    "process_input",
    "aprocess_input",
    "get_conversation_history",
    "clear_conversation",
]
