"""N.I.A. Graph Builder - NIAGraph class and public API.

This module contains the main NIAGraph class that builds and manages
the LangGraph state machine for the NIA supervisor architecture.
"""
from __future__ import annotations

from core.logger import setup_logger
import sqlite3
from pathlib import Path
from typing import Any, Optional, List

logger = setup_logger("NIA.Graph")

# Import state and nodes
from nia.state import (
    AgentState,
    AGENT_SUPERVISOR,
    AGENT_IRIS,
    AGENT_TARA,
    AGENT_END,
    create_initial_state,
    extract_response,
)
from nia.agent import SupervisorAgent
from .nodes import (
    supervisor_node,
    iris_node,
    general_assistant,   # Chat/non-automation node
    call_tara_2,         # TARA 2.0 automation node
    route_from_tara,
    route_from_supervisor,
)

# TARA 2.0 is now the only TARA - no legacy fallback needed

# Try to import real IrisAgent
try:
    from iris.agent import IrisAgent
    _HAS_IRIS = True
except ImportError:
    _HAS_IRIS = False
    IrisAgent = None  # type: ignore

# Logger already initialized at top of file via setup_logger("NIA.Graph")

# =============================================================================
# LangGraph Imports
# =============================================================================

try:
    from langgraph.graph import StateGraph, END
    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False
    StateGraph = None  # type: ignore
    END = "__end__"
    logger.warning("langgraph not installed. Install with: pip install langgraph")

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

DEFAULT_STATE_DB = Path("data/state.db")


# =============================================================================
# NIAGraph Class
# =============================================================================

class NIAGraph:
    """LangGraph-based state machine for NIA supervisor architecture.
    
    The graph structure:
    ```
    [START] → [supervisor] → routing decision
                               ├── direct response → [END]
                               ├── IRIS → [iris] → [END]
                               └── TARA → [tara] → [END] or [supervisor]
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
        
        try:
            from core.memory import get_memory_manager
            self.memory = get_memory_manager()
        except Exception:
            self.memory = None
            logger.debug("MemoryManager not available for skill tracking")
        
        if _HAS_IRIS and IrisAgent:
            self.iris = IrisAgent(temperature=temperature)
            logger.info("👁️ IRIS agent initialized (vision enabled)")
        else:
            self.iris = None
            logger.warning("⚠️ IRIS agent not available (vision disabled)")
        
        # TARA 2.0: No legacy agent needed - call_tara_2 node handles everything
        logger.info("🛠️ TARA 2.0 will be initialized via call_tara_2 node")
        
        self.supervisor = SupervisorAgent(
            tara_agent=None,  # TARA 2.0 doesn't need legacy agent
            iris_agent=self.iris,
            model_type=model_type,
            temperature=temperature,
        )
        
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
            
            # Create SQLite connection
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            
            # Create checkpointer
            checkpointer = SqliteSaver(self._conn)
            logger.debug("SQLite checkpointer initialized: %s", db_path)
            return checkpointer
            
        except Exception as exc:
            logger.warning("Failed to initialize checkpointer: %s", exc)
            return None
    
    def _build_graph(self) -> None:
        """Build the LangGraph state machine."""
        # Create the graph with AgentState schema
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node(AGENT_SUPERVISOR, lambda s: supervisor_node(
            s, self.supervisor, 
            getattr(self.supervisor, '_llm', None)
        ))
        graph.add_node(AGENT_IRIS, lambda s: iris_node(s, self.iris))
        
        # TARA 2.0 Node (hardcoded - no legacy fallback)
        graph.add_node(AGENT_TARA, call_tara_2)
        logger.info("🚀 TARA 2.0 node registered")
        
        # General Assistant Node (for chat responses)
        graph.add_node("general", general_assistant)
        logger.info("💬 General Assistant node registered")
        
        # Set entry point
        graph.set_entry_point(AGENT_SUPERVISOR)
        
        # Add conditional edges from supervisor
        # Routes to: TARA (automation), IRIS (vision), general (chat), or END
        graph.add_conditional_edges(
            AGENT_SUPERVISOR,
            route_from_supervisor,
            {
                AGENT_IRIS: AGENT_IRIS,
                AGENT_TARA: AGENT_TARA,
                "general": "general",
                AGENT_END: END,
            }
        )
        
        # Terminal edges: All specialists return to END
        graph.add_edge(AGENT_IRIS, END)
        graph.add_edge("general", END)
        
        # Add conditional edges from TARA (allows looping back for Active Listening)
        graph.add_conditional_edges(
            AGENT_TARA,
            route_from_tara,
            {
                AGENT_SUPERVISOR: AGENT_SUPERVISOR,
                AGENT_END: END,
            }
        )
        
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
    
    def run(
        self,
        user_input: str,
        thread_id: str = "default",
    ) -> str:
        """Run the graph with user input and return response.
        
        Args:
            user_input: The user's message.
            thread_id: Conversation thread ID for persistence.
            
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
            try:
                final_state = self._compiled.invoke(initial_state, config)
                return extract_response(final_state)
            except Exception as exc:
                logger.exception("Graph execution failed: %s", exc)
                return f"I encountered an error: {str(exc)}"
        else:
            return self._fallback_run(initial_state)
    
    def _fallback_run(self, state: AgentState) -> str:
        """Fallback execution when LangGraph is not available."""
        try:
            result = self.supervisor.process(state)
            return extract_response(result)
        except Exception as exc:
            logger.exception("Fallback execution failed: %s", exc)
            return f"I'm sorry, I encountered an error: {str(exc)}"
    
    async def arun(
        self,
        user_input: str,
        thread_id: str = "default",
    ) -> str:
        """Async version of run.
        
        Note:
            Currently wraps synchronous execution. LangGraph's ainvoke
            requires careful state handling for complex agent graphs.
        """
        # Sync-to-async wrapper (stable implementation)
        return self.run(user_input, thread_id)
    
    def get_thread_history(self, thread_id: str) -> List:
        """Get conversation history for a thread."""
        if not self._checkpointer:
            return []
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = self._checkpointer.get(config)
            if checkpoint and "channel_values" in checkpoint:
                return checkpoint["channel_values"].get("messages", [])
        except Exception as exc:
            logger.error("Failed to get thread history: %s", exc)
        return []
    
    def clear_thread(self, thread_id: str) -> bool:
        """Clear conversation history for a thread."""
        if not self._conn:
            return False
        
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (thread_id,)
            )
            self._conn.commit()
            logger.info("Cleared thread: %s", thread_id)
            return True
        except Exception as exc:
            logger.error("Failed to clear thread: %s", exc)
            return False
    
    def close(self) -> None:
        """Close database connections."""
        if self._conn:
            self._conn.close()
            self._conn = None


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
    return graph.run(text.strip(), thread_id=thread_id)


async def aprocess_input(
    text: str,
    thread_id: str = "default",
    model_type: str = "smart",
    temperature: float = 0.7,
) -> str:
    """Async version of process_input."""
    if not text or not text.strip():
        return "I didn't receive any input. How can I help you?"
    
    graph = get_graph(model_type=model_type, temperature=temperature)
    return await graph.arun(text.strip(), thread_id=thread_id)


def get_conversation_history(thread_id: str = "default") -> list:
    """Get conversation history for a thread."""
    graph = get_graph()
    return graph.get_thread_history(thread_id)


def clear_conversation(thread_id: str = "default") -> bool:
    """Clear conversation history for a thread."""
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
