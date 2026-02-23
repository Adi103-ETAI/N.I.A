"""N.I.A. Graph Builder - NIAGraph class and public API.

This module contains the main NIAGraph class that builds and manages
the LangGraph state machine for the NIA supervisor architecture.
"""
from __future__ import annotations

from src.core.logger import setup_logger
from pathlib import Path
from typing import Any, Optional, List
try:
    import aiosqlite
except ImportError:
    aiosqlite = None

logger = setup_logger("NIA.Graph")

# Import state and nodes
from src.agents.nia.state import (
    AgentState,
    AGENT_SUPERVISOR,
    AGENT_IRIS,
    AGENT_TARA,
    AGENT_DOCKER, # Phase 2
    AGENT_END,
    create_initial_state,
    extract_response,
)
from src.agents.nia.persona.supervisor import SupervisorAgent
from .nodes import (
    supervisor_node,
    iris_node,
    call_tara_2,         # TARA 2.0 automation node
    docker_node,         # Phase 2: Docker Execution
    route_from_tara,
    router_node,         # Phase 3: AI Router
    route_from_router,   # Phase 3: Routing Logic
)

# TARA 2.0 is now the only TARA - no legacy fallback needed

# Try to import real IrisAgent
try:
    from src.agents.iris.agent import IrisAgent
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
    logger.warning("langgraph not installed. Install with: uv add langgraph")

# Async Checkpointer Only
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    _HAS_ASYNC_CHECKPOINTER = True
except ImportError:
    try:
        from langgraph_checkpoint_sqlite.aio import AsyncSqliteSaver
        _HAS_ASYNC_CHECKPOINTER = True
    except ImportError:
        _HAS_ASYNC_CHECKPOINTER = False
        AsyncSqliteSaver = None 
        logger.debug("AsyncSqliteSaver not available.")


# =============================================================================
# Default Paths
# =============================================================================

DEFAULT_STATE_DB = Path("data/state.db")


# =============================================================================
# NIAGraph Class
# =============================================================================

class NIAGraph:
    """LangGraph-based state machine for NIA supervisor architecture.
    
    Phase 3 Graph Structure:
    ```
    [START] → [router] → routing decision
                            ├── chat    → [supervisor] → [END]
                            ├── swarm   → [docker]     → [END]
                            ├── system  → [tara]       → [END] or [supervisor]
                            └── iris    → [iris]       → [END]
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
        self.enable_persistence = enable_persistence and _HAS_ASYNC_CHECKPOINTER
        
        try:
            from src.core.di import ServiceRegistry
            self.memory = ServiceRegistry.get("memory")
            if self.memory is None:
                # Fallback: create and register if engine hasn't done it yet
                from src.core.memory import get_memory_manager
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
            iris_agent=self.iris,
            model_type=model_type,
            temperature=temperature,
        )
        
        # Persistence setup
        self._db_path = state_db_path or str(DEFAULT_STATE_DB)
        # self._conn and self._checkpointer are not used in async mode (created on-demand)
        
        # Build the graph
        self._graph = None
        self._compiled = None
        
        if _HAS_LANGGRAPH:
            self._build_graph()
        else:
            logger.warning("LangGraph not available. Using fallback execution.")
    
    # No sync checkpointer initialization
    def _init_checkpointer(self) -> None:
        """Deprecated."""
        return None
    
    def _build_graph(self) -> None:
        """Build the LangGraph state machine."""
        # Create the graph with AgentState schema
        graph = StateGraph(AgentState)
        
        # Add nodes
        # Define async wrappers to bind arguments
        # CRITICAL FIX: explicit async def prevents "result is coroutine" errors
        async def run_supervisor(state):
            return await supervisor_node(
                state, 
                self.supervisor, 
                getattr(self.supervisor, '_llm', None)
            )

        async def run_iris(state):
            return await iris_node(state, self.iris)

        async def run_router(state):
             # Router is stateless in terms of class instance, but needs async
             return await router_node(state)

        # Add nodes with async wrappers
        graph.add_node("router", run_router)
        graph.add_node(AGENT_SUPERVISOR, run_supervisor)
        graph.add_node(AGENT_IRIS, run_iris)
        
        # TARA 2.0 Node (hardcoded - no legacy fallback)
        graph.add_node(AGENT_TARA, call_tara_2)
        logger.info("🚀 TARA 2.0 node registered")
        
        # Phase 2: Docker Node
        graph.add_node(AGENT_DOCKER, docker_node)
        logger.info("🐳 Docker Node registered")
        
        
        # Set entry point
        graph.set_entry_point("router")
        
        # Add conditional edges from ROUTER
        graph.add_conditional_edges(
            "router",
            route_from_router,
            {
                AGENT_SUPERVISOR: AGENT_SUPERVISOR, # Chat
                AGENT_IRIS: AGENT_IRIS,
                AGENT_TARA: AGENT_TARA,
                AGENT_DOCKER: AGENT_DOCKER,
                "tara": AGENT_TARA,         # Alias
                "docker": AGENT_DOCKER,     # Alias
                "iris": AGENT_IRIS,         # Alias
                "supervisor": AGENT_SUPERVISOR, # Alias
            }
        )
        
        # Supervisor now just chats and ends (or could loop, but let's keep it simple for now)
        graph.add_edge(AGENT_SUPERVISOR, END)
        
        # Terminal edges: All specialists return to END
        graph.add_edge(AGENT_IRIS, END)
        # graph.add_edge("general", END) # General is now handled by Supervisor
        graph.add_edge(AGENT_DOCKER, END)
        
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
        # In Async mode, we do NOT compile with a static checkpointer here.
        # We compile ON-DEMAND in arun() with the async context manager.
        
        # Compile the graph (without persistence for sync fallback/structure)
        self._graph = graph
        self._compiled = graph.compile()
        logger.info("NIA graph structure compiled (persistence determined at runtime)")
    
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
                # Sync run does NOT support persistence in this async-first architecture
                # It uses the stateless compiled graph
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
        """Async version of run (Native).
        
        Uses LangGraph's ainvoke for non-blocking execution.
        Handles async persistence via AsyncSqliteSaver.
        """
        # Create initial state
        initial_state = create_initial_state(user_input)
        
        # Build config with thread ID for checkpointing
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        
        # 🌊 ASYNC PERSISTENCE IMPLEMENTATION
        # Use AsyncSqliteSaver as a context manager to ensure non-blocking cleanup
        if self.enable_persistence and _HAS_ASYNC_CHECKPOINTER:
            try:
                # Ensure db directory exists
                db_path = Path(self._db_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)

                # Use context manager for auto-closing connection
                async with AsyncSqliteSaver.from_conn_string(self._db_path) as checkpointer:
                    # Compile the graph structure with async checkpointer
                    # This is lightweight as the graph definition is cached in self._graph
                    app = self._graph.compile(checkpointer=checkpointer)
                    
                    # Invoke async
                    final_state = await app.ainvoke(initial_state, config)
                    return extract_response(final_state)
            except Exception as exc:
                logger.exception("Async graph execution failed: %s", exc)
                return f"I encountered an error: {str(exc)}"
                
        # Fallback: Stateless execution
        elif self._compiled:
            try:
                # Native async invoke on stateless graph
                final_state = await self._compiled.ainvoke(initial_state, config)
                return extract_response(final_state)
            except Exception as exc:
                logger.exception("Graph execution failed: %s", exc)
                return f"I encountered an error: {str(exc)}"
        else:
            # Fallback for no LangGraph
            import asyncio
            return await asyncio.to_thread(self._fallback_run, initial_state)
    
    async def aget_thread_history(self, thread_id: str) -> List:
        """Get conversation history for a thread (Async)."""
        if not self.enable_persistence or not _HAS_ASYNC_CHECKPOINTER:
            return []
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            async with AsyncSqliteSaver.from_conn_string(self._db_path) as checkpointer:
                checkpoint = await checkpointer.aget(config)
                if checkpoint and "channel_values" in checkpoint:
                    return checkpoint["channel_values"].get("messages", [])
        except Exception as exc:
            logger.error("Failed to get thread history: %s", exc)
        return []
    
    def get_thread_history(self, thread_id: str) -> List:
        """Sync wrapper for aget_thread_history (Not recommended)."""
        import asyncio
        try:
             return asyncio.run(self.aget_thread_history(thread_id))
        except RuntimeError:
             # If loop already running, we can't use asyncio.run
             logger.warning("get_thread_history called from async context - returning empty. Use aget_thread_history.")
             return []

    async def aclear_thread(self, thread_id: str) -> bool:
        """Clear conversation history for a thread (Async)."""
        try:
            db_path = Path(self._db_path)
            if not db_path.exists():
                return False
                
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                await db.commit()
                logger.info("Cleared thread: %s", thread_id)
                return True
        except Exception as exc:
            logger.error("Failed to clear thread: %s", exc)
            return False

    def clear_thread(self, thread_id: str) -> bool:
        """Sync wrapper for aclear_thread."""
        import asyncio
        try:
            return asyncio.run(self.aclear_thread(thread_id))
        except RuntimeError:
             return False


# =============================================================================
# ServiceRegistry Integration
# =============================================================================

def get_graph(
    model_type: str = "smart",
    temperature: float = 0.7,
    state_db_path: Optional[str] = None,
    enable_persistence: bool = True,
    force_new: bool = False,
) -> NIAGraph:
    """Get or create the NIAGraph via ServiceRegistry.
    
    The NIAGraph is registered as "graph" in the ServiceRegistry.
    If not yet registered, it will be created and registered automatically.
    
    Args:
        model_type: Type of model to use ('smart' or 'fast').
        temperature: Sampling temperature.
        state_db_path: Path to SQLite database for persistence.
        enable_persistence: Whether to enable conversation persistence.
        force_new: If True, create a new instance (replaces existing).
        
    Returns:
        NIAGraph instance.
    """
    from src.core.di import ServiceRegistry
    
    graph = ServiceRegistry.get("graph")
    
    if graph is None or force_new:
        # Close existing graph if forcing new
        if graph is not None and force_new:
            graph.close()
        
        graph = NIAGraph(
            model_type=model_type,
            temperature=temperature,
            state_db_path=state_db_path,
            enable_persistence=enable_persistence,
        )
        ServiceRegistry.register("graph", graph)
        logger.info("NIAGraph registered in ServiceRegistry")
    
    return graph


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

async def aget_conversation_history(thread_id: str = "default") -> list:
    """Async Get conversation history for a thread."""
    graph = get_graph()
    return await graph.aget_thread_history(thread_id)


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
