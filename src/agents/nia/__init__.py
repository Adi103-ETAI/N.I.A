"""N.I.A. - Neural Intelligence Assistant.

A LangGraph-based supervisor architecture for intelligent query routing
and multi-agent conversation handling.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                         NIA System                              │
    │                                                                 │
    │  [User Input] → [Supervisor] → routing decision                 │
    │                      │                                          │
    │          ┌───────────┼───────────┐                              │
    │          ▼           ▼           ▼                              │
    │     [Direct]     [IRIS]      [TARA]                             │
    │      Chat        Vision      Logic                              │
    │          │           │           │                              │
    │          └───────────┴───────────┘                              │
    │                      │                                          │
    │                      ▼                                          │
    │               [Response] → [Voice Output via NOLA]              │
    └─────────────────────────────────────────────────────────────────┘

Components:
    - SupervisorAgent: Routes queries and handles general conversation
    - IrisAgent: Vision specialist
    - TaraAgent: Tool execution specialist

Quick Start:
    from src.agents.nia import process_input
    
    response = process_input("Hello, who are you?")
    print(response)  # "Hello! I'm N.I.A., your Neural Intelligence Assistant..."

Version: 4.0.0

LAZY LOADING:
    Uses Python's ``__getattr__`` pattern to defer heavy imports
    (LangGraph, LangChain, ModelManager, TARA tools) until first access.
    This keeps boot time under 5 seconds.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from src.core.logger import setup_logger

# Module logger
logger = setup_logger("NIA")

# =============================================================================
# Lightweight Imports (loaded immediately - no heavy dependencies)
# =============================================================================

# State definitions are lightweight (just TypedDict and string constants)
from .state import (
    AgentState,
    AgentName,
    AGENT_SUPERVISOR,
    AGENT_IRIS,
    AGENT_TARA,
    AGENT_END,
    create_initial_state,
    extract_response,
)


# =============================================================================
# TYPE_CHECKING Block: IDE-only imports (no runtime cost)
# =============================================================================

if TYPE_CHECKING:
    # These imports are for static analysis ONLY - not loaded at runtime
    from .agent import SupervisorAgent as _SupervisorAgent
    from .graph import (
        NIAGraph as _NIAGraph,
        get_graph as _get_graph,
        process_input as _process_input,
        aprocess_input as _aprocess_input,
        get_conversation_history as _get_conversation_history,
        clear_conversation as _clear_conversation,
    )


# =============================================================================
# Package Metadata
# =============================================================================

__version__ = "3.1.0"
__author__ = "NIA Team"

__all__ = [
    # Main interface (lazy-loaded)
    "process_input",
    "aprocess_input",
    "get_conversation_history",
    "clear_conversation",
    
    # Graph (lazy-loaded)
    "NIAGraph",
    "get_graph",
    
    # Agents (lazy-loaded)
    "SupervisorAgent",
    
    # State (already imported - lightweight)
    "AgentState",
    "AgentName",
    "AGENT_SUPERVISOR",
    "AGENT_IRIS",
    "AGENT_TARA",
    "AGENT_END",
    "create_initial_state",
    "extract_response",
    
    # Convenience functions
    "check_dependencies",
    "print_status",
]


# =============================================================================
# Lazy Import Cache & __getattr__ (Python 3.7+ Module-Level Lazy Loading)
# =============================================================================

_lazy_cache: dict = {}

# Names that trigger graph submodule loading
_GRAPH_NAMES = frozenset({
    "NIAGraph",
    "get_graph", 
    "process_input",
    "aprocess_input",
    "get_conversation_history",
    "clear_conversation",
})


def __getattr__(name: str):
    """Lazy-load heavy modules on first access.
    
    This is a Python 3.7+ feature that enables module-level lazy loading.
    When you access `nia.process_input`, this function is called if
    `process_input` is not already defined in the module namespace.
    
    Args:
        name: Attribute name being accessed.
        
    Returns:
        The requested attribute (loaded on first access).
        
    Raises:
        AttributeError: If the attribute doesn't exist.
    """
    # Return from cache if already loaded
    if name in _lazy_cache:
        return _lazy_cache[name]
    
    # --- SupervisorAgent ---
    if name == "SupervisorAgent":
        from .persona.supervisor import SupervisorAgent
        _lazy_cache["SupervisorAgent"] = SupervisorAgent
        return SupervisorAgent
    
    # --- Graph module exports (all loaded together) ---
    if name in _GRAPH_NAMES:
        from . import graph as _graph_module
        
        # Cache all graph exports at once (they're bundled anyway)
        _lazy_cache["NIAGraph"] = _graph_module.NIAGraph
        _lazy_cache["get_graph"] = _graph_module.get_graph
        _lazy_cache["process_input"] = _graph_module.process_input
        _lazy_cache["aprocess_input"] = _graph_module.aprocess_input
        _lazy_cache["get_conversation_history"] = _graph_module.get_conversation_history
        _lazy_cache["clear_conversation"] = _graph_module.clear_conversation
        
        return _lazy_cache[name]
    
    # --- Attribute not found ---
    raise AttributeError(f"module 'nia' has no attribute '{name}'")


# =============================================================================
# Convenience Functions (Lightweight - defined here, not lazy)
# =============================================================================

def check_dependencies() -> dict:
    """Check availability of required dependencies.
    
    Returns:
        Dict mapping dependency names to availability status.
    """
    import os
    deps = {}
    
    # Check for packages (existence, not functionality)
    try:
        import langchain_core
        deps["langchain"] = True
    except ImportError:
        deps["langchain"] = False
    
    try:
        import langgraph
        deps["langgraph"] = True
    except ImportError:
        deps["langgraph"] = False
    
    try:
        from dotenv import load_dotenv
        deps["python-dotenv"] = True
    except ImportError:
        deps["python-dotenv"] = False
    
    # Check for API keys
    deps["NVIDIA_API_KEY"] = bool(os.environ.get("NVIDIA_API_KEY"))
    deps["OPENAI_API_KEY"] = bool(os.environ.get("OPENAI_API_KEY"))
    
    return deps


def print_status() -> None:
    """Print NIA system status (v3.1: Uses logger instead of print)."""
    deps = check_dependencies()
    
    logger.info("\n" + "=" * 50)
    logger.info("  N.I.A. System Status")
    logger.info("=" * 50)
    
    for name, available in deps.items():
        status = "✓ Available" if available else "✗ Missing"
        logger.info(f"  {name}: {status}")
    
    if all(deps.values()):
        logger.info("  ✅ All dependencies installed. NIA is ready!")
    else:
        missing = [k for k, v in deps.items() if not v]
        logger.warning(f"  ⚠️  Missing: {', '.join(missing)}")
        api_keys = [k for k in missing if "API_KEY" in k]
        if api_keys:
            logger.warning(f"     Set {', '.join(api_keys)} in .env file")
        pkg_missing = [k for k in missing if "API_KEY" not in k]
        if pkg_missing:
            logger.warning(f"     Install packages: uv add {' '.join(pkg_missing)}")

