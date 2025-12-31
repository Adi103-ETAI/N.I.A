"""N.I.A. - Neural Intelligence Assistant.

A LangGraph-based supervisor architecture for intelligent query routing
and multi-agent conversation handling.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                         NIA System                               │
    │                                                                  │
    │  [User Input] → [Supervisor] → routing decision                 │
    │                      │                                           │
    │          ┌───────────┼───────────┐                              │
    │          ▼           ▼           ▼                              │
    │     [Direct]     [IRIS]      [TARA]                            │
    │      Chat        Vision      Logic                              │
    │          │           │           │                              │
    │          └───────────┴───────────┘                              │
    │                      │                                           │
    │                      ▼                                           │
    │               [Response] → [Voice Output via NOLA]              │
    └─────────────────────────────────────────────────────────────────┘

Components:
    - SupervisorAgent: Routes queries and handles general conversation
    - IrisAgent: Vision specialist (placeholder - coming soon)
    - TaraAgent: Logic/reasoning specialist (placeholder - coming soon)

Quick Start:
    from nia import process_input
    
    response = process_input("Hello, who are you?")
    print(response)  # "Hello! I'm N.I.A., your Neural Intelligence Assistant..."
    
    response = process_input("What's in this image?")
    print(response)  # Routes to IRIS (placeholder response)
    
    response = process_input("Solve 2x + 5 = 15")
    print(response)  # Routes to TARA (placeholder response)

Integration with NOLA:
    from nia import process_input
    from nola import NOLAManager
    
    nola = NOLAManager()
    nola.start()
    
    while True:
        result = nola.get_input(timeout=0.5)
        if result:
            response = process_input(result.text)
            nola.speak(response)

Version: 1.0.0
"""
from __future__ import annotations

# State definitions
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

# Agent implementations
from .agent import (
    SupervisorAgent,
    IrisAgent,
    TaraAgent,
)

# Graph and execution
from .graph import (
    NIAGraph,
    get_graph,
    process_input,
    aprocess_input,
    get_conversation_history,
    clear_conversation,
)

# Package metadata
__version__ = "1.0.0"
__author__ = "NIA Team"

__all__ = [
    # Main interface
    "process_input",
    "aprocess_input",
    "get_conversation_history",
    "clear_conversation",
    
    # Graph
    "NIAGraph",
    "get_graph",
    
    # Agents
    "SupervisorAgent",
    "IrisAgent", 
    "TaraAgent",
    
    # State
    "AgentState",
    "AgentName",
    "AGENT_SUPERVISOR",
    "AGENT_IRIS",
    "AGENT_TARA",
    "AGENT_END",
    "create_initial_state",
    "extract_response",
]


# =============================================================================
# Convenience Functions
# =============================================================================

def check_dependencies() -> dict:
    """Check availability of required dependencies.
    
    Returns:
        Dict mapping dependency names to availability status.
    """
    deps = {}
    
    try:
        import langchain_openai
        deps["langchain-openai"] = True
    except ImportError:
        deps["langchain-openai"] = False
    
    try:
        import langgraph
        deps["langgraph"] = True
    except ImportError:
        deps["langgraph"] = False
    
    try:
        import dotenv
        deps["python-dotenv"] = True
    except ImportError:
        deps["python-dotenv"] = False
    
    # Check for API key
    import os
    deps["OPENAI_API_KEY"] = bool(os.environ.get("OPENAI_API_KEY"))
    
    return deps


def print_status() -> None:
    """Print NIA system status."""
    deps = check_dependencies()
    
    print("\n" + "=" * 50)
    print("  N.I.A. System Status")
    print("=" * 50)
    
    for name, available in deps.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"  {name}: {status}")
    
    print()
    
    if all(deps.values()):
        print("  ✅ All dependencies installed. NIA is ready!")
    else:
        missing = [k for k, v in deps.items() if not v]
        print(f"  ⚠️  Missing: {', '.join(missing)}")
        if "OPENAI_API_KEY" in missing:
            print("     Set OPENAI_API_KEY in .env file")
        pkg_missing = [k for k in missing if k != "OPENAI_API_KEY"]
        if pkg_missing:
            print(f"     Install packages: pip install {' '.join(pkg_missing)}")
    
    print()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Run a quick demo of the NIA system."""
    print_status()
    
    print("NIA Demo - Type 'exit' to quit\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            
            response = process_input(user_input)
            print(f"\nNIA: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    demo()
