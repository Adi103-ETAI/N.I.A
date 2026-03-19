"""T.A.R.A. — Technical Agent for Reasoning & Analysis.

LangGraph-based specialist agent that handles all tool-use tasks for NIA:
desktop automation, UI interaction, browser control, and system operations.

Package Structure::

    tara/
    ├── __init__.py          # Public exports (this file)
    ├── protocols.py         # HOW_TO_USE guide injected into TARA's system prompt
    ├── security.py          # Warden security gate (Operation Iron Cage)
    ├── graph/
    │   ├── __init__.py      # run_tara() entry point + tara_app compiled graph
    │   ├── state.py         # TaraState TypedDict + TaraStateUpdate
    │   ├── prompts.py       # System prompts for TARA's ReAct loop
    │   ├── workflow.py      # LangGraph graph construction
    │   └── nodes/
    │       ├── utils.py     # Shared helpers (msg extraction, formatting)
    │       ├── reasoner.py  # ReAct reasoning node
    │       ├── executor.py  # Tool execution node
    │       └── formatter.py # Final response formatter node

Usage::

    # Via NIA (recommended — NIA routes automatically):
    # The NIA supervisor calls TARA via the call_tara node.

    # Direct usage:
    from src.agents.tara.graph import run_tara
    result = await run_tara("Open Notepad and type Hello World")

Version: 4.0.0
"""
from __future__ import annotations

# TARA 2.0: Export graph components
try:
    from src.agents.tara.graph import run_tara, get_tara_subgraph, TaraState
    _HAS_GRAPH = True
except ImportError:
    run_tara = None  # type: ignore
    get_tara_subgraph = None  # type: ignore
    TaraState = None  # type: ignore
    _HAS_GRAPH = False

# TARA 2.0: Export tool interface
try:
    from src.agents.tara.tools import get_tara_tools
    _HAS_TOOLS = True
except ImportError:
    get_tara_tools = None  # type: ignore
    _HAS_TOOLS = False


__version__ = "3.1.0"
__all__ = ["run_tara", "get_tara_subgraph", "TaraState", "get_tara_tools"]
