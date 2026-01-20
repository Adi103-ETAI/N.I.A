"""T.A.R.A. 2.0 - Technical Agent for Reasoning & Analysis.

A LangGraph-based specialist agent for NIA that handles:
- Desktop automation (apps, windows, clipboard)  
- UI interaction (buttons, menus, typing)
- Browser control (navigation, forms)
- File operations (read, write, search)

Architecture:
    TARA 2.0 is now a compiled LangGraph SubGraph accessed via:
    - `from tara.graph import run_tara, tara_app`
    - `from tara.tools import get_tara_tools`

Usage:
    # Via NIA (recommended):
    NIA routes to TARA automatically via call_tara_2 node
    
    # Direct usage:
    from tara.graph import run_tara
    result = run_tara("Open Notepad")
"""
from __future__ import annotations

# TARA 2.0: Export graph components
try:
    from tara.graph import run_tara, tara_app, TaraState
    _HAS_GRAPH = True
except ImportError:
    run_tara = None  # type: ignore
    tara_app = None  # type: ignore
    TaraState = None  # type: ignore
    _HAS_GRAPH = False

# TARA 2.0: Export tool interface
try:
    from tara.tools import get_tara_tools
    _HAS_TOOLS = True
except ImportError:
    get_tara_tools = None  # type: ignore
    _HAS_TOOLS = False


__version__ = "2.5.0"
__all__ = ["run_tara", "tara_app", "TaraState", "get_tara_tools"]
