"""
TARA 2.0 Graph Package.

Provides the LangGraph SubGraph for TARA's reasoning loop.

Architecture:
    tara/graph/
    ├── __init__.py    ← You are here (Package exports)
    ├── state.py       ← TaraState TypedDict
    ├── prompts.py     ← System prompts & context builder
    ├── nodes.py       ← Reasoner & tool executor nodes
    └── workflow.py    ← StateGraph compilation

Usage:
    # Quick run
    from tara.graph import run_tara
    response = run_tara("Open Notepad and type hello")
    
    # With compiled app
    from tara.graph import tara_app, create_initial_tara_state
    state = create_initial_tara_state("My goal")
    result = tara_app.invoke(state)
    
    # Build custom graph
    from tara.graph import build_tara_graph
    app = build_tara_graph(with_memory=True)
"""
from __future__ import annotations

# State
from .state import (
    TaraState,
    TaraNextStep,
    create_initial_tara_state,
)

# Prompts
from .prompts import (
    TARA_SYSTEM_PROMPT,
    TOOL_RESULT_PROMPT,
    build_tara_context,
    build_full_system_prompt,
)

# Nodes
from .nodes import (
    reasoner,
    tool_executor,
    response_formatter,
    should_continue,
)

# Workflow
from .workflow import (
    build_tara_graph,
    get_tara_graph,
    run_tara,
    tara_app,
)


__all__ = [
    # State
    "TaraState",
    "TaraNextStep",
    "create_initial_tara_state",
    # Prompts
    "TARA_SYSTEM_PROMPT",
    "TOOL_RESULT_PROMPT",
    "build_tara_context",
    "build_full_system_prompt",
    # Nodes
    "reasoner",
    "tool_executor",
    "response_formatter",
    "should_continue",
    # Workflow (Main Exports)
    "build_tara_graph",
    "get_tara_graph",
    "run_tara",
    "tara_app",
]
