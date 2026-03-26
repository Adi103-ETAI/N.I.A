"""Unit tests for TARA (Tool Execution Agent)."""
import pytest
from unittest.mock import MagicMock, patch

def test_tool_discovery():
    """Verify tool auto-discovery works."""
    from src.capabilities.interface import get_tara_tools, get_tools_by_category
    
    # Get all tools
    tools = get_tara_tools()
    assert len(tools) > 0, "No tools discovered"

    # Show by category
    categories = get_tools_by_category()
    assert len(categories) > 0, "No categories found"


def test_context_builder():
    """Verify context builder formats correctly."""
    from src.agents.tara.graph.prompts import build_tara_context
    
    # Mock state
    mock_state = {
        "user_goal": "Test goal",
        "screen_context": "[1] {Button} \"Save File\"\n[2] {Edit} \"filename.txt\"",
        "active_app": "notepad_1",
        "clipboard": "Test clipboard content",
        "last_error": "None",
        "iteration_count": 1,
    }

    context = build_tara_context(mock_state)
    assert isinstance(context, str)
    assert "Test goal" in context
    assert "notepad_1" in context


def test_state_creation():
    """Verify state factory works."""
    from src.core.schema.states import create_initial_tara_state
    
    state = create_initial_tara_state("Test goal")
    assert state.get('user_goal') == "Test goal"
    assert state.get('iteration_count') == 0
    assert isinstance(state.get('messages', []), list)


def test_graph_compilation():
    """Verify graph compiles without error."""
    try:
        from src.agents.tara.graph.workflow import build_tara_graph
        app = build_tara_graph()
        assert app is not None
    except ImportError:
        pytest.skip("langgraph not installed or graph dependencies missing")

@pytest.mark.skip(reason="Requires LLM and user interaction")
def test_full_execution():
    """Skip full execution test in unit tests."""
    pass
