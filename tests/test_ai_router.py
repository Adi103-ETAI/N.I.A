
import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import AIMessage, HumanMessage
from src.agents.nia.planner import MissionPlanner
from src.core.schema.states import AgentState


@pytest.mark.asyncio
async def test_route_chat():
    """Verify chat queries create simple conversation missions."""
    planner = MissionPlanner()
    
    # Mock LLM to return simple chat manifest
    mock_llm = MagicMock()
    mock_response = AIMessage(content=json.dumps({
        "mission_type": "conversation",
        "scope": "single_turn",
        "steps": [{
            "step_id": "chat_1",
            "role": "supervisor",
            "instruction": "Respond to greeting",
            "dependencies": []
        }]
    }))
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    planner._llm = mock_llm
    
    state: AgentState = {
        "messages": [HumanMessage(content="Hello, how are you?")],
        "next": "",
        "user_input": "Hello, how are you?",
        "final_response": None,
        "route_reason": None,
        "metadata": {},
        "session_id": "test_session",
        "sandbox_result": None,
        "subagent_results": []
    }
    
    manifest = await planner.plan(state)
    
    assert manifest is not None
    assert manifest["mission_type"] == "conversation"
    assert len(manifest["steps"]) > 0


@pytest.mark.asyncio
async def test_route_coding_task():
    """Verify coding queries create agent_spawn missions with multiple steps."""
    planner = MissionPlanner()
    
    mock_llm = MagicMock()
    mock_response = AIMessage(content=json.dumps({
        "mission_type": "agent_spawn",
        "scope": "agent_spawn",
        "execution_mode": "deep",
        "steps": [
            {
                "step_id": "code_1",
                "role": "coder",
                "instruction": "Create snake game logic",
                "dependencies": []
            },
            {
                "step_id": "code_2",
                "role": "coder",
                "instruction": "Add game loop",
                "dependencies": ["code_1"]
            }
        ]
    }))
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    planner._llm = mock_llm
    
    state: AgentState = {
        "messages": [HumanMessage(content="Create a snake game in Python")],
        "next": "",
        "user_input": "Create a snake game in Python",
        "final_response": None,
        "route_reason": None,
        "metadata": {},
        "session_id": "test_session",
        "sandbox_result": None,
        "subagent_results": []
    }
    
    manifest = await planner.plan(state)
    
    assert manifest is not None
    assert manifest["mission_type"] == "agent_spawn"
    assert len(manifest["steps"]) >= 2  # Multi-step task


@pytest.mark.asyncio
async def test_planner_handles_markdown_fenced():
    """Verify planner handles markdown-fenced JSON responses."""
    planner = MissionPlanner()
    
    mock_llm = MagicMock()
    # Simulate LLM wrapping JSON in markdown code fence
    fenced_json = '''```json
{
    "mission_type": "tool_execution",
    "scope": "single_turn",
    "steps": [{
        "step_id": "exec_1",
        "role": "tara",
        "instruction": "Run python script",
        "dependencies": []
    }]
}
```'''
    mock_response = AIMessage(content=fenced_json)
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    planner._llm = mock_llm
    
    state: AgentState = {
        "messages": [HumanMessage(content="Run a python script")],
        "next": "",
        "user_input": "Run a python script",
        "final_response": None,
        "route_reason": None,
        "metadata": {},
        "session_id": "test_session",
        "sandbox_result": None,
        "subagent_results": []
    }
    
    manifest = await planner.plan(state)
    
    assert manifest is not None
    assert manifest["mission_type"] == "tool_execution"


@pytest.mark.asyncio
async def test_fallback_on_invalid_json():
    """Verify planner handles invalid JSON gracefully."""
    planner = MissionPlanner()
    
    mock_llm = MagicMock()
    mock_response = AIMessage(content="I think you should use the coding agent for this.")
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    planner._llm = mock_llm
    
    state: AgentState = {
        "messages": [HumanMessage(content="Run a python script to print hello")],
        "next": "",
        "user_input": "Run a python script to print hello",
        "final_response": None,
        "route_reason": None,
        "metadata": {},
        "session_id": "test_session",
        "sandbox_result": None,
        "subagent_results": []
    }
    
    # Should handle gracefully - either return None or fallback manifest
    try:
        manifest = await planner.plan(state)
        # If it returns something, check it's valid
        if manifest:
            assert "mission_type" in manifest
            assert "steps" in manifest
    except Exception as e:
        # Graceful failure is acceptable
        assert "json" in str(e).lower() or "parse" in str(e).lower()


@pytest.mark.asyncio
async def test_fallback_on_llm_crash():
    """Verify planner handles LLM failures gracefully."""
    planner = MissionPlanner()
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("Connection refused"))
    planner._llm = mock_llm
    
    state: AgentState = {
        "messages": [HumanMessage(content="Hello")],
        "next": "",
        "user_input": "Hello",
        "final_response": None,
        "route_reason": None,
        "metadata": {},
        "session_id": "test_session",
        "sandbox_result": None,
        "subagent_results": []
    }
    
    # Should handle gracefully
    try:
        manifest = await planner.plan(state)
        # May return None or fallback
        if manifest:
            assert "mission_type" in manifest
    except Exception as e:
        # Expected behavior - propagate error or handle gracefully
        assert "connection" in str(e).lower() or "refused" in str(e).lower()


@pytest.mark.asyncio
async def test_empty_input():
    """Verify planner handles empty input gracefully."""
    planner = MissionPlanner()
    
    state: AgentState = {
        "messages": [HumanMessage(content="")],
        "next": "",
        "user_input": "",
        "final_response": None,
        "route_reason": None,
        "metadata": {},
        "session_id": "test_session",
        "sandbox_result": None,
        "subagent_results": []
    }
    
    # Should handle gracefully - return None or simple fallback
    manifest = await planner.plan(state)
    
    # Either None or a valid fallback manifest
    if manifest:
        assert "mission_type" in manifest
        assert "steps" in manifest
