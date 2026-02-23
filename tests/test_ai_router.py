
import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import AIMessage
from src.agents.nia.decision.router import DecisionCore, RoutingDecision


@pytest.mark.asyncio
async def test_route_chat():
    """Verify chat queries are routed to 'chat'."""
    core = DecisionCore()
    
    # Mock LLM to return JSON string
    mock_llm = MagicMock()
    mock_response = AIMessage(content=json.dumps({
        "target": "chat",
        "skill": None,
        "reasoning": "General greeting"
    }))
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    core._llm = mock_llm
    
    result = await core.aroute("Hello, how are you?")
    
    assert result.target == "chat"
    assert result.skill is None
    assert "greeting" in result.reasoning.lower()


@pytest.mark.asyncio
async def test_route_swarm():
    """Verify coding queries are routed to 'swarm'."""
    core = DecisionCore()
    
    mock_llm = MagicMock()
    mock_response = AIMessage(content=json.dumps({
        "target": "swarm",
        "skill": "coding-agent",
        "reasoning": "User wants to create a game"
    }))
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    core._llm = mock_llm
    
    result = await core.aroute("Create a snake game in Python")
    
    assert result.target == "swarm"
    assert result.skill == "coding-agent"


@pytest.mark.asyncio
async def test_route_swarm_markdown_fenced():
    """Verify router handles markdown-fenced JSON responses."""
    core = DecisionCore()
    
    mock_llm = MagicMock()
    # Simulate LLM wrapping JSON in markdown code fence
    fenced_json = '```json\n{"target": "swarm", "skill": "coding-agent", "reasoning": "code task"}\n```'
    mock_response = AIMessage(content=fenced_json)
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    core._llm = mock_llm
    
    result = await core.aroute("Run a python script")
    
    assert result.target == "swarm"
    assert result.skill == "coding-agent"


@pytest.mark.asyncio
async def test_fallback_on_invalid_json():
    """Verify keyword fallback when LLM returns invalid JSON."""
    core = DecisionCore()
    
    mock_llm = MagicMock()
    mock_response = AIMessage(content="I think you should use the coding agent for this.")
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    core._llm = mock_llm
    
    result = await core.aroute("Run a python script to print hello")
    
    # Keyword fallback should catch "python" or "script" or "run"
    assert result.target == "swarm"
    assert result.skill == "coding-agent"
    assert "keyword fallback" in result.reasoning.lower()


@pytest.mark.asyncio
async def test_fallback_on_llm_crash():
    """Verify fallback to chat when LLM completely fails."""
    core = DecisionCore()
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("Connection refused"))
    core._llm = mock_llm
    
    result = await core.aroute("Hello")
    
    assert result.target == "chat"
    assert "error" in result.reasoning.lower()


@pytest.mark.asyncio
async def test_empty_input():
    """Verify empty input returns chat."""
    core = DecisionCore()
    
    result = await core.aroute("")
    
    assert result.target == "chat"
    assert "empty" in result.reasoning.lower()
