
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.nia.state import AgentState
# Import strictly what we need to avoid side effects
from src.agents.nia.agent import SupervisorAgent
from src.agents.nia.graph.nodes import docker_node

@pytest.mark.asyncio
async def test_supervisor_routing_docker():
    """Verify Supervisor routes to DOCKER when Gatekeeper detects it."""
    
    # Mock ModelManager to return our MagicMock LLM
    with patch("src.models.manager.get_smart_model") as mock_get_model, \
         patch("src.persona.profile.get_system_prompt", return_value="System Prompt"), \
         patch("src.core.skills.loader.get_skills_prompt", return_value="Skills: coding-agent"):
        
        mock_llm = MagicMock()
        # Async invoke setup
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="ROUTE:DOCKER:coding-agent list files"))
        mock_get_model.return_value = mock_llm
        
        agent = SupervisorAgent()
        
        state = {"messages": [HumanMessage(content="Use coding agent to list files")]}
        
        # Execute
        result = await agent.aprocess(state)
        
        # Verify
        assert result["next"] == "docker"
        assert result["metadata"]["target_skill"] == "coding-agent"
        assert result["metadata"]["skill_query"] == "list files"


@pytest.mark.asyncio
async def test_docker_node_manifest_creation():
    """Verify docker_node creates manifest with host_workdir."""
    
    # Mock dependencies inside docker_node
    # We strip the module path to match where it is imported IN docker_node (which is inside function)
    # But patching src... should work if sys.modules cache is used.
    # Actually, since imports are inside function, we must patch the module where they are defined
    # OR patch sys.modules/imports.
    # Easiest is to patch the strict imports in nodes.py IF they were top level.
    # Since they are specific imports inside function, we can patch `src.agents.nia.graph.nodes.DockerBridge` 
    # IF it was imported. But it's valid to patch the original location if using `patch`.
    
    with patch("src.infrastructure.container_engine.bridge.DockerBridge") as MockBridge, \
         patch("src.infrastructure.container_engine.manager.DockerEngine") as MockEngine, \
         patch("src.core.skills.loader.load_docker_skills") as mock_load_skills, \
         patch("src.core.skills.loader.get_skill_source_code") as mock_get_code:
         
        # Setup Mocks
        mock_load_skills.return_value = [{"name": "coding-agent", "runtime": "python", "pty": False}]
        mock_get_code.return_value = "print('hello')"
        
        mock_instance = MockBridge.return_value
        # Mock successful result
        mock_result = MagicMock()
        mock_result.output = "Mission Accomplished"
        mock_result.error = None
        mock_instance.execute_mission.return_value = mock_result
        
        # Input State with Explicit Host Workdir
        state = {
            "messages": [],
            "metadata": {
                "target_skill": "coding-agent",
                "skill_query": "do something",
                "workdir": "/tmp/host_test" 
            }
        }
        
        # Execute Node
        result = await docker_node(state)
        
        # Assertions
        assert "Mission Accomplished" in result["final_response"]
        
        # Verify Manifest
        args, _ = mock_instance.execute_mission.call_args
        manifest = args[0]
        
        assert manifest.soldier_type == "coding"
        assert manifest.host_workdir == "/tmp/host_test"
        assert manifest.objective == "do something"
