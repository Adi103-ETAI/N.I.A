
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.nia.state import AgentState
# Import strictly what we need to avoid side effects
from src.agents.nia.decision.router import RoutingDecision
from src.agents.nia.graph.nodes import docker_node, router_node

@pytest.mark.asyncio
async def test_router_routing_to_docker():
    """Verify Router Node correctly routes to DOCKER."""
    
    # Mock DecisionCore inside router_node
    with patch("src.agents.nia.decision.router.DecisionCore") as MockRouterClass:
        mock_router_instance = MockRouterClass.return_value
        
        # Setup mock decision
        mock_router_instance.aroute = AsyncMock(return_value=RoutingDecision(
            target="swarm",
            skill="coding-agent",
            reasoning="User wants to code"
        ))
        
        state = {"user_input": "Use coding agent to list files", "messages": [], "metadata": {}}
        
        # Execute Router Node
        result = await router_node(state)
        
        # Verify
        assert result["next"] == "docker"
        assert result["metadata"]["target_skill"] == "coding-agent"
        assert result["metadata"]["skill_query"] == "Use coding agent to list files"



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
