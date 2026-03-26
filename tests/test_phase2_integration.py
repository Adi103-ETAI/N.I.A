
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.nia.state import AgentState
# Import strictly what we need to avoid side effects
from src.agents.nia.graph.nodes import docker_node, planner_node

@pytest.mark.asyncio
async def test_planner_routing_to_docker():
    """Verify Planner Node correctly routes to DOCKER for code execution."""
    
    # Mock MissionPlanner - it's imported inside the planner_node function
    with patch("src.agents.nia.planner.MissionPlanner") as MockPlannerClass:
        mock_planner_instance = MockPlannerClass.return_value
        
        # Setup mock mission manifest for docker routing
        mock_manifest = {
            "mission_type": "agent_spawn",
            "scope": "agent_spawn",
            "execution_mode": "quick",
            "steps": [{
                "step_id": "code_1",
                "role": "coder",
                "instruction": "List files",
                "dependencies": []
            }]
        }
        mock_planner_instance.plan = AsyncMock(return_value=mock_manifest)
        
        # Mock preflight approval - also imported inside function
        with patch("src.agents.nia.graph.nodes.planner.run_preflight_approval") as mock_approval:
            mock_approval.return_value = (True, None)
            
            state: AgentState = {
                "user_input": "Use coding agent to list files",
                "messages": [HumanMessage(content="Use coding agent to list files")],
                "metadata": {},
                "next": "",
                "final_response": None,
                "route_reason": None,
                "session_id": "test",
                "sandbox_result": None,
                "subagent_results": []
            }
            
            # Execute Planner Node
            result = await planner_node(state)
            
            # Verify routing decision
            # With single-step quick mode, should route to docker/tara directly
            assert result["next"] in ["docker", "tara", "coordinator"]
            assert "mission_manifest" in result["metadata"]



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
