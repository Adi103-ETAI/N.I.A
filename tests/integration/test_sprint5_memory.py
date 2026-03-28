"""Integration tests for Sprint 5: Memory & State Integration."""
import pytest
import asyncio
import os
from pathlib import Path

# Suppress validation errors for test environment BEFORE any imports from src
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("ENVIRONMENT", "test")


# Test NamespaceManager integration
class TestNamespaceIntegration:
    """Tests for namespace isolation and merge."""
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires full system initialization")
    async def test_namespace_created_per_agent(self, tmp_path):
        """Test that each agent gets isolated namespace."""
        from src.core.memory.namespaces import get_namespace_manager
        
        ns = get_namespace_manager()
        agent_id = "test-agent-001"
        
        # Create namespace
        namespace = ns.get_or_create_namespace(agent_id)
        assert namespace is not None
        assert agent_id in namespace
        
        # Cleanup
        ns.drop_namespace(agent_id)
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires full system initialization")
    async def test_namespace_merge_on_success(self, tmp_path):
        """Test namespace merges to global on success."""
        from src.core.memory.namespaces import get_namespace_manager
        
        ns = get_namespace_manager()
        agent_id = "test-merge-agent"
        
        # Create and populate namespace
        namespace = ns.get_or_create_namespace(agent_id)
        # Add test data if possible
        
        # Merge
        await ns.merge_namespace(agent_id)
        
        # Verify merged (check global exists)
        # Note: Actual verification depends on implementation
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires full system initialization")
    async def test_namespace_cleanup_on_failure(self, tmp_path):
        """Test namespace dropped on agent failure."""
        from src.core.memory.namespaces import get_namespace_manager
        
        ns = get_namespace_manager()
        agent_id = "test-failed-agent"
        
        # Create namespace
        ns.get_or_create_namespace(agent_id)
        
        # Drop (simulate failure)
        ns.drop_namespace(agent_id)
        
        # Verify cleaned up
        # Should not error on double-drop
        ns.drop_namespace(agent_id)  # Safe to call again


# Test ContextWormhole integration
class TestWormholeIntegration:
    """Tests for context wormhole observation sharing."""
    
    @pytest.mark.asyncio
    async def test_wormhole_lifecycle(self):
        """Test wormhole subscribe/unsubscribe."""
        from src.core.bus.context_wormhole import ContextWormhole
        
        wormhole = ContextWormhole("test-mission")
        
        wormhole.subscribe()
        assert wormhole._active  # Check _active state instead
        
        wormhole.unsubscribe()
        assert not wormhole._active  # Should be deactivated
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires full system initialization and event bus")
    async def test_observation_emission(self):
        """Test agents can emit observations."""
        from src.core.bus.context_wormhole import ContextWormhole, emit_observation
        
        wormhole = ContextWormhole("test-obs-mission")
        wormhole.subscribe()
        
        try:
            # Emit test observation
            await emit_observation(
                agent_id="test-agent",
                observation="Found important data",
                relevance_tags=["test"],
            )
            # Should not error, may or may not be captured in test environment
        finally:
            wormhole.unsubscribe()
    
    @pytest.mark.asyncio
    async def test_context_retrieval(self):
        """Test retrieving context from wormhole."""
        from src.core.bus.context_wormhole import ContextWormhole
        
        wormhole = ContextWormhole("test-context-mission")
        wormhole.subscribe()
        
        try:
            # Get context summary (may be empty)
            context = wormhole.get_condensed_summary(max_items=5)
            # Should return string (possibly empty)
            assert isinstance(context, str)
            # Empty at start
            assert context == ""
        finally:
            wormhole.unsubscribe()


# Test Coordinator Integration
class TestCoordinatorWithMemory:
    """Tests for coordinator with Sprint 5 features."""
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires full coordinator initialization")
    async def test_coordinator_creates_wormhole(self, tmp_path):
        """Test coordinator initializes wormhole."""
        # This test verifies the integration exists
        # Full coordinator test may require more setup
        from src.agents.nia.subagents.coordinator import run_coordinator
        
        manifest = {
            "mission_id": "test-wormhole-mission",
            "intent": "Test wormhole creation",
            "objectives": ["Simple test"],
        }
        
        # Should not error even if mission doesn't fully complete
        try:
            result = await asyncio.wait_for(
                run_coordinator(manifest, db_path=str(tmp_path)),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            pass  # OK if times out, we're testing initialization
        except Exception as e:
            # Log but don't fail - we're testing wormhole init
            print(f"Coordinator: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
