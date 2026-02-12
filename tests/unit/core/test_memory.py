"""Unit tests for src/core/memory.py - N.I.A. 4-Layer Hybrid Memory System.

Tests MemoryManager including:
- Initialization
- Preferences layer (SQLite)
- Security logging layer
- Statistics
"""
import pytest
import tempfile
import os
from pathlib import Path


class TestMemoryManagerInit:
    """Test MemoryManager initialization."""
    
    def test_memory_manager_imports(self):
        """Test that MemoryManager can be imported."""
        from src.core.memory import MemoryManager, get_memory_manager
        assert MemoryManager is not None
        assert get_memory_manager is not None
    
    def test_memory_manager_init(self, tmp_path):
        """Test MemoryManager initializes with custom paths."""
        from src.core.memory import MemoryManager
        
        vectors_dir = tmp_path / "vectors"
        skills_file = tmp_path / "skills.gml"
        db_path = tmp_path / "memory.db"
        
        mgr = MemoryManager(
            vectors_dir=str(vectors_dir),
            skills_file=str(skills_file),
            db_path=str(db_path),
        )
        
        assert mgr is not None
        assert Path(db_path).exists()


class TestPreferencesLayer:
    """Test Preferences (SQLite) layer."""
    
    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create a temporary MemoryManager."""
        from src.core.memory import MemoryManager
        
        return MemoryManager(
            vectors_dir=str(tmp_path / "vectors"),
            skills_file=str(tmp_path / "skills.gml"),
            db_path=str(tmp_path / "memory.db"),
        )
    
    @pytest.mark.asyncio
    async def test_set_and_get_preference(self, memory_manager):
        """Test setting and getting preferences."""
        result = await memory_manager.set_preference("user_name", "Alice")
        assert result is True
        
        value = await memory_manager.get_preference("user_name")
        assert value == "Alice"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_preference(self, memory_manager):
        """Test getting a preference that doesn't exist."""
        value = await memory_manager.get_preference("nonexistent_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_get_all_preferences(self, memory_manager):
        """Test getting all preferences."""
        await memory_manager.set_preference("key1", "value1")
        await memory_manager.set_preference("key2", "value2")
        
        prefs = await memory_manager.get_all_preferences()
        
        assert isinstance(prefs, dict)
        assert prefs.get("key1") == "value1"
        assert prefs.get("key2") == "value2"


class TestSecurityLayer:
    """Test Security logging layer."""
    
    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create a temporary MemoryManager."""
        from src.core.memory import MemoryManager
        
        return MemoryManager(
            vectors_dir=str(tmp_path / "vectors"),
            skills_file=str(tmp_path / "skills.gml"),
            db_path=str(tmp_path / "memory.db"),
        )
    
    def test_log_security_event(self, memory_manager):
        """Test logging a security event."""
        result = memory_manager.log_security_event("rm -rf /", "blocked")
        assert result is True
    
    def test_is_blocked(self, memory_manager):
        """Test checking if a trigger is blocked."""
        memory_manager.log_security_event("dangerous_command", "blocked")
        
        assert memory_manager.is_blocked("dangerous_command") is True
        assert memory_manager.is_blocked("safe_command") is False


class TestSkillStats:
    """Test skill statistics tracking."""
    
    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create a temporary MemoryManager."""
        from src.core.memory import MemoryManager
        
        return MemoryManager(
            vectors_dir=str(tmp_path / "vectors"),
            skills_file=str(tmp_path / "skills.gml"),
            db_path=str(tmp_path / "memory.db"),
        )
    
    @pytest.mark.asyncio
    async def test_record_skill_usage(self, memory_manager):
        """Test recording skill usage."""
        result = await memory_manager.record_skill_usage("launch_app")
        assert result is True
        
        # Record again to increment
        await memory_manager.record_skill_usage("launch_app")
        
        stats = await memory_manager.get_skill_stats()
        assert "launch_app" in stats
        assert stats["launch_app"]["usage_count"] == 2


class TestMemoryStats:
    """Test memory statistics."""
    
    def test_get_stats(self, tmp_path):
        """Test getting memory statistics."""
        from src.core.memory import MemoryManager
        
        mgr = MemoryManager(
            vectors_dir=str(tmp_path / "vectors"),
            skills_file=str(tmp_path / "skills.gml"),
            db_path=str(tmp_path / "memory.db"),
        )
        
        stats = mgr.get_stats()
        
        assert isinstance(stats, dict)
        assert "episodic" in stats
        assert "skills" in stats
        assert "preferences" in stats
        assert "security" in stats
    
    def test_check_db_health(self, tmp_path):
        """Test database health check."""
        from src.core.memory import MemoryManager
        
        mgr = MemoryManager(
            vectors_dir=str(tmp_path / "vectors"),
            skills_file=str(tmp_path / "skills.gml"),
            db_path=str(tmp_path / "memory.db"),
        )
        
        health = mgr.check_db_health()
        
        assert isinstance(health, dict)
        assert health["sql"] is True  # SQLite should always work


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
