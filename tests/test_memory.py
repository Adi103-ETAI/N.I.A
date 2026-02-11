"""Unit tests for core/memory.py - Memory Management System.

Tests MemoryManager including:
- Preferences (set/get)
- Vacuum database hygiene
"""
from src.core.memory import MemoryManager
import tempfile
import os
import shutil


def test_vacuum_db_runs_without_error():
    """Test that vacuum completes without raising."""
    # Use a temp directory for all memory data
    temp_dir = tempfile.mkdtemp()
    try:
        mgr = MemoryManager(
            vectors_dir=os.path.join(temp_dir, "vectors"),
            skills_file=os.path.join(temp_dir, "skills.gml"),
            db_path=os.path.join(temp_dir, "memory.db")
        )
        # Vacuum should not raise
        mgr._vacuum_memory_db()
        assert True
    finally:
        import gc
        gc.collect()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def test_preference_crud():
    """Test set/get preference operations."""
    temp_dir = tempfile.mkdtemp()
    try:
        mgr = MemoryManager(
            vectors_dir=os.path.join(temp_dir, "vectors"),
            skills_file=os.path.join(temp_dir, "skills.gml"),
            db_path=os.path.join(temp_dir, "memory.db")
        )
        # Set and get a preference
        mgr.set_preference("theme", "dark", category="ui")
        result = mgr.get_preference("theme")
        assert result == "dark"
        
        # Get all preferences
        all_prefs = mgr.get_all_preferences()
        assert "theme" in all_prefs
    finally:
        import gc
        gc.collect()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def test_stats_returns_dict():
    """Test that get_stats returns a dictionary."""
    temp_dir = tempfile.mkdtemp()
    try:
        mgr = MemoryManager(
            vectors_dir=os.path.join(temp_dir, "vectors"),
            skills_file=os.path.join(temp_dir, "skills.gml"),
            db_path=os.path.join(temp_dir, "memory.db")
        )
        stats = mgr.get_stats()
        assert isinstance(stats, dict)
    finally:
        import gc
        gc.collect()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
