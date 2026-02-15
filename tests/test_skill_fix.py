"""Verification Tests for OpenClaw Skill Loader Upgrade.

Test 1: test_dual_path_discovery
    - Verifies that a skill in src/core/skills/library/ overrides one in data/skills/
Test 2: test_coding_agent_metadata
    - Verifies that the 'coding-agent' skill is correctly loaded with PTY=True,
    - and that 'metadata.nia.requires' (codex) is parsed correctly.
"""
import pytest
import shutil
from pathlib import Path
from src.core.skills.loader import load_docker_skills, _DATA_SKILLS_DIR, _LIBRARY_SKILLS_DIR


@pytest.fixture
def clean_skill_env():
    """Ensure clean state for tests."""
    # Backup existing
    yield
    # Cleanup dummy skills if any


def test_dual_path_discovery(tmp_path):
    """
    Verify that a skill in library/ overrides a skill in data/ with the same name.
    
    We can't easily modify the real _LIBRARY_SKILLS_DIR during tests without
    messing up the dev env. So we will rely on the fact that 'coding-agent'
    is NOW in the library. We will create a dummy 'coding-agent' in data/skills/
    and verify that load_docker_skills() returns the library version (source='library').
    """
    # 1. Create dummy 'coding-agent' in data/skills/ (Low Priority)
    dummy_skill_dir = _DATA_SKILLS_DIR / "coding_agent"
    dummy_skill_dir.mkdir(parents=True, exist_ok=True)
    
    (dummy_skill_dir / "skill.md").write_text("""---
name: coding-agent
description: I am a dummy skill from data/
runtime: node
---
""", encoding="utf-8")
    (dummy_skill_dir / "source.js").write_text("console.log('dummy')", encoding="utf-8")
    
    try:
        # 2. Load skills
        skills = load_docker_skills()
        
        # 3. Find coding-agent
        target = next((s for s in skills if s["name"] == "coding-agent"), None)
        assert target is not None
        
        # 4. Assert it is the Library version (Python), NOT the Dummy (Node)
        assert target["runtime"] == "python", "Should load the LIBRARY version (python), not data version (node)"
        assert target["source"] == "library", "Source should be 'library'"
        assert "Spawns an AI coding agent" in target["description"]
        
    finally:
        # Cleanup
        if dummy_skill_dir.exists():
            shutil.rmtree(dummy_skill_dir)


def test_coding_agent_metadata():
    """Verify that the real coding-agent in library/ has correct NIA metadata."""
    skills = load_docker_skills()
    target = next((s for s in skills if s["name"] == "coding-agent"), None)
    
    assert target is not None
    assert target["pty"] is True, "Coding Agent must have pty=True"
    assert "codex" in target["requires"], "Metadata parsing failed for 'requires' (via metadata.nia)"
    assert target["workdir"] == "/workspace/project"
    assert target["emoji"] == "🤖"
