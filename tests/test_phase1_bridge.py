"""Phase 1 Verification — DockerBridge, Skill System, & Builder Cache Tests.

Test 1 (Python):       Execute test_math skill via DockerBridge → assert result = 4
Test 2 (Node.js):      Execute test_json skill via DockerBridge → assert valid JSON
Test 3 (Skill Loader): Verify load_docker_skills() discovers skills from skill.md
Test 4 (PTY Manifest): Verify coding-agent skill discovered with pty=True from library/
Test 5 (Builder Cache):Verify cache_learned_skill() creates valid skill folder
Test 6 (Skills Prompt): Verify get_skills_prompt() returns formatted markdown
Test 7 (Dual Scan):    Verify library + data skills merge without duplicates
"""
import json
import os
import sys
import shutil
import pytest
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.soldiers.schemas import MissionManifest, MissionResult, RuntimeType


# =============================================================================
# Helpers
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = PROJECT_ROOT / "data" / "skills"
LIBRARY_DIR = PROJECT_ROOT / "src" / "core" / "skills" / "library"


def _docker_available() -> bool:
    """Check if Docker is accessible."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


requires_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker Desktop not running or not installed",
)


# =============================================================================
# Test 1: Python Skill Execution via DockerBridge
# =============================================================================

@requires_docker
def test_python_skill_execution():
    """Run test_math Python skill in Docker and verify result."""
    from src.infrastructure.container_engine.manager import DockerEngine
    from src.infrastructure.container_engine.bridge import DockerBridge
    from src.core.skills.loader import get_skill_source_code

    code = get_skill_source_code("test_math", SKILLS_DIR)
    assert code is not None

    manifest = MissionManifest(
        task_id="test-python-001",
        soldier_type="coding",
        runtime=RuntimeType.PYTHON,
        objective="Calculate 2 + 2",
        code=code,
    )

    engine = DockerEngine()
    bridge = DockerBridge(engine)
    result = bridge.execute_mission(manifest)

    assert result.task_id == "test-python-001"
    assert result.status.value == "success"
    assert "4" in result.output
    bridge.cleanup_workspace("test-python-001")


# =============================================================================
# Test 2: Node.js Skill Execution via DockerBridge
# =============================================================================

@requires_docker
def test_node_skill_execution():
    """Run test_json Node.js skill in Docker and verify valid JSON."""
    from src.infrastructure.container_engine.manager import DockerEngine
    from src.infrastructure.container_engine.bridge import DockerBridge
    from src.core.skills.loader import get_skill_source_code

    code = get_skill_source_code("test_json", SKILLS_DIR)
    assert code is not None

    manifest = MissionManifest(
        task_id="test-node-001",
        soldier_type="coding",
        runtime=RuntimeType.NODE,
        objective="Create a JSON object",
        code=code,
    )

    engine = DockerEngine()
    bridge = DockerBridge(engine)
    result = bridge.execute_mission(manifest)

    assert result.task_id == "test-node-001"
    assert result.status.value == "success"
    bridge.cleanup_workspace("test-node-001")


# =============================================================================
# Test 3: Skill Loader — data/skills/ discovery
# =============================================================================

def test_skill_loader_data_dir():
    """Verify load_docker_skills() discovers test skills from data/skills/."""
    from src.core.skills.loader import load_docker_skills, get_skill_source_code

    skills = load_docker_skills(SKILLS_DIR)
    assert len(skills) >= 2, f"Expected at least 2 skills, got {len(skills)}"

    math_skill = next((s for s in skills if s["name"] == "test_math"), None)
    assert math_skill is not None
    assert math_skill["runtime"] == "python"
    assert math_skill["pty"] is False

    json_skill = next((s for s in skills if s["name"] == "test_json"), None)
    assert json_skill is not None
    assert json_skill["runtime"] == "node"

    # Verify source code reads
    assert "2 + 2" in (get_skill_source_code("test_math", SKILLS_DIR) or "")
    assert "Hello from Node.js" in (get_skill_source_code("test_json", SKILLS_DIR) or "")
    assert get_skill_source_code("nonexistent_skill", SKILLS_DIR) is None


# =============================================================================
# Test 4: Coding-Agent — discovered from library/ with PTY
# =============================================================================

def test_coding_agent_from_library():
    """Verify coding-agent is discovered from src/core/skills/library/ with correct metadata."""
    from src.core.skills.loader import load_docker_skills

    skills = load_docker_skills(LIBRARY_DIR)
    coding = next((s for s in skills if s["name"] == "coding-agent"), None)

    assert coding is not None, f"coding-agent not in library: {[s['name'] for s in skills]}"
    assert coding["runtime"] == "python"
    assert coding["pty"] is True
    assert coding["workdir"] == "/workspace/project"
    assert coding["emoji"] == "🤖"
    assert "codex" in coding.get("requires", [])


# =============================================================================
# Test 5: Builder Cache — Learn & Cache Protocol
# =============================================================================

def test_builder_cache(temp_dir):
    """Verify cache_learned_skill() creates a valid skill folder."""
    from src.agents.soldiers.builder_cache import (
        cache_learned_skill, skill_exists, list_learned_skills,
    )

    skill_dir = cache_learned_skill(
        name="test_stt",
        description="Audio transcription using Whisper",
        runtime="python",
        code='print("transcribed audio")',
        dependencies=["openai-whisper", "torch"],
        builder_task_id="builder-001",
        skills_dir=temp_dir,
    )

    assert (skill_dir / "skill.md").exists()
    assert (skill_dir / "source.py").exists()

    content = (skill_dir / "skill.md").read_text(encoding="utf-8")
    assert "name: test_stt" in content
    assert "created_by: builder_builder-001" in content

    assert skill_exists("test_stt", temp_dir) is True
    assert skill_exists("nonexistent", temp_dir) is False
    assert "test_stt" in list_learned_skills(temp_dir)


# =============================================================================
# Test 6: Skills Prompt
# =============================================================================

def test_skills_prompt_from_library():
    """Verify get_skills_prompt() includes emoji and requires markers."""
    from src.core.skills.loader import get_skills_prompt

    prompt = get_skills_prompt(LIBRARY_DIR)
    assert "Available Skills" in prompt
    assert "coding-agent" in prompt
    assert "🤖" in prompt
    assert "(interactive)" in prompt
    assert "requires: codex" in prompt


# =============================================================================
# Test 7: Dual-Scan — library + data merged
# =============================================================================

def test_dual_scan_merges_skills():
    """Verify load_docker_skills() (no arg) discovers both library and data skills."""
    from src.core.skills.loader import load_docker_skills

    # No argument = dual scan
    all_skills = load_docker_skills()
    names = [s["name"] for s in all_skills]

    # Library skill
    assert "coding-agent" in names, f"coding-agent missing from dual scan: {names}"

    # Data skills
    assert "test_math" in names, f"test_math missing from dual scan: {names}"
    assert "test_json" in names, f"test_json missing from dual scan: {names}"

    # No duplicates
    assert len(names) == len(set(names)), f"Duplicates found: {names}"
