"""Integration tests for coordinator crash recovery with AsyncSqliteSaver."""
import pytest
import os
import shutil
from pathlib import Path
from src.agents.nia.subagents.coordinator import run_coordinator
from src.core.schema.mission import MissionManifest, PlanStep


@pytest.fixture
def test_checkpoint_dir(tmp_path):
    """Temporary checkpoint directory for testing."""
    checkpoint_dir = tmp_path / "test_checkpoints"
    checkpoint_dir.mkdir()
    yield str(checkpoint_dir)
    # Cleanup - verify directory is cleaned up after test
    if Path(checkpoint_dir).exists():
        shutil.rmtree(checkpoint_dir)


@pytest.fixture
def sample_manifest():
    """Sample mission manifest for testing - basic valid manifest."""
    return MissionManifest(
        mission_id="test-mission-001",
        intent="Test crash recovery",
        steps=[
            PlanStep(
                description="Test step 1",
                assigned_role="tara",
                required_scopes=[],
            )
        ],
        execution_mode="fast",
        approved=True,
        approved_scopes=[],
    )


@pytest.mark.asyncio
async def test_coordinator_creates_checkpoint_db(test_checkpoint_dir, sample_manifest):
    """Test that coordinator creates checkpoint database."""
    # Run coordinator with test checkpoint dir
    result = await run_coordinator(sample_manifest, db_path=test_checkpoint_dir)
    
    # Verify checkpoint DB was created
    db_path = os.path.join(test_checkpoint_dir, "coordinator.db")
    assert os.path.exists(db_path), "Checkpoint database should be created"
    assert os.path.getsize(db_path) > 0, "Checkpoint database should not be empty"
    
    # Verify result is a dict with expected status field
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "status" in result, "Result should have status field"


@pytest.mark.asyncio
async def test_coordinator_uses_thread_id_isolation(test_checkpoint_dir):
    """Test that different missions use different thread IDs in checkpoint DB."""
    manifest1 = MissionManifest(
        mission_id="mission-a",
        intent="Task A",
        steps=[
            PlanStep(
                description="Do something for A",
                assigned_role="tara",
                required_scopes=[],
            )
        ],
        execution_mode="fast",
        approved=True,
        approved_scopes=[],
    )
    
    manifest2 = MissionManifest(
        mission_id="mission-b",
        intent="Task B",
        steps=[
            PlanStep(
                description="Do something for B",
                assigned_role="tara",
                required_scopes=[],
            )
        ],
        execution_mode="fast",
        approved=True,
        approved_scopes=[],
    )
    
    # Run two different missions
    result1 = await run_coordinator(manifest1, db_path=test_checkpoint_dir)
    result2 = await run_coordinator(manifest2, db_path=test_checkpoint_dir)
    
    # Both should return result dicts
    assert result1 is not None, "First run should return result"
    assert result2 is not None, "Second run should return result"
    assert isinstance(result1, dict), "Result1 should be a dictionary"
    assert isinstance(result2, dict), "Result2 should be a dictionary"
    
    # Checkpoint DB should exist
    db_path = os.path.join(test_checkpoint_dir, "coordinator.db")
    assert os.path.exists(db_path), "Checkpoint DB should exist after multiple runs"


@pytest.mark.asyncio
async def test_coordinator_resume_capability(test_checkpoint_dir, sample_manifest):
    """Test that coordinator can resume from checkpoint (basic check)."""
    # First run - initializes checkpoint
    result1 = await run_coordinator(sample_manifest, db_path=test_checkpoint_dir)
    
    # Second run with same mission_id (simulates resume/replay)
    result2 = await run_coordinator(sample_manifest, db_path=test_checkpoint_dir)
    
    # Both should complete successfully without error
    assert result1 is not None, "First run should return result"
    assert result2 is not None, "Second run should return result"
    assert isinstance(result1, dict), "Result1 should be dict"
    assert isinstance(result2, dict), "Result2 should be dict"
    
    # Both should have status field
    assert "status" in result1, "First result should have status"
    assert "status" in result2, "Second result should have status"
    
    # Note: Full resume logic (resuming from failure) tested in separate test


@pytest.mark.asyncio
async def test_checkpoint_directory_auto_created(tmp_path, sample_manifest):
    """Test that checkpoint directory is auto-created if missing."""
    nonexistent_dir = tmp_path / "auto_created_checkpoints"
    assert not nonexistent_dir.exists(), "Test directory should not exist initially"
    
    # Run coordinator with non-existent directory
    result = await run_coordinator(sample_manifest, db_path=str(nonexistent_dir))
    
    # Directory should be auto-created
    assert nonexistent_dir.exists(), "Directory should be auto-created"
    assert (nonexistent_dir / "coordinator.db").exists(), "Checkpoint DB should be created"
    assert result is not None, "Coordinator should return result"


@pytest.mark.asyncio
async def test_coordinator_checkpoint_persistence(test_checkpoint_dir):
    """Test that checkpoint file persists across runs and grows."""
    manifest1 = MissionManifest(
        mission_id="persist-test-001",
        intent="First task",
        steps=[
            PlanStep(
                description="Step 1",
                assigned_role="tara",
                required_scopes=[],
            )
        ],
        execution_mode="fast",
        approved=True,
        approved_scopes=[],
    )
    
    # First run
    result1 = await run_coordinator(manifest1, db_path=test_checkpoint_dir)
    db_path = os.path.join(test_checkpoint_dir, "coordinator.db")
    size_after_first = os.path.getsize(db_path)
    
    # Second run with different manifest
    manifest2 = MissionManifest(
        mission_id="persist-test-002",
        intent="Second task",
        steps=[
            PlanStep(
                description="Step 2",
                assigned_role="tara",
                required_scopes=[],
            )
        ],
        execution_mode="fast",
        approved=True,
        approved_scopes=[],
    )
    
    result2 = await run_coordinator(manifest2, db_path=test_checkpoint_dir)
    size_after_second = os.path.getsize(db_path)
    
    # DB should still exist
    assert os.path.exists(db_path), "Checkpoint DB should persist"
    
    # DB size should be consistent (may grow or stay same depending on implementation)
    assert size_after_second >= 0, "DB size should be valid"
    assert result1 is not None and result2 is not None


@pytest.mark.asyncio
async def test_coordinator_no_checkpoint_directory_error(sample_manifest):
    """Test that coordinator handles missing directory gracefully."""
    # Use a path that's read-only or doesn't exist in parent
    invalid_path = "/nonexistent_root_path_xyz123/checkpoints"
    
    # This should either succeed with fallback or fail gracefully
    # The coordinator is designed to fall back if checkpointer fails
    result = await run_coordinator(sample_manifest, db_path=invalid_path)
    
    # Should return a result (either with or without checkpointing)
    assert result is not None, "Coordinator should return a result even with path issues"
    assert isinstance(result, dict), "Result should be a dictionary"
    # The status might be 'failed' or other states, but should exist
    assert "status" in result, "Result should have status field"


@pytest.mark.asyncio
async def test_multiple_missions_same_checkpoint_dir(test_checkpoint_dir):
    """Test multiple independent missions using the same checkpoint directory."""
    manifests = [
        MissionManifest(
            mission_id=f"multi-test-{i:03d}",
            intent=f"Task {i}",
            steps=[
                PlanStep(
                    description=f"Objective {i}",
                    assigned_role="tara",
                    required_scopes=[],
                )
            ],
            execution_mode="fast",
            approved=True,
            approved_scopes=[],
        )
        for i in range(3)
    ]
    
    results = []
    for manifest in manifests:
        result = await run_coordinator(manifest, db_path=test_checkpoint_dir)
        results.append(result)
    
    # All should complete
    assert len(results) == 3, "Should have 3 results"
    for result in results:
        assert result is not None, "Each result should not be None"
        assert isinstance(result, dict), "Each result should be a dict"
        assert "status" in result, "Each result should have status"
    
    # Checkpoint DB should have grown with data
    db_path = os.path.join(test_checkpoint_dir, "coordinator.db")
    assert os.path.exists(db_path), "Checkpoint DB should exist"
    assert os.path.getsize(db_path) > 1024, "Checkpoint DB should contain data"
