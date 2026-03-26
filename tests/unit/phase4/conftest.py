"""Shared fixtures for Phase 4 unit tests."""
from __future__ import annotations

import uuid
from typing import List

import pytest

from src.core.policy.scopes import CapabilityScope
from src.core.schema.mission import MissionManifest, PlanStep, SubagentResult
from src.core.schema.coordinator import (
    BudgetExtensionRequest,
    ContextObservation,
    SwarmLimits,
)


# ---------------------------------------------------------------------------
# MissionManifest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def read_only_manifest() -> MissionManifest:
    """A manifest that only requires READ_ONLY scope."""
    return MissionManifest(
        mission_id="test-ro-001",
        intent="List files in the project directory",
        steps=[
            PlanStep(
                description="Read directory listing",
                required_scopes=[CapabilityScope.READ_ONLY],
                assigned_role="researcher",
            ),
        ],
        required_scopes=[CapabilityScope.READ_ONLY],
        estimated_depth=1,
        estimated_agents=1,
        execution_mode="fast",
        approved=False,
        approved_scopes=[],
    )


@pytest.fixture
def write_manifest() -> MissionManifest:
    """A manifest requiring WRITE and READ_ONLY scopes."""
    return MissionManifest(
        mission_id="test-wr-002",
        intent="Create a configuration file",
        steps=[
            PlanStep(
                description="Read existing config",
                required_scopes=[CapabilityScope.READ_ONLY],
                assigned_role="researcher",
            ),
            PlanStep(
                description="Write new config file",
                required_scopes=[CapabilityScope.WRITE],
                assigned_role="coder",
            ),
        ],
        required_scopes=[CapabilityScope.READ_ONLY, CapabilityScope.WRITE],
        estimated_depth=2,
        estimated_agents=2,
        execution_mode="standard",
        approved=False,
        approved_scopes=[],
    )


@pytest.fixture
def approved_manifest() -> MissionManifest:
    """A fully approved manifest with WRITE and EXECUTE scopes."""
    return MissionManifest(
        mission_id="test-ap-003",
        intent="Build and test the project",
        steps=[
            PlanStep(
                description="Read source code",
                required_scopes=[CapabilityScope.READ_ONLY],
                assigned_role="researcher",
            ),
            PlanStep(
                description="Write build script",
                required_scopes=[CapabilityScope.WRITE],
                assigned_role="coder",
            ),
            PlanStep(
                description="Execute build",
                required_scopes=[CapabilityScope.EXECUTE],
                assigned_role="coder",
            ),
        ],
        required_scopes=[
            CapabilityScope.READ_ONLY,
            CapabilityScope.WRITE,
            CapabilityScope.EXECUTE,
        ],
        estimated_depth=2,
        estimated_agents=3,
        execution_mode="deep",
        approved=True,
        approved_scopes=[
            CapabilityScope.READ_ONLY,
            CapabilityScope.WRITE,
            CapabilityScope.EXECUTE,
        ],
    )


@pytest.fixture
def multi_scope_manifest() -> MissionManifest:
    """A manifest with all scope types for comprehensive testing."""
    return MissionManifest(
        mission_id="test-ms-004",
        intent="Full system deployment",
        steps=[
            PlanStep(
                description="Read deployment config",
                required_scopes=[CapabilityScope.READ_ONLY],
                assigned_role="researcher",
            ),
            PlanStep(
                description="Write deployment manifest",
                required_scopes=[CapabilityScope.WRITE],
                assigned_role="coder",
            ),
            PlanStep(
                description="Execute deployment",
                required_scopes=[CapabilityScope.EXECUTE],
                assigned_role="coder",
            ),
            PlanStep(
                description="Fetch remote dependencies",
                required_scopes=[CapabilityScope.NETWORK],
                assigned_role="coder",
            ),
            PlanStep(
                description="Spawn monitoring agent",
                required_scopes=[CapabilityScope.AGENT_SPAWN],
                assigned_role="coder",
            ),
        ],
        required_scopes=[
            CapabilityScope.READ_ONLY,
            CapabilityScope.WRITE,
            CapabilityScope.EXECUTE,
            CapabilityScope.NETWORK,
            CapabilityScope.AGENT_SPAWN,
        ],
        estimated_depth=3,
        estimated_agents=5,
        execution_mode="deep",
        approved=True,
        approved_scopes=[
            CapabilityScope.READ_ONLY,
            CapabilityScope.WRITE,
            CapabilityScope.EXECUTE,
            CapabilityScope.NETWORK,
            CapabilityScope.AGENT_SPAWN,
        ],
    )


# ---------------------------------------------------------------------------
# SubagentResult fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def success_result() -> SubagentResult:
    """A successful subagent result."""
    return SubagentResult(
        agent_id="tara-abc123",
        status="success",
        output="Task completed successfully. Created output.txt.",
        artifacts_created=["output.txt"],
        scopes_used=[CapabilityScope.READ_ONLY, CapabilityScope.WRITE],
        tokens_used=1500,
    )


@pytest.fixture
def failed_result() -> SubagentResult:
    """A failed subagent result."""
    return SubagentResult(
        agent_id="tara-def456",
        status="failed",
        output="Could not find the target file.",
        artifacts_created=[],
        scopes_used=[CapabilityScope.READ_ONLY],
        tokens_used=800,
        failure_trace="FileNotFoundError: /tmp/target.txt",
    )


# ---------------------------------------------------------------------------
# BudgetExtensionRequest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def budget_request_with_artifacts() -> BudgetExtensionRequest:
    """A budget request from a subagent that has produced artifacts."""
    return BudgetExtensionRequest(
        agent_id="tara-budg-001",
        current_step=2,
        steps_requested=2,
        justification="Need additional steps to add error handling to generated code.",
        artifacts_produced_so_far=["main.py", "utils.py"],
        tools_called_so_far=["write_file", "read_file"],
    )


@pytest.fixture
def budget_request_no_artifacts_late() -> BudgetExtensionRequest:
    """A budget request with no artifacts at a late stage."""
    return BudgetExtensionRequest(
        agent_id="tara-budg-002",
        current_step=4,
        steps_requested=2,
        justification="Still researching the right approach.",
        artifacts_produced_so_far=[],
        tools_called_so_far=["web_search"],
    )


@pytest.fixture
def budget_request_too_many_steps() -> BudgetExtensionRequest:
    """A budget request asking for too many additional steps."""
    return BudgetExtensionRequest(
        agent_id="tara-budg-003",
        current_step=1,
        steps_requested=10,
        justification="This task is much larger than expected.",
        artifacts_produced_so_far=["draft.py"],
        tools_called_so_far=["write_file"],
    )


@pytest.fixture
def budget_request_early_no_artifacts() -> BudgetExtensionRequest:
    """A budget request at an early stage with no artifacts (benefit of doubt)."""
    return BudgetExtensionRequest(
        agent_id="tara-budg-004",
        current_step=1,
        steps_requested=2,
        justification="Need more time to complete initial research.",
        artifacts_produced_so_far=[],
        tools_called_so_far=["web_search"],
    )
