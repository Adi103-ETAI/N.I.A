"""Unit tests for Phase 4 Pydantic schemas.

Covers MissionManifest, SubagentResult, CoordinatorState factory,
SwarmLimits immutability, ContextObservation defaults, and BudgetExtensionRequest.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.policy.scopes import CapabilityScope
from src.core.schema.mission import MissionManifest, PlanStep, SubagentResult
from src.core.schema.coordinator import (
    BudgetExtensionRequest,
    ContextObservation,
    SwarmLimits,
)
from src.core.schema.states import create_coordinator_state


# ---------------------------------------------------------------------------
# MissionManifest
# ---------------------------------------------------------------------------


class TestMissionManifest:
    """Tests for MissionManifest creation and serialisation."""

    def test_mission_manifest_creation(self, approved_manifest: MissionManifest) -> None:
        """Create a manifest with all fields and verify model_dump round-trip."""
        data = approved_manifest.model_dump()

        assert data["mission_id"] == "test-ap-003"
        assert data["intent"] == "Build and test the project"
        assert len(data["steps"]) == 3
        assert data["approved"] is True
        assert data["execution_mode"] == "deep"
        assert data["estimated_depth"] == 2
        assert data["estimated_agents"] == 3
        assert len(data["required_scopes"]) == 3
        assert len(data["approved_scopes"]) == 3


# ---------------------------------------------------------------------------
# SubagentResult
# ---------------------------------------------------------------------------


class TestSubagentResult:
    """Tests for SubagentResult with each status type."""

    @pytest.mark.parametrize(
        "status",
        ["success", "failed", "scope_violation", "stuck", "needs_clarification"],
    )
    def test_subagent_result_statuses(self, status: str) -> None:
        """Each valid status literal must be accepted."""
        result = SubagentResult(
            agent_id=f"test-{status}",
            status=status,
            output=f"Test output for {status}",
        )
        assert result.status == status
        assert result.agent_id == f"test-{status}"

    def test_subagent_result_defaults(self) -> None:
        """Default fields must have sensible zero-values."""
        result = SubagentResult(agent_id="x", status="success")

        assert result.output == ""
        assert result.artifacts_created == []
        assert result.scopes_used == []
        assert result.tokens_used == 0
        assert result.failure_trace is None


# ---------------------------------------------------------------------------
# CoordinatorState factory
# ---------------------------------------------------------------------------


class TestCoordinatorStateFactory:
    """Tests for create_coordinator_state."""

    def test_coordinator_state_factory(self, approved_manifest: MissionManifest) -> None:
        """Factory must produce a valid state dict from a manifest."""
        state = create_coordinator_state(approved_manifest)

        assert state["mission"]["mission_id"] == approved_manifest.mission_id
        assert state["status"] == "planning"
        assert state["tree_depth"] == 0
        assert state["total_nodes_spawned"] == 0
        assert state["human_question"] is None
        assert state["final_output"] is None
        assert isinstance(state["pending_steps"], list)
        assert isinstance(state["completed_results"], list)


# ---------------------------------------------------------------------------
# SwarmLimits
# ---------------------------------------------------------------------------


class TestSwarmLimits:
    """Tests for SwarmLimits frozen model config."""

    def test_swarm_limits_frozen(self) -> None:
        """SwarmLimits must be immutable (frozen=True)."""
        limits = SwarmLimits()

        with pytest.raises(ValidationError):
            limits.max_depth = 99

    def test_swarm_limits_defaults(self) -> None:
        """Default ceilings must match the documented values."""
        limits = SwarmLimits()

        assert limits.max_depth == 3
        assert limits.max_total_nodes == 10
        assert limits.max_concurrent_leaves == 4


# ---------------------------------------------------------------------------
# ContextObservation
# ---------------------------------------------------------------------------


class TestContextObservation:
    """Tests for ContextObservation defaults."""

    def test_context_observation_defaults(self) -> None:
        """Timestamp must be auto-created; relevance_tags default to empty."""
        obs = ContextObservation(
            agent_id="tara-001",
            observation="Found the config file at /etc/app.conf",
        )

        assert obs.agent_id == "tara-001"
        assert obs.observation == "Found the config file at /etc/app.conf"
        assert obs.timestamp is not None
        assert isinstance(obs.relevance_tags, list)
        assert len(obs.relevance_tags) == 0


# ---------------------------------------------------------------------------
# BudgetExtensionRequest
# ---------------------------------------------------------------------------


class TestBudgetExtensionRequest:
    """Tests for BudgetExtensionRequest model."""

    def test_budget_extension_request(
        self, budget_request_with_artifacts: BudgetExtensionRequest
    ) -> None:
        """All fields must be accessible after creation."""
        req = budget_request_with_artifacts

        assert req.agent_id == "tara-budg-001"
        assert req.current_step == 2
        assert req.steps_requested == 2
        assert "error handling" in req.justification
        assert len(req.artifacts_produced_so_far) == 2
        assert len(req.tools_called_so_far) == 2
