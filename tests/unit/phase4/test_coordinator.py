"""Unit tests for the Coordinator router and state factory.

Tests coordinator_router branching logic and create_coordinator_state
initial values.
"""
from __future__ import annotations

import pytest

from src.core.policy.scopes import CapabilityScope
from src.core.schema.mission import MissionManifest, PlanStep
from src.agents.nia.subagents.coordinator import coordinator_router
from src.agents.nia.subagents.state import create_coordinator_state


# ---------------------------------------------------------------------------
# coordinator_router
# ---------------------------------------------------------------------------


class TestCoordinatorRouter:
    """Tests for coordinator_router — deterministic routing decisions."""

    def test_coordinator_router_executing_returns_dispatch(self) -> None:
        """status='executing' must route to 'dispatch'."""
        state = {"status": "executing"}
        assert coordinator_router(state) == "dispatch"

    def test_coordinator_router_reflecting_returns_reflect(self) -> None:
        """status='reflecting' must route to 'reflect'."""
        state = {"status": "reflecting"}
        assert coordinator_router(state) == "reflect"

    def test_coordinator_router_completed_returns_end(self) -> None:
        """status='completed' must route to '__end__'."""
        state = {"status": "completed"}
        assert coordinator_router(state) == "__end__"

    def test_coordinator_router_failed_returns_end(self) -> None:
        """status='failed' must route to '__end__'."""
        state = {"status": "failed"}
        assert coordinator_router(state) == "__end__"

    def test_coordinator_router_needs_human_returns_end(self) -> None:
        """status='needs_human' must route to '__end__'."""
        state = {"status": "needs_human"}
        assert coordinator_router(state) == "__end__"

    def test_coordinator_router_unknown_defaults_to_end(self) -> None:
        """An unrecognised status should default to '__end__'."""
        state = {"status": "some_unknown_status"}
        assert coordinator_router(state) == "__end__"

    def test_coordinator_router_missing_status_defaults_to_end(self) -> None:
        """A state dict with no 'status' key should default to '__end__'."""
        state = {}
        assert coordinator_router(state) == "__end__"


# ---------------------------------------------------------------------------
# create_coordinator_state
# ---------------------------------------------------------------------------


class TestCreateCoordinatorState:
    """Tests for the coordinator state factory function."""

    def test_create_coordinator_state_pending_steps(
        self, approved_manifest: MissionManifest
    ) -> None:
        """pending_steps must match the number of steps in the manifest."""
        state = create_coordinator_state(approved_manifest)

        assert len(state["pending_steps"]) == len(approved_manifest.steps)

        # Each pending step must carry the step_index and description.
        for idx, pending in enumerate(state["pending_steps"]):
            assert pending["step_index"] == idx
            assert pending["description"] == approved_manifest.steps[idx].description
            assert "assigned_role" in pending
            assert "required_scopes" in pending

    def test_create_coordinator_state_initial_status(
        self, approved_manifest: MissionManifest
    ) -> None:
        """Initial status must be 'planning'."""
        state = create_coordinator_state(approved_manifest)
        assert state["status"] == "planning"

    def test_create_coordinator_state_empty_collections(
        self, approved_manifest: MissionManifest
    ) -> None:
        """Active tasks, completed results, context log must start empty."""
        state = create_coordinator_state(approved_manifest)

        assert state["active_tasks"] == []
        assert state["completed_results"] == []
        assert state["context_log"] == []
        assert state["budget_extensions"] == []

    def test_create_coordinator_state_counters_zero(
        self, approved_manifest: MissionManifest
    ) -> None:
        """Tree depth and total_nodes_spawned must start at zero."""
        state = create_coordinator_state(approved_manifest)

        assert state["tree_depth"] == 0
        assert state["total_nodes_spawned"] == 0

    def test_create_coordinator_state_mission_round_trip(
        self, approved_manifest: MissionManifest
    ) -> None:
        """The serialised mission dict must round-trip back to a MissionManifest."""
        state = create_coordinator_state(approved_manifest)
        restored = MissionManifest(**state["mission"])

        assert restored.mission_id == approved_manifest.mission_id
        assert restored.intent == approved_manifest.intent
        assert len(restored.steps) == len(approved_manifest.steps)
