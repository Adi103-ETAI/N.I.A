"""Unit tests for the Policy Engine (src/core/policy/engine.py).

Tests audit_plan scope classification and enforce_at_runtime permission checks.
"""
from __future__ import annotations

import pytest

from src.core.policy.engine import (
    CapabilityAudit,
    ScopeViolation,
    audit_plan,
    enforce_at_runtime,
)
from src.core.policy.scopes import CapabilityScope
from src.core.schema.mission import MissionManifest, PlanStep


# ---------------------------------------------------------------------------
# audit_plan
# ---------------------------------------------------------------------------


class TestAuditPlan:
    """Tests for audit_plan — scope classification before human approval."""

    def test_audit_plan_read_only_auto_approved(
        self, read_only_manifest: MissionManifest
    ) -> None:
        """READ_ONLY scopes should appear in auto_approved, not needs_approval."""
        audit: CapabilityAudit = audit_plan(read_only_manifest)

        assert CapabilityScope.READ_ONLY in audit.auto_approved
        assert len(audit.needs_approval) == 0

    def test_audit_plan_write_needs_approval(
        self, write_manifest: MissionManifest
    ) -> None:
        """WRITE (and any non-READ_ONLY) scopes should land in needs_approval."""
        audit: CapabilityAudit = audit_plan(write_manifest)

        assert CapabilityScope.WRITE in audit.needs_approval
        assert CapabilityScope.READ_ONLY in audit.auto_approved

    def test_audit_plan_all_non_read_scopes_need_approval(
        self, multi_scope_manifest: MissionManifest
    ) -> None:
        """Every scope except READ_ONLY must require approval."""
        audit: CapabilityAudit = audit_plan(multi_scope_manifest)

        for scope in audit.needs_approval:
            assert scope != CapabilityScope.READ_ONLY

        assert CapabilityScope.READ_ONLY in audit.auto_approved
        assert len(audit.needs_approval) >= 1


# ---------------------------------------------------------------------------
# enforce_at_runtime
# ---------------------------------------------------------------------------


class TestEnforceAtRuntime:
    """Tests for enforce_at_runtime — runtime permission gate."""

    def test_enforce_runtime_approved_scope_passes(
        self, approved_manifest: MissionManifest
    ) -> None:
        """Approved scopes must return True without raising."""
        result = enforce_at_runtime(CapabilityScope.WRITE, approved_manifest)
        assert result is True

    def test_enforce_runtime_unapproved_scope_raises(
        self, approved_manifest: MissionManifest
    ) -> None:
        """A scope not in approved_scopes must raise ScopeViolation."""
        # approved_manifest has READ_ONLY, WRITE, EXECUTE but NOT DESTRUCTIVE
        with pytest.raises(ScopeViolation):
            enforce_at_runtime(CapabilityScope.DESTRUCTIVE, approved_manifest)

    def test_enforce_runtime_unapproved_mission_raises(
        self, write_manifest: MissionManifest
    ) -> None:
        """When manifest.approved is False, non-read scopes must raise."""
        assert write_manifest.approved is False

        with pytest.raises(ScopeViolation, match="has not been approved"):
            enforce_at_runtime(CapabilityScope.WRITE, write_manifest)

    def test_enforce_runtime_read_only_always_passes(
        self, write_manifest: MissionManifest
    ) -> None:
        """READ_ONLY must always pass, even when the mission is not approved."""
        assert write_manifest.approved is False

        result = enforce_at_runtime(CapabilityScope.READ_ONLY, write_manifest)
        assert result is True
