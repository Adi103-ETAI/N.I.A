"""Unit tests for the OutputValidator (src/core/validation/__init__.py).

Tests validation rules for coder, researcher, and generic roles, plus the
apply_validation convenience function.
"""
from __future__ import annotations

import pytest

from src.core.validation import (
    OutputValidator,
    ValidationResult,
    ValidationVerdict,
    apply_validation,
)
from src.core.schema.mission import SubagentResult
from src.core.policy.scopes import CapabilityScope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result_dict(
    status: str = "success",
    output: str = "",
    artifacts: list[str] | None = None,
    agent_id: str = "test-validator",
) -> dict:
    """Shorthand factory for SubagentResult-style dicts."""
    return SubagentResult(
        agent_id=agent_id,
        status=status,
        output=output,
        artifacts_created=artifacts or [],
    ).model_dump()


# ---------------------------------------------------------------------------
# Coder role validation
# ---------------------------------------------------------------------------


class TestValidateCoder:
    """Validation rules for the 'coder' role."""

    def test_validate_code_success_no_artifacts_fails(self) -> None:
        """A coder reporting success with no artifacts must FAIL validation."""
        validator = OutputValidator()
        result_dict = _make_result_dict(
            status="success",
            output="Done with the work, everything is complete.",
            artifacts=[],
        )

        vr: ValidationResult = validator.validate(result_dict, role="coder")

        assert vr.verdict == ValidationVerdict.FAIL

    def test_validate_code_success_with_artifacts_passes(self) -> None:
        """A coder reporting success with artifacts must PASS validation."""
        validator = OutputValidator()
        result_dict = _make_result_dict(
            status="success",
            output="Created main.py with the requested feature implementation.",
            artifacts=["main.py"],
        )

        vr: ValidationResult = validator.validate(result_dict, role="coder")

        assert vr.verdict == ValidationVerdict.PASS


# ---------------------------------------------------------------------------
# Researcher role validation
# ---------------------------------------------------------------------------


class TestValidateResearcher:
    """Validation rules for the 'researcher' role."""

    def test_validate_research_too_short_fails(self) -> None:
        """A researcher with output shorter than threshold must FAIL."""
        validator = OutputValidator()
        result_dict = _make_result_dict(status="success", output="Short")

        vr: ValidationResult = validator.validate(result_dict, role="researcher")

        assert vr.verdict == ValidationVerdict.FAIL

    def test_validate_research_adequate_passes(self) -> None:
        """A researcher with adequate output length must PASS."""
        validator = OutputValidator()
        long_output = (
            "After extensive research into the topic, I found multiple relevant "
            "sources that confirm the following findings. The primary approach "
            "involves configuring the system parameters according to the "
            "documentation. Here are the detailed steps and considerations "
            "that need to be addressed for a successful implementation."
        )
        result_dict = _make_result_dict(status="success", output=long_output)

        vr: ValidationResult = validator.validate(result_dict, role="researcher")

        assert vr.verdict == ValidationVerdict.PASS


# ---------------------------------------------------------------------------
# Generic validation
# ---------------------------------------------------------------------------


class TestValidateGeneric:
    """Validation rules that apply to unknown/generic roles."""

    def test_validate_generic_empty_output_fails(self) -> None:
        """Success with completely empty output must FAIL."""
        validator = OutputValidator()
        result_dict = _make_result_dict(status="success", output="")

        vr: ValidationResult = validator.validate(result_dict, role="generic")

        assert vr.verdict == ValidationVerdict.FAIL

    def test_validate_generic_adequate_output_passes(self) -> None:
        """Success with adequate output must PASS."""
        validator = OutputValidator()
        result_dict = _make_result_dict(
            status="success",
            output="The task has been completed successfully with all expected outcomes.",
        )

        vr: ValidationResult = validator.validate(result_dict, role="generic")

        assert vr.verdict == ValidationVerdict.PASS


# ---------------------------------------------------------------------------
# apply_validation
# ---------------------------------------------------------------------------


class TestApplyValidation:
    """Tests for the apply_validation convenience function."""

    def test_apply_validation_fail_changes_status(self) -> None:
        """A FAIL ValidationResult must change status to 'failed'."""
        result_dict = _make_result_dict(
            status="success",
            output="Done",
            artifacts=[],
        )

        updated = apply_validation(result_dict, role="coder")

        assert updated["status"] == "failed"

    def test_apply_validation_pass_unchanged(self) -> None:
        """A PASS ValidationResult must leave the status as 'success'."""
        result_dict = _make_result_dict(
            status="success",
            output="Created main.py with the feature implementation as requested.",
            artifacts=["main.py"],
        )

        updated = apply_validation(result_dict, role="coder")

        assert updated["status"] == "success"

    def test_apply_validation_returns_dict(self) -> None:
        """apply_validation must return a dict."""
        result_dict = _make_result_dict(status="success", output="ok" * 50)

        updated = apply_validation(result_dict, role="generic")

        assert isinstance(updated, dict)
