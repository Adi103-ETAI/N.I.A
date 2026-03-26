"""Output Correctness Validation — Sprint 6.

Lightweight sanity checks on SubagentResult outputs.  These are NOT
full evaluations — they catch the "structurally valid but empty"
failure mode that Pydantic cannot detect.

Validation failures do NOT crash the system — they mark the result
as ``failed`` so the reflect/reformulate loop can retry with a
better objective.

Public API:
    OutputValidator       — configurable validator with role-specific methods.
    ValidationResult      — verdict + reasons Pydantic model.
    ValidationVerdict     — PASS / WARN / FAIL enum.
    apply_validation      — convenience: validate & patch a result dict in-place.
    get_validator         — module-level singleton accessor.
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import List

from pydantic import BaseModel, Field

logger = logging.getLogger("NIA.Validation")

# ---------------------------------------------------------------------------
# Rubber-stamp patterns (case-insensitive)
# ---------------------------------------------------------------------------
_RUBBER_STAMP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*looks?\s+good\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*lgtm\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*no\s+issues?\s+(found|detected)\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*everything\s+(looks?|seems?)\s+(fine|good|ok|okay)\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*approved\.?\s*$", re.IGNORECASE),
]

# Shallow one-liner patterns for research output
_SHALLOW_RESEARCH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*i\s+found\s+.{0,40}\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*here\s+is\s+.{0,40}\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*the\s+answer\s+is\s+.{0,40}\.?\s*$", re.IGNORECASE),
]


# ============================================================================
# Verdict Enum
# ============================================================================

class ValidationVerdict(str, Enum):
    """Outcome of a correctness validation pass."""

    PASS = "pass"
    WARN = "warn"   # suspicious but allow
    FAIL = "fail"   # treat as failed, trigger reflect


# ============================================================================
# ValidationResult
# ============================================================================

class ValidationResult(BaseModel):
    """Aggregated outcome from one or more validation checks."""

    verdict: ValidationVerdict
    reasons: List[str] = Field(default_factory=list)
    original_status: str = ""  # the SubagentResult.status before validation


# ============================================================================
# OutputValidator
# ============================================================================

class OutputValidator:
    """Validates SubagentResult outputs beyond structural correctness.

    All checks are pure heuristics — no LLM calls, no filesystem access.
    The validator is intentionally lenient: only clear emptiness / rubber-
    stamping triggers a FAIL; borderline cases emit WARN.
    """

    # Role -> method dispatch table (populated in __init__)
    _dispatch: dict[str, str]

    def __init__(
        self,
        min_output_length: int = 20,
        min_research_length: int = 100,
    ) -> None:
        self._min_output_length = min_output_length
        self._min_research_length = min_research_length
        self._dispatch = {
            "coder": "validate_code_output",
            "researcher": "validate_research_output",
            "reviewer": "validate_review_output",
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def validate(
        self,
        result_dict: dict,
        role: str = "tara",
    ) -> ValidationResult:
        """Run all validators and return a combined result.

        Dispatches to a role-specific validator when one exists for *role*,
        otherwise falls back to :meth:`validate_generic`.

        Args:
            result_dict: A ``SubagentResult`` as a dict (from ``.model_dump()``).
            role: The ``assigned_role`` of the plan step
                  (``coder``, ``researcher``, ``reviewer``, etc.).

        Returns:
            A :class:`ValidationResult` with the combined verdict.
        """
        method_name = self._dispatch.get(role)
        if method_name is not None:
            method = getattr(self, method_name)
            return method(result_dict)
        return self.validate_generic(result_dict)

    # ------------------------------------------------------------------
    # Role-specific validators
    # ------------------------------------------------------------------

    def validate_code_output(self, result_dict: dict) -> ValidationResult:
        """Validate output from a coder agent.

        Checks:
            - If status is ``"success"``, ``artifacts_created`` must not be empty.
            - Output should mention what was created/modified.
            - Output should not be suspiciously short (< *min_output_length* chars).
        """
        reasons: list[str] = []
        verdict = ValidationVerdict.PASS
        status = result_dict.get("status", "")
        output = result_dict.get("output", "")
        artifacts = result_dict.get("artifacts_created", [])

        if status != "success":
            # Non-success results are not subject to correctness checks —
            # they will already be handled by the retry loop.
            return ValidationResult(
                verdict=ValidationVerdict.PASS,
                original_status=status,
            )

        # -- Artifact check ---------------------------------------------------
        if not artifacts:
            reasons.append(
                "Coder claimed success but artifacts_created is empty."
            )
            verdict = ValidationVerdict.FAIL

        # -- Output length check ----------------------------------------------
        if len(output.strip()) < self._min_output_length:
            reasons.append(
                f"Coder output is suspiciously short "
                f"({len(output.strip())} chars, minimum {self._min_output_length})."
            )
            # Short output with artifacts is a WARN; without artifacts is
            # already a FAIL from the check above.
            if verdict != ValidationVerdict.FAIL:
                verdict = ValidationVerdict.WARN

        # -- Substance check: output should reference creation/modification ---
        _creation_keywords = (
            "created", "modified", "wrote", "generated", "updated",
            "added", "implemented", "built", "saved", "file",
        )
        output_lower = output.lower()
        if (
            artifacts
            and not any(kw in output_lower for kw in _creation_keywords)
        ):
            reasons.append(
                "Coder output does not mention what was created or modified."
            )
            if verdict == ValidationVerdict.PASS:
                verdict = ValidationVerdict.WARN

        if reasons:
            logger.log(
                logging.WARNING if verdict == ValidationVerdict.FAIL else logging.DEBUG,
                "Code validation %s: %s",
                verdict.value,
                "; ".join(reasons),
            )

        return ValidationResult(
            verdict=verdict,
            reasons=reasons,
            original_status=status,
        )

    def validate_research_output(self, result_dict: dict) -> ValidationResult:
        """Validate output from a researcher agent.

        Checks:
            - Output must be longer than *min_research_length* chars.
            - Output should contain substantive content (not just
              ``"I found X"``).
            - Should not be a single generic sentence.
        """
        reasons: list[str] = []
        verdict = ValidationVerdict.PASS
        status = result_dict.get("status", "")
        output = result_dict.get("output", "")

        if status != "success":
            return ValidationResult(
                verdict=ValidationVerdict.PASS,
                original_status=status,
            )

        stripped = output.strip()

        # -- Length check -----------------------------------------------------
        if len(stripped) < self._min_research_length:
            reasons.append(
                f"Research output is too short "
                f"({len(stripped)} chars, minimum {self._min_research_length})."
            )
            verdict = ValidationVerdict.FAIL

        # -- Shallow one-liner check -----------------------------------------
        for pat in _SHALLOW_RESEARCH_PATTERNS:
            if pat.match(stripped):
                reasons.append(
                    "Research output appears to be a shallow one-liner "
                    "without substantive content."
                )
                verdict = ValidationVerdict.FAIL
                break

        # -- Single sentence check -------------------------------------------
        # A single sentence under the research threshold is suspicious.
        sentence_count = len(re.split(r"[.!?]+", stripped.strip(".!? ")))
        if sentence_count <= 1 and verdict == ValidationVerdict.PASS:
            reasons.append(
                "Research output appears to be a single sentence; "
                "expected more substantive analysis."
            )
            verdict = ValidationVerdict.WARN

        if reasons:
            logger.log(
                logging.WARNING if verdict == ValidationVerdict.FAIL else logging.DEBUG,
                "Research validation %s: %s",
                verdict.value,
                "; ".join(reasons),
            )

        return ValidationResult(
            verdict=verdict,
            reasons=reasons,
            original_status=status,
        )

    def validate_review_output(self, result_dict: dict) -> ValidationResult:
        """Validate output from a reviewer agent.

        Checks:
            - Output should not be a rubber-stamp (``"looks good"`` with no
              detail).
            - Should be longer than *min_output_length*.
        """
        reasons: list[str] = []
        verdict = ValidationVerdict.PASS
        status = result_dict.get("status", "")
        output = result_dict.get("output", "")

        if status != "success":
            return ValidationResult(
                verdict=ValidationVerdict.PASS,
                original_status=status,
            )

        stripped = output.strip()

        # -- Rubber-stamp check -----------------------------------------------
        for pat in _RUBBER_STAMP_PATTERNS:
            if pat.match(stripped):
                reasons.append(
                    "Review output appears to be a rubber-stamp with no "
                    "substantive feedback."
                )
                verdict = ValidationVerdict.FAIL
                break

        # -- Length check -----------------------------------------------------
        if len(stripped) < self._min_output_length:
            reasons.append(
                f"Review output is too short "
                f"({len(stripped)} chars, minimum {self._min_output_length})."
            )
            if verdict == ValidationVerdict.PASS:
                verdict = ValidationVerdict.FAIL

        if reasons:
            logger.log(
                logging.WARNING if verdict == ValidationVerdict.FAIL else logging.DEBUG,
                "Review validation %s: %s",
                verdict.value,
                "; ".join(reasons),
            )

        return ValidationResult(
            verdict=verdict,
            reasons=reasons,
            original_status=status,
        )

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def validate_generic(self, result_dict: dict) -> ValidationResult:
        """Fallback validator for unknown roles.

        Checks:
            - Status ``"success"`` implies non-empty output.
            - Output is not suspiciously short.
        """
        reasons: list[str] = []
        verdict = ValidationVerdict.PASS
        status = result_dict.get("status", "")
        output = result_dict.get("output", "")

        if status != "success":
            return ValidationResult(
                verdict=ValidationVerdict.PASS,
                original_status=status,
            )

        stripped = output.strip()

        # -- Empty output on success ------------------------------------------
        if not stripped:
            reasons.append(
                "Agent claimed success but produced empty output."
            )
            verdict = ValidationVerdict.FAIL

        # -- Short output warning --------------------------------------------
        elif len(stripped) < self._min_output_length:
            reasons.append(
                f"Agent output is suspiciously short "
                f"({len(stripped)} chars, minimum {self._min_output_length})."
            )
            verdict = ValidationVerdict.WARN

        if reasons:
            logger.log(
                logging.WARNING if verdict == ValidationVerdict.FAIL else logging.DEBUG,
                "Generic validation %s: %s",
                verdict.value,
                "; ".join(reasons),
            )

        return ValidationResult(
            verdict=verdict,
            reasons=reasons,
            original_status=status,
        )


# ============================================================================
# Convenience function
# ============================================================================

def apply_validation(result_dict: dict, role: str = "tara") -> dict:
    """Validate and update a result dict if needed.

    If validation verdict is **FAIL**:
        - Change ``status`` to ``"failed"``.
        - Append validation reasons to ``failure_trace``.
        - Preserve the original output unchanged.

    If validation verdict is **WARN**:
        - Keep the original status.
        - Append a warning note to ``output``.

    If validation verdict is **PASS**:
        - Return the dict unmodified.

    Args:
        result_dict: A ``SubagentResult``-style dict (e.g. from
            ``SubagentResult.model_dump()``).
        role: The ``assigned_role`` of the plan step.

    Returns:
        The (possibly modified) result dict.  Mutation is in-place for
        efficiency but the dict is also returned for chaining.
    """
    validator = get_validator()
    vr = validator.validate(result_dict, role=role)

    if vr.verdict == ValidationVerdict.PASS:
        return result_dict

    reason_text = "; ".join(vr.reasons)

    if vr.verdict == ValidationVerdict.FAIL:
        logger.warning(
            "Validation FAIL for agent %s (role=%s): %s — overriding status to 'failed'.",
            result_dict.get("agent_id", "?"),
            role,
            reason_text,
        )
        result_dict["status"] = "failed"
        existing_trace = result_dict.get("failure_trace") or ""
        separator = "\n" if existing_trace else ""
        result_dict["failure_trace"] = (
            f"{existing_trace}{separator}"
            f"[validation] {reason_text}"
        )

    elif vr.verdict == ValidationVerdict.WARN:
        logger.info(
            "Validation WARN for agent %s (role=%s): %s",
            result_dict.get("agent_id", "?"),
            role,
            reason_text,
        )
        existing_output = result_dict.get("output", "")
        result_dict["output"] = (
            f"{existing_output}\n\n"
            f"[validation warning] {reason_text}"
        )

    return result_dict


# ============================================================================
# Module-level singleton
# ============================================================================

_validator: OutputValidator | None = None


def get_validator() -> OutputValidator:
    """Get or create the global OutputValidator instance."""
    global _validator
    if _validator is None:
        _validator = OutputValidator()
    return _validator


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "ValidationVerdict",
    "ValidationResult",
    "OutputValidator",
    "apply_validation",
    "get_validator",
]
