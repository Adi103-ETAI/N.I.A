"""Unit tests for the Reflect/Reformulate engine (src/agents/nia/subagents/reflect.py).

Tests evaluate_budget_extension heuristic logic and build_escalation_message
output formatting. reflect_and_reformulate is async + LLM-backed, so it is
tested with mocked LLM responses.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.schema.coordinator import BudgetExtensionRequest
from src.agents.nia.subagents.reflect import (
    build_escalation_message,
    evaluate_budget_extension,
    reflect_and_reformulate,
)


# ---------------------------------------------------------------------------
# evaluate_budget_extension
# ---------------------------------------------------------------------------


class TestEvaluateBudgetExtension:
    """Tests for the heuristic budget-extension gate."""

    @pytest.mark.asyncio
    async def test_evaluate_budget_no_artifacts_late_denies(
        self, budget_request_no_artifacts_late: BudgetExtensionRequest
    ) -> None:
        """No artifacts produced at step >= 3 must be denied."""
        result = await evaluate_budget_extension(budget_request_no_artifacts_late)
        assert result is False

    @pytest.mark.asyncio
    async def test_evaluate_budget_too_many_steps_denies(
        self, budget_request_too_many_steps: BudgetExtensionRequest
    ) -> None:
        """Requesting more steps than max_extra_steps must be denied."""
        result = await evaluate_budget_extension(
            budget_request_too_many_steps, max_extra_steps=3
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_evaluate_budget_has_artifacts_grants(
        self, budget_request_with_artifacts: BudgetExtensionRequest
    ) -> None:
        """Artifacts present indicates progress; must be granted."""
        result = await evaluate_budget_extension(budget_request_with_artifacts)
        assert result is True

    @pytest.mark.asyncio
    async def test_evaluate_budget_early_stage_grants(
        self, budget_request_early_no_artifacts: BudgetExtensionRequest
    ) -> None:
        """Early stage (step < 3) with no artifacts gets benefit of the doubt."""
        result = await evaluate_budget_extension(budget_request_early_no_artifacts)
        assert result is True

    @pytest.mark.asyncio
    async def test_evaluate_budget_boundary_step_2_no_artifacts_grants(self) -> None:
        """Step == 2 with no artifacts should still grant (< 3)."""
        request = BudgetExtensionRequest(
            agent_id="boundary",
            current_step=2,
            steps_requested=1,
            justification="Almost there.",
            artifacts_produced_so_far=[],
            tools_called_so_far=[],
        )
        result = await evaluate_budget_extension(request)
        assert result is True

    @pytest.mark.asyncio
    async def test_evaluate_budget_boundary_step_3_no_artifacts_denies(self) -> None:
        """Step == 3 with no artifacts must be denied (>= 3)."""
        request = BudgetExtensionRequest(
            agent_id="boundary",
            current_step=3,
            steps_requested=1,
            justification="Still working.",
            artifacts_produced_so_far=[],
            tools_called_so_far=[],
        )
        result = await evaluate_budget_extension(request)
        assert result is False


# ---------------------------------------------------------------------------
# build_escalation_message
# ---------------------------------------------------------------------------


class TestBuildEscalationMessage:
    """Tests for the escalation message builder."""

    def test_build_escalation_message_includes_objective(self) -> None:
        """The objective text must appear in the escalation message."""
        objective = "Deploy the microservice to staging"
        results = [
            {
                "agent_id": "tara-001",
                "status": "failed",
                "output": "Connection timeout",
                "failure_trace": "TimeoutError",
            }
        ]

        message = build_escalation_message(objective, results)
        assert objective in message

    def test_build_escalation_message_max_retries(self) -> None:
        """When max_retries_hit=True, the message must mention retry exhaustion."""
        objective = "Run integration tests"
        results = [
            {"agent_id": "tara-002", "status": "failed", "output": "Test failures"},
            {"agent_id": "tara-003", "status": "failed", "output": "Test failures"},
        ]

        message = build_escalation_message(
            objective, results, max_retries_hit=True
        )
        assert "exhausted" in message.lower() or "retry" in message.lower()

    def test_build_escalation_message_scope_violation(self) -> None:
        """Scope violation results should trigger permission-related help text."""
        objective = "Write to system directory"
        results = [
            {
                "agent_id": "tara-004",
                "status": "scope_violation",
                "output": "WRITE scope not approved",
            }
        ]

        message = build_escalation_message(objective, results)
        assert "permission" in message.lower() or "scope" in message.lower()

    def test_build_escalation_message_empty_results(self) -> None:
        """An empty results list should still produce a valid message."""
        message = build_escalation_message("Do something", [])
        assert "Do something" in message

    def test_build_escalation_message_includes_attempt_count(self) -> None:
        """Message must report the number of attempts."""
        results = [
            {"agent_id": "a", "status": "failed", "output": "err1"},
            {"agent_id": "b", "status": "failed", "output": "err2"},
            {"agent_id": "c", "status": "failed", "output": "err3"},
        ]
        message = build_escalation_message("objective", results)
        assert "3" in message


# ---------------------------------------------------------------------------
# reflect_and_reformulate (mocked LLM)
# ---------------------------------------------------------------------------


class TestReflectAndReformulate:
    """Tests for LLM-backed reflection — LLM calls are fully mocked."""

    @pytest.mark.asyncio
    async def test_reflect_returns_reformulated_objective(self) -> None:
        """When the LLM returns a non-empty string, it should be returned."""
        mock_response = MagicMock()
        mock_response.content = "Use a different API endpoint to avoid the timeout."

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "src.agents.nia.subagents.reflect._get_llm",
            return_value=mock_llm,
        ):
            result = await reflect_and_reformulate(
                original_objective="Call the API",
                failure_trace="TimeoutError after 30s",
                attempt_number=1,
            )

        assert result == "Use a different API endpoint to avoid the timeout."

    @pytest.mark.asyncio
    async def test_reflect_fallback_on_llm_error(self) -> None:
        """When the LLM raises, the original objective should be returned."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))

        with patch(
            "src.agents.nia.subagents.reflect._get_llm",
            return_value=mock_llm,
        ):
            result = await reflect_and_reformulate(
                original_objective="Original task",
                failure_trace="Some error",
                attempt_number=1,
            )

        assert result == "Original task"

    @pytest.mark.asyncio
    async def test_reflect_fallback_on_empty_response(self) -> None:
        """When the LLM returns empty content, the original is returned."""
        mock_response = MagicMock()
        mock_response.content = "   "

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "src.agents.nia.subagents.reflect._get_llm",
            return_value=mock_llm,
        ):
            result = await reflect_and_reformulate(
                original_objective="Original task",
                failure_trace="Error details",
                attempt_number=2,
            )

        assert result == "Original task"
