"""Tests for the delegate_task tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from niaharness.api.client import ApiMessageCompleteEvent, ApiTextDeltaEvent
from niaharness.api.usage import UsageSnapshot
from niaharness.engine.messages import ConversationMessage, TextBlock
from niaharness.engine.stream_events import AssistantTextDelta, AssistantTurnComplete
from niaharness.tools.base import ToolExecutionContext
from niaharness.tools.delegate_task_tool import (
    DELEGATE_BLOCKED_TOOLS,
    DelegateTaskTool,
    DelegateTaskInput,
)


@pytest.fixture
def context(tmp_path: Path) -> ToolExecutionContext:
    """Build a context with a mock api_client and tool_registry."""
    from niaharness.tools import create_default_tool_registry

    registry = create_default_tool_registry()
    mock_api_client = MagicMock()

    return ToolExecutionContext(
        cwd=tmp_path,
        metadata={
            "tool_registry": registry,
            "api_client": mock_api_client,
            "model": "test-model",
            "max_tokens": 4096,
            "_delegate_depth": 0,
        },
    )


# ---------------------------------------------------------------------------
# Schema / blocked tools
# ---------------------------------------------------------------------------


class TestBlockedTools:
    def test_delegate_task_is_blocked(self):
        assert "delegate_task" in DELEGATE_BLOCKED_TOOLS

    def test_ask_user_question_is_blocked(self):
        assert "ask_user_question" in DELEGATE_BLOCKED_TOOLS

    def test_nia_memory_is_blocked(self):
        assert "nia_memory" in DELEGATE_BLOCKED_TOOLS

    def test_skill_manage_is_blocked(self):
        assert "skill_manage" in DELEGATE_BLOCKED_TOOLS


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_no_goal_no_tasks_returns_error(self, context: ToolExecutionContext):
        result = await DelegateTaskTool().execute(
            DelegateTaskInput(),
            context,
        )
        assert result.is_error is True
        assert "goal" in result.output.lower() or "tasks" in result.output.lower()

    @pytest.mark.asyncio
    async def test_depth_limit_exceeded(self, tmp_path: Path):
        """When delegation depth exceeds the limit, return an error."""
        from niaharness.tools import create_default_tool_registry

        ctx = ToolExecutionContext(
            cwd=tmp_path,
            metadata={"_delegate_depth": 99},  # way over the limit
        )
        result = await DelegateTaskTool().execute(
            DelegateTaskInput(goal="test"),
            ctx,
        )
        assert result.is_error is True
        assert "depth" in result.output.lower()

    @pytest.mark.asyncio
    async def test_depth_limit_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """NIA_DELEGATE_MAX_DEPTH env var should override the default."""
        monkeypatch.setenv("NIA_DELEGATE_MAX_DEPTH", "1")
        ctx = ToolExecutionContext(
            cwd=tmp_path,
            metadata={"_delegate_depth": 1},  # equals the limit
        )
        result = await DelegateTaskTool().execute(
            DelegateTaskInput(goal="test"),
            ctx,
        )
        assert result.is_error is True
        assert "depth" in result.output.lower()


# ---------------------------------------------------------------------------
# Child registry building
# ---------------------------------------------------------------------------


class TestChildRegistry:
    def test_blocked_tools_excluded(self, context: ToolExecutionContext):
        tool = DelegateTaskTool()
        registry = tool._build_child_registry(None, context)
        tool_names = {t.name for t in registry.list_tools()}
        for blocked in DELEGATE_BLOCKED_TOOLS:
            assert blocked not in tool_names, f"{blocked} should be blocked"

    def test_whitelist_restricts_tools(self, context: ToolExecutionContext):
        tool = DelegateTaskTool()
        registry = tool._build_child_registry("read_file,bash,grep", context)
        tool_names = {t.name for t in registry.list_tools()}
        assert "read_file" in tool_names
        assert "bash" in tool_names
        assert "grep" in tool_names
        assert "write_file" not in tool_names
        assert "web_search" not in tool_names

    def test_whitelist_still_blocks_blocked_tools(self, context: ToolExecutionContext):
        """Even if the whitelist includes a blocked tool, it should be excluded."""
        tool = DelegateTaskTool()
        registry = tool._build_child_registry("read_file,delegate_task", context)
        tool_names = {t.name for t in registry.list_tools()}
        assert "read_file" in tool_names
        assert "delegate_task" not in tool_names  # blocked even if whitelisted


# ---------------------------------------------------------------------------
# Read-only flag
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_is_not_read_only(self):
        tool = DelegateTaskTool()
        assert tool.is_read_only(DelegateTaskInput(goal="test")) is False


# ---------------------------------------------------------------------------
# Mocked subagent execution
# ---------------------------------------------------------------------------


class TestMockedExecution:
    @pytest.mark.asyncio
    async def test_single_mode_returns_result(self, context: ToolExecutionContext):
        """Test that a single subagent call returns the result text."""
        # Mock the QueryEngine to return a simple response.
        mock_engine = MagicMock()
        mock_events = [
            AssistantTextDelta(text="Here is the answer."),
            AssistantTurnComplete(
                message=ConversationMessage(
                    role="assistant", content=[TextBlock(text="Here is the answer.")]
                ),
                usage=UsageSnapshot(input_tokens=10, output_tokens=5),
            ),
        ]

        async def _mock_submit(msg):
            for ev in mock_events:
                yield ev

        mock_engine.submit_message = _mock_submit

        with patch("niaharness.tools.delegate_task_tool.DelegateTaskTool._build_child_registry") as mock_reg:
            mock_reg.return_value = context.metadata["tool_registry"]
            with patch("niaharness.engine.query_engine.QueryEngine", return_value=mock_engine):
                result = await DelegateTaskTool().execute(
                    DelegateTaskInput(goal="What is 2+2?"),
                    context,
                )

        assert result.is_error is False
        assert "Here is the answer." in result.output
        assert "1 turns" in result.output

    @pytest.mark.asyncio
    async def test_batch_mode_returns_all_results(self, context: ToolExecutionContext):
        """Test that batch mode returns results for all tasks."""
        mock_engine = MagicMock()
        call_count = [0]

        async def _mock_submit(msg):
            call_count[0] += 1
            yield AssistantTextDelta(text=f"Result for task {call_count[0]}")
            yield AssistantTurnComplete(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text=f"Result for task {call_count[0]}")],
                ),
                usage=UsageSnapshot(input_tokens=5, output_tokens=5),
            )

        mock_engine.submit_message = _mock_submit

        with patch("niaharness.tools.delegate_task_tool.DelegateTaskTool._build_child_registry") as mock_reg:
            mock_reg.return_value = context.metadata["tool_registry"]
            with patch("niaharness.engine.query_engine.QueryEngine", return_value=mock_engine):
                result = await DelegateTaskTool().execute(
                    DelegateTaskInput(
                        tasks=[
                            {"goal": "Task A"},
                            {"goal": "Task B"},
                            {"goal": "Task C"},
                        ]
                    ),
                    context,
                )

        assert result.is_error is False
        assert "Batch delegation complete (3 tasks)" in result.output
        assert "Result for task 1" in result.output
        assert "Result for task 2" in result.output
        assert "Result for task 3" in result.output

    @pytest.mark.asyncio
    async def test_no_api_client_returns_error(self, tmp_path: Path):
        """When no api_client is in the context, return an error."""
        from niaharness.tools import create_default_tool_registry

        ctx = ToolExecutionContext(
            cwd=tmp_path,
            metadata={
                "tool_registry": create_default_tool_registry(),
                # No api_client!
                "model": "test",
            },
        )
        result = await DelegateTaskTool().execute(
            DelegateTaskInput(goal="test"),
            ctx,
        )
        assert result.is_error is True
        assert "API client" in result.output or "api_client" in result.output.lower()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_format_single_result(self):
        tool = DelegateTaskTool()
        result = {
            "goal": "test",
            "label": "Task",
            "result": "The answer is 42.",
            "turns": 3,
            "usage": {"input": 100, "output": 50},
            "error": None,
        }
        formatted = tool._format_single_result(result)
        assert "3 turns" in formatted
        assert "The answer is 42." in formatted
        assert "100" in formatted  # usage

    def test_format_batch_results(self):
        tool = DelegateTaskTool()
        results = [
            {
                "goal": "Task A",
                "label": "Task 1",
                "result": "Did A.",
                "turns": 2,
                "usage": {"input": 10, "output": 5},
                "error": None,
            },
            Exception("Connection failed"),
        ]
        formatted = tool._format_batch_results(results, 2)
        assert "Batch delegation complete (2 tasks)" in formatted
        assert "Task 1" in formatted
        assert "Did A." in formatted
        assert "Connection failed" in formatted
