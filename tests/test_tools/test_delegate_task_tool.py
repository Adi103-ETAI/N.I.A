"""Tests for the delegate_task tool (post-audit-fix version)."""

from __future__ import annotations

import json
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
    """Build a context with api_client and tool_registry (post-fix)."""
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
# Blocked tools (expanded blocklist)
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

    def test_send_message_is_blocked(self):
        assert "send_message" in DELEGATE_BLOCKED_TOOLS

    def test_run_code_is_blocked(self):
        assert "run_code" in DELEGATE_BLOCKED_TOOLS

    def test_agent_is_blocked(self):
        assert "agent" in DELEGATE_BLOCKED_TOOLS

    def test_task_create_is_blocked(self):
        assert "task_create" in DELEGATE_BLOCKED_TOOLS

    def test_browser_is_not_blocked(self):
        # browser is allowed — useful for subagents
        assert "browser" not in DELEGATE_BLOCKED_TOOLS

    def test_read_file_is_not_blocked(self):
        assert "read_file" not in DELEGATE_BLOCKED_TOOLS

    def test_bash_is_not_blocked(self):
        assert "bash" not in DELEGATE_BLOCKED_TOOLS


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_no_goal_no_tasks_returns_error(self, context: ToolExecutionContext):
        result = await DelegateTaskTool().execute(DelegateTaskInput(), context)
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_depth_limit_exceeded(self, tmp_path: Path):
        ctx = ToolExecutionContext(cwd=tmp_path, metadata={"_delegate_depth": 99})
        result = await DelegateTaskTool().execute(
            DelegateTaskInput(goal="test"), ctx
        )
        assert result.is_error is True
        assert "depth" in result.output.lower()

    @pytest.mark.asyncio
    async def test_depth_limit_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NIA_DELEGATE_MAX_DEPTH", "1")
        ctx = ToolExecutionContext(cwd=tmp_path, metadata={"_delegate_depth": 1})
        result = await DelegateTaskTool().execute(
            DelegateTaskInput(goal="test"), ctx
        )
        assert result.is_error is True


# ---------------------------------------------------------------------------
# Child registry (expanded blocklist)
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

    def test_whitelist_still_blocks_blocked_tools(self, context: ToolExecutionContext):
        tool = DelegateTaskTool()
        registry = tool._build_child_registry("read_file,delegate_task,send_message", context)
        tool_names = {t.name for t in registry.list_tools()}
        assert "read_file" in tool_names
        assert "delegate_task" not in tool_names
        assert "send_message" not in tool_names


# ---------------------------------------------------------------------------
# Read-only flag
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_is_not_read_only(self):
        assert DelegateTaskTool().is_read_only(DelegateTaskInput(goal="test")) is False


# ---------------------------------------------------------------------------
# Mocked subagent execution
# ---------------------------------------------------------------------------


class TestMockedExecution:
    @pytest.mark.asyncio
    async def test_single_mode_returns_structured_json(self, context: ToolExecutionContext):
        """Test that a single subagent call returns structured JSON."""
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

        with patch("niaharness.tools.delegate_task_tool.DelegateTaskTool._build_child_registry", return_value=context.metadata["tool_registry"]):
            with patch("niaharness.engine.query_engine.QueryEngine", return_value=mock_engine):
                result = await DelegateTaskTool().execute(
                    DelegateTaskInput(goal="What is 2+2?"),
                    context,
                )

        assert result.is_error is False
        # Output should be valid JSON with structured fields.
        parsed = json.loads(result.output)
        assert parsed["result"] == "Here is the answer."
        assert parsed["status"] == "completed"
        assert parsed["exit_reason"] == "completed"
        assert parsed["turns"] == 1
        assert parsed["model"] == "test-model"
        assert "duration_seconds" in parsed
        assert parsed["usage"]["input"] == 10
        assert parsed["usage"]["output"] == 5

    @pytest.mark.asyncio
    async def test_batch_mode_returns_structured_json(self, context: ToolExecutionContext):
        """Test that batch mode returns structured JSON with results array."""
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

        with patch("niaharness.tools.delegate_task_tool.DelegateTaskTool._build_child_registry", return_value=context.metadata["tool_registry"]):
            with patch("niaharness.engine.query_engine.QueryEngine", return_value=mock_engine):
                result = await DelegateTaskTool().execute(
                    DelegateTaskInput(tasks=[{"goal": "Task A"}, {"goal": "Task B"}, {"goal": "Task C"}]),
                    context,
                )

        assert result.is_error is False
        parsed = json.loads(result.output)
        assert "results" in parsed
        assert parsed["total_tasks"] == 3
        assert len(parsed["results"]) == 3
        # Each result should have structured fields.
        for r in parsed["results"]:
            assert "status" in r
            assert "exit_reason" in r
            assert "task_index" in r

    @pytest.mark.asyncio
    async def test_no_api_client_returns_structured_error(self, tmp_path: Path):
        """When no api_client is in the context, return a structured error."""
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
            DelegateTaskInput(goal="test"), ctx
        )
        assert result.is_error is True
        parsed = json.loads(result.output)
        assert parsed["status"] == "failed"
        assert parsed["exit_reason"] == "no_api_client"
        assert "api_client" in parsed["error"]


# ---------------------------------------------------------------------------
# Depth propagation
# ---------------------------------------------------------------------------


class TestDepthPropagation:
    @pytest.mark.asyncio
    async def test_child_gets_incremented_depth(self, context: ToolExecutionContext):
        """The child QueryEngine should receive _delegate_depth = parent_depth + 1."""
        mock_engine = MagicMock()

        async def _mock_submit(msg):
            yield AssistantTurnComplete(
                message=ConversationMessage(
                    role="assistant", content=[TextBlock(text="done")]
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )

        mock_engine.submit_message = _mock_submit

        captured_kwargs = {}
        def _capture_init(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_engine

        with patch("niaharness.tools.delegate_task_tool.DelegateTaskTool._build_child_registry", return_value=context.metadata["tool_registry"]):
            with patch("niaharness.engine.query_engine.QueryEngine", side_effect=_capture_init):
                await DelegateTaskTool().execute(
                    DelegateTaskInput(goal="test"), context
                )

        # The child should have _delegate_depth = 1 in its tool_metadata.
        child_metadata = captured_kwargs.get("tool_metadata", {})
        assert child_metadata.get("_delegate_depth") == 1
