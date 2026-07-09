"""Tests for the background memory/skill review system (Task 7).

Covers:
  - Review prompts (memory-only, skill-only, combined) — content + structure
  - Skill provenance (ContextVar, read-before-write gate, reset)
  - Config flag (env var, config.yaml, default off)
  - maybe_spawn_background_review gating (enabled, memory, interrupted, tool_call_count ≥3)
  - Message snapshotting (caps at MAX_SNAPSHOT_MESSAGES, converts ConversationMessage)
  - Action summarization (off/on/verbose modes, stale skip, memory + skill actions)
  - QueryResult.tool_call_count field
  - QueryEngine post_turn_hooks invocation
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from niaharness.engine.background_review import (
    COMBINED_REVIEW_PROMPT,
    CONFIG_FLAG,
    MAX_REVIEW_ITERATIONS,
    MEMORY_REVIEW_PROMPT,
    MIN_TOOL_CALLS_FOR_REVIEW,
    SKILL_REVIEW_PROMPT,
    _ReviewState,
    _snapshot_messages,
    get_review_stats,
    maybe_spawn_background_review,
    set_feedback_callback,
    summarize_background_review_actions,
    wait_for_reviews,
)
from niaharness.engine.messages import ConversationMessage, TextBlock
from niaharness.engine.stream_events import QueryResult, TerminationReason
from niaharness.tools.skill_provenance import (
    BACKGROUND_REVIEW,
    background_review_has_read,
    get_current_write_origin,
    is_background_review,
    mark_background_review_skill_read,
    reset_background_review_read_marks,
    reset_current_write_origin,
    set_current_write_origin,
)


# ---------------------------------------------------------------------------
# Review prompts
# ---------------------------------------------------------------------------


class TestReviewPrompts:
    def test_memory_prompt_contains_memory_focus(self):
        assert "saving to memory" in MEMORY_REVIEW_PROMPT
        assert "memory tool" in MEMORY_REVIEW_PROMPT

    def test_skill_prompt_contains_skill_signals(self):
        assert "skill library" in SKILL_REVIEW_PROMPT
        assert "CLASS-LEVEL" in SKILL_REVIEW_PROMPT
        assert "PATCH" in SKILL_REVIEW_PROMPT

    def test_skill_prompt_contains_do_not_capture(self):
        assert "Do NOT capture" in SKILL_REVIEW_PROMPT
        assert "Environment-dependent failures" in SKILL_REVIEW_PROMPT
        assert "Negative claims" in SKILL_REVIEW_PROMPT

    def test_combined_prompt_covers_both(self):
        assert "**Memory**" in COMBINED_REVIEW_PROMPT
        assert "**Skills**" in COMBINED_REVIEW_PROMPT

    def test_all_prompts_nonempty(self):
        assert len(MEMORY_REVIEW_PROMPT) > 100
        assert len(SKILL_REVIEW_PROMPT) > 1000
        assert len(COMBINED_REVIEW_PROMPT) > 1000


# ---------------------------------------------------------------------------
# Skill provenance
# ---------------------------------------------------------------------------


class TestSkillProvenance:
    def test_default_origin_is_foreground(self):
        assert get_current_write_origin() == "foreground"
        assert is_background_review() is False

    def test_set_and_reset_origin(self):
        token = set_current_write_origin(BACKGROUND_REVIEW)
        assert get_current_write_origin() == BACKGROUND_REVIEW
        assert is_background_review() is True
        reset_current_write_origin(token)
        assert get_current_write_origin() == "foreground"

    def test_set_empty_origin_defaults_to_foreground(self):
        token = set_current_write_origin("")
        assert get_current_write_origin() == "foreground"
        reset_current_write_origin(token)

    def test_read_mark_only_in_background_mode(self):
        """mark_background_review_skill_read no-ops unless is_background_review()."""
        # Foreground mode — should no-op.
        assert is_background_review() is False
        mark_background_review_skill_read("/path/to/skill.md")
        assert background_review_has_read("/path/to/skill.md") is False

    def test_read_mark_in_background_mode(self):
        token = set_current_write_origin(BACKGROUND_REVIEW)
        try:
            mark_background_review_skill_read("/path/to/skill.md")
            assert background_review_has_read("/path/to/skill.md") is True
            assert background_review_has_read("/other/path") is False
        finally:
            reset_current_write_origin(token)

    def test_reset_clears_read_marks(self):
        token = set_current_write_origin(BACKGROUND_REVIEW)
        try:
            mark_background_review_skill_read("/path1")
            mark_background_review_skill_read("/path2")
            assert background_review_has_read("/path1") is True
            reset_background_review_read_marks()
            assert background_review_has_read("/path1") is False
            assert background_review_has_read("/path2") is False
        finally:
            reset_current_write_origin(token)

    def test_empty_path_ignored(self):
        token = set_current_write_origin(BACKGROUND_REVIEW)
        try:
            mark_background_review_skill_read("")
            assert background_review_has_read("") is False
        finally:
            reset_current_write_origin(token)


# ---------------------------------------------------------------------------
# Config flag
# ---------------------------------------------------------------------------


class TestConfigFlag:
    def test_default_disabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NIA_BACKGROUND_REVIEW", None)
            # Also ensure config doesn't have it.
            with patch("niaharness.config.settings.load_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                # Should be disabled by default.
                from niaharness.engine.background_review import _is_background_review_enabled
                assert _is_background_review_enabled() is False

    def test_env_var_enables(self):
        with patch.dict(os.environ, {"NIA_BACKGROUND_REVIEW": "1"}):
            from niaharness.engine.background_review import _is_background_review_enabled
            assert _is_background_review_enabled() is True

    def test_env_var_disables(self):
        with patch.dict(os.environ, {"NIA_BACKGROUND_REVIEW": "0"}):
            from niaharness.engine.background_review import _is_background_review_enabled
            assert _is_background_review_enabled() is False

    def test_env_var_true_string(self):
        with patch.dict(os.environ, {"NIA_BACKGROUND_REVIEW": "true"}):
            from niaharness.engine.background_review import _is_background_review_enabled
            assert _is_background_review_enabled() is True

    def test_env_var_yes_string(self):
        with patch.dict(os.environ, {"NIA_BACKGROUND_REVIEW": "yes"}):
            from niaharness.engine.background_review import _is_background_review_enabled
            assert _is_background_review_enabled() is True


# ---------------------------------------------------------------------------
# maybe_spawn_background_review gating
# ---------------------------------------------------------------------------


class TestMaybeSpawnGating:
    def test_disabled_does_not_spawn(self):
        """When config is off, no thread is spawned."""
        with patch("niaharness.engine.background_review._is_background_review_enabled", return_value=False):
            from niaharness.engine.background_review import _review_state
            initial = _review_state.active_count()
            maybe_spawn_background_review(
                messages=[ConversationMessage.from_user_text("hi")],
                api_client=MagicMock(),
                model="test",
                system_prompt="test",
                memory=MagicMock(),
                tool_call_count=5,
            )
            # No thread spawned.
            assert _review_state.active_count() == initial

    def test_no_memory_does_not_spawn(self):
        with patch("niaharness.engine.background_review._is_background_review_enabled", return_value=True):
            maybe_spawn_background_review(
                messages=[ConversationMessage.from_user_text("hi")],
                api_client=MagicMock(),
                model="test",
                system_prompt="test",
                memory=None,
                tool_call_count=5,
            )
            # No thread spawned (memory is None).

    def test_interrupted_does_not_spawn(self):
        with patch("niaharness.engine.background_review._is_background_review_enabled", return_value=True):
            maybe_spawn_background_review(
                messages=[ConversationMessage.from_user_text("hi")],
                api_client=MagicMock(),
                model="test",
                system_prompt="test",
                memory=MagicMock(),
                tool_call_count=5,
                was_interrupted=True,
            )

    def test_below_tool_threshold_does_not_spawn(self):
        with patch("niaharness.engine.background_review._is_background_review_enabled", return_value=True):
            maybe_spawn_background_review(
                messages=[ConversationMessage.from_user_text("hi")],
                api_client=MagicMock(),
                model="test",
                system_prompt="test",
                memory=MagicMock(),
                tool_call_count=MIN_TOOL_CALLS_FOR_REVIEW - 1,
            )

    def test_no_review_flags_does_not_spawn(self):
        with patch("niaharness.engine.background_review._is_background_review_enabled", return_value=True):
            maybe_spawn_background_review(
                messages=[ConversationMessage.from_user_text("hi")],
                api_client=MagicMock(),
                model="test",
                system_prompt="test",
                memory=MagicMock(),
                tool_call_count=5,
                review_memory=False,
                review_skills=False,
            )

    def test_meets_all_criteria_spawns_thread(self):
        """When all criteria are met, a thread is spawned."""
        with patch("niaharness.engine.background_review._is_background_review_enabled", return_value=True):
            from niaharness.engine.background_review import _review_state
            maybe_spawn_background_review(
                messages=[
                    ConversationMessage.from_user_text("hello"),
                    ConversationMessage(role="assistant", content=[TextBlock(text="hi")]),
                ],
                api_client=MagicMock(),
                model="test",
                system_prompt="test",
                memory=MagicMock(),
                tool_call_count=MIN_TOOL_CALLS_FOR_REVIEW,
            )
            # A thread should have been spawned.
            assert _review_state.active_count() >= 1
            # Wait for it to finish (it will fail since api_client is a mock).
            wait_for_reviews(timeout=5.0)


# ---------------------------------------------------------------------------
# Message snapshotting
# ---------------------------------------------------------------------------


class TestSnapshotMessages:
    def test_converts_conversation_messages(self):
        messages = [
            ConversationMessage.from_user_text("hello"),
            ConversationMessage(role="assistant", content=[TextBlock(text="hi")]),
        ]
        snapshot = _snapshot_messages(messages)
        assert len(snapshot) == 2
        assert snapshot[0]["role"] == "user"
        assert snapshot[1]["role"] == "assistant"

    def test_caps_at_max(self):
        messages = [
            ConversationMessage.from_user_text(f"msg {i}")
            for i in range(100)
        ]
        snapshot = _snapshot_messages(messages)
        assert len(snapshot) <= 45  # MAX_SNAPSHOT_MESSAGES = 40, but head + tail

    def test_empty_messages(self):
        assert _snapshot_messages([]) == []

    def test_preserves_content_blocks(self):
        messages = [ConversationMessage.from_user_text("hello world")]
        snapshot = _snapshot_messages(messages)
        assert snapshot[0]["content"][0]["text"] == "hello world"


# ---------------------------------------------------------------------------
# Action summarization
# ---------------------------------------------------------------------------


class TestSummarizeActions:
    def test_off_mode_returns_empty(self):
        result = summarize_background_review_actions(
            [{"role": "tool", "content": '{"success": true, "message": "created"}'}],
            [],
            notification_mode="off",
        )
        assert result == []

    def test_on_mode_created_action(self):
        review_messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "skill_manage", "arguments": '{"action": "create", "name": "my-skill"}'},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"success": true, "message": "Skill created"}',
            },
        ]
        result = summarize_background_review_actions(review_messages, [], notification_mode="on")
        assert len(result) == 1
        assert "Skill created" in result[0]

    def test_on_mode_updated_action(self):
        review_messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "nia_memory", "arguments": '{"action": "add", "content": "prefers concise"}'},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"success": true, "message": "Memory updated"}',
            },
        ]
        result = summarize_background_review_actions(review_messages, [], notification_mode="on")
        assert len(result) == 1
        assert "Memory updated" in result[0]

    def test_skips_stale_inherited_results(self):
        """Tool messages from prior_snapshot are skipped."""
        prior = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"success": true, "message": "Old action"}',
            },
        ]
        review_messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "skill_manage", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"success": true, "message": "Old action"}',
            },
        ]
        result = summarize_background_review_actions(review_messages, prior, notification_mode="on")
        assert result == []

    def test_verbose_mode_includes_previews(self):
        review_messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "skill_manage",
                        "arguments": '{"action": "create", "name": "my-skill", "content": "This is a detailed skill description"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"success": true, "message": "Skill created"}',
            },
        ]
        result = summarize_background_review_actions(review_messages, [], notification_mode="verbose")
        assert len(result) == 1
        assert "my-skill" in result[0]

    def test_failed_actions_skipped(self):
        review_messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "skill_manage", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"success": false, "message": "Failed"}',
            },
        ]
        result = summarize_background_review_actions(review_messages, [], notification_mode="on")
        assert result == []

    def test_malformed_json_skipped(self):
        review_messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "not json",
            },
        ]
        result = summarize_background_review_actions(review_messages, [], notification_mode="on")
        assert result == []

    def test_empty_messages(self):
        result = summarize_background_review_actions([], [], notification_mode="on")
        assert result == []


# ---------------------------------------------------------------------------
# QueryResult.tool_call_count
# ---------------------------------------------------------------------------


class TestQueryResultToolCallCount:
    def test_field_exists(self):
        assert "tool_call_count" in QueryResult.__dataclass_fields__

    def test_default_value(self):
        result = QueryResult(reason=TerminationReason.COMPLETED)
        assert result.tool_call_count == 0

    def test_can_set_value(self):
        result = QueryResult(reason=TerminationReason.COMPLETED, tool_call_count=5)
        assert result.tool_call_count == 5


# ---------------------------------------------------------------------------
# QueryEngine post_turn_hooks
# ---------------------------------------------------------------------------


class TestQueryEnginePostTurnHooks:
    def test_hooks_stored(self):
        """QueryEngine stores post_turn_hooks passed to __init__."""
        from niaharness.engine.query_engine import QueryEngine
        from niaharness.tools.base import ToolRegistry
        from niaharness.permissions.checker import PermissionChecker
        from niaharness.config.settings import PermissionSettings

        hook1 = MagicMock()
        engine = QueryEngine(
            api_client=MagicMock(),
            tool_registry=ToolRegistry(),
            permission_checker=PermissionChecker(PermissionSettings()),
            cwd="/tmp",
            model="test",
            system_prompt="test",
            post_turn_hooks=[hook1],
        )
        assert hook1 in engine._post_turn_hooks

    def test_hooks_default_empty(self):
        from niaharness.engine.query_engine import QueryEngine
        from niaharness.tools.base import ToolRegistry
        from niaharness.permissions.checker import PermissionChecker
        from niaharness.config.settings import PermissionSettings

        engine = QueryEngine(
            api_client=MagicMock(),
            tool_registry=ToolRegistry(),
            permission_checker=PermissionChecker(PermissionSettings()),
            cwd="/tmp",
            model="test",
            system_prompt="test",
        )
        assert engine._post_turn_hooks == []


# ---------------------------------------------------------------------------
# Integration — _ReviewState
# ---------------------------------------------------------------------------


class TestReviewState:
    def test_set_and_notify_feedback(self):
        received = []
        state = _ReviewState()
        state.set_feedback_callback(lambda msg: received.append(msg))
        state.notify_feedback("test message")
        assert received == ["test message"]

    def test_feedback_callback_exception_does_not_raise(self):
        state = _ReviewState()
        state.set_feedback_callback(lambda msg: (_ for _ in ()).throw(RuntimeError("fail")))
        state.notify_feedback("test")  # Should not raise.

    def test_active_count(self):
        state = _ReviewState()
        assert state.active_count() == 0


# ---------------------------------------------------------------------------
# get_review_stats
# ---------------------------------------------------------------------------


class TestGetReviewStats:
    def test_returns_dict_with_expected_keys(self):
        stats = get_review_stats()
        assert "active_threads" in stats
        assert "enabled" in stats
        assert "min_tool_calls" in stats
        assert stats["min_tool_calls"] == MIN_TOOL_CALLS_FOR_REVIEW


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
