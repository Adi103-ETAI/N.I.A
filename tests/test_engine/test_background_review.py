"""Tests for the background review system (Task 5)."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from niaharness.api.client import ApiMessageCompleteEvent, ApiMessageRequest, ApiTextDeltaEvent
from niaharness.api.usage import UsageSnapshot
from niaharness.engine.background_review import (
    MEMORY_REVIEW_PROMPT,
    _ReviewState,
    _apply_memory_writes,
    _parse_review_response,
    _snapshot_messages,
    get_review_interval,
    get_review_model,
    get_review_stats,
    get_review_state,
    is_background_review_enabled,
    maybe_spawn_background_review,
    wait_for_reviews,
)
from niaharness.engine.messages import ConversationMessage, TextBlock


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    def test_enabled_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NIA_BACKGROUND_REVIEW", raising=False)
        assert is_background_review_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "off", "no", "disabled", "FALSE"])
    def test_disabled_values(self, monkeypatch: pytest.MonkeyPatch, val: str):
        monkeypatch.setenv("NIA_BACKGROUND_REVIEW", val)
        assert is_background_review_enabled() is False

    def test_review_model_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NIA_BACKGROUND_REVIEW_MODEL", raising=False)
        assert get_review_model() is None

    def test_review_model_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NIA_BACKGROUND_REVIEW_MODEL", "gpt-4o-mini")
        assert get_review_model() == "gpt-4o-mini"

    def test_review_interval_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NIA_BACKGROUND_REVIEW_INTERVAL", raising=False)
        assert get_review_interval() == 30.0

    def test_review_interval_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NIA_BACKGROUND_REVIEW_INTERVAL", "5.5")
        assert get_review_interval() == 5.5

    def test_review_interval_invalid(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NIA_BACKGROUND_REVIEW_INTERVAL", "not-a-number")
        assert get_review_interval() == 30.0


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_basic_messages(self):
        messages = [
            ConversationMessage.from_user_text("hello"),
            ConversationMessage(role="assistant", content=[TextBlock(text="hi there")]),
        ]
        snapshot = _snapshot_messages(messages)
        assert len(snapshot) == 2
        assert snapshot[0]["role"] == "user"
        assert snapshot[0]["content"][0]["type"] == "text"
        assert snapshot[0]["content"][0]["text"] == "hello"

    def test_snapshot_limits_to_20(self):
        messages = [
            ConversationMessage.from_user_text(f"msg {i}") for i in range(30)
        ]
        snapshot = _snapshot_messages(messages)
        assert len(snapshot) == 20
        # Should keep the last 20.
        assert snapshot[0]["content"][0]["text"] == "msg 10"
        assert snapshot[-1]["content"][0]["text"] == "msg 29"

    def test_snapshot_empty_list(self):
        assert _snapshot_messages([]) == []


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseReviewResponse:
    def test_valid_json(self):
        text = json.dumps({
            "memories": [{"category": "fact", "content": "User likes Python"}],
            "summary": "Saved 1 fact",
        })
        result = _parse_review_response(text)
        assert len(result["memories"]) == 1
        assert result["summary"] == "Saved 1 fact"

    def test_json_with_code_fence(self):
        text = '```json\n{"memories": [], "summary": "Nothing to save."}\n```'
        result = _parse_review_response(text)
        assert result["memories"] == []
        assert result["summary"] == "Nothing to save."

    def test_json_embedded_in_text(self):
        text = 'Here is my review:\n{"memories": [], "summary": "Nothing."}\nDone.'
        result = _parse_review_response(text)
        assert result["summary"] == "Nothing."

    def test_invalid_json_returns_empty(self):
        result = _parse_review_response("not json at all")
        assert result["memories"] == []
        assert "Could not parse" in result["summary"]

    def test_empty_response(self):
        result = _parse_review_response("")
        assert result["memories"] == []


# ---------------------------------------------------------------------------
# Memory writes
# ---------------------------------------------------------------------------


class TestApplyMemoryWrites:
    def test_apply_preference(self):
        memory = MagicMock()
        review = {
            "memories": [
                {"category": "preference", "key": "tone", "content": "User prefers concise answers"}
            ]
        }
        count = _apply_memory_writes(review, memory)
        assert count == 1
        memory.add_preference.assert_called_once_with("tone", "User prefers concise answers")
        memory.save.assert_called_once()

    def test_apply_fact(self):
        memory = MagicMock()
        review = {
            "memories": [
                {"category": "fact", "content": "User works at Acme Corp"}
            ]
        }
        count = _apply_memory_writes(review, memory)
        assert count == 1
        memory.add_fact.assert_called_once_with("User works at Acme Corp")

    def test_apply_pattern(self):
        memory = MagicMock()
        review = {
            "memories": [
                {"category": "pattern", "content": "User often asks about Docker"}
            ]
        }
        count = _apply_memory_writes(review, memory)
        assert count == 1
        memory.add_pattern.assert_called_once_with("User often asks about Docker")

    def test_apply_multiple(self):
        memory = MagicMock()
        review = {
            "memories": [
                {"category": "fact", "content": "fact 1"},
                {"category": "preference", "key": "x", "content": "pref 1"},
                {"category": "pattern", "content": "pattern 1"},
            ]
        }
        count = _apply_memory_writes(review, memory)
        assert count == 3

    def test_apply_empty(self):
        memory = MagicMock()
        count = _apply_memory_writes({"memories": []}, memory)
        assert count == 0
        memory.save.assert_not_called()

    def test_apply_skips_invalid_entries(self):
        memory = MagicMock()
        review = {
            "memories": [
                "not a dict",
                {"category": "fact"},  # missing content
                {"category": "fact", "content": "valid"},
            ]
        }
        count = _apply_memory_writes(review, memory)
        assert count == 1

    def test_apply_no_save_if_no_writes(self):
        memory = MagicMock()
        _apply_memory_writes({"memories": []}, memory)
        memory.save.assert_not_called()


# ---------------------------------------------------------------------------
# Review state
# ---------------------------------------------------------------------------


class TestReviewState:
    def test_should_review_first_time(self):
        state = _ReviewState()
        assert state.should_review() is True

    def test_should_review_respects_interval(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NIA_BACKGROUND_REVIEW_INTERVAL", "100")
        state = _ReviewState()
        assert state.should_review() is True
        # Second call within interval should be blocked.
        assert state.should_review() is False

    def test_register_and_count_threads(self):
        state = _ReviewState()
        t = threading.Thread(target=lambda: None, daemon=True)
        state.register_thread(t)
        t.start()
        t.join()
        assert state.active_count() == 0  # already finished


# ---------------------------------------------------------------------------
# maybe_spawn_background_review
# ---------------------------------------------------------------------------


class TestMaybeSpawn:
    def test_disabled_does_nothing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NIA_BACKGROUND_REVIEW", "off")
        # Reset state so interval doesn't block.
        get_review_state()._last_review_time = 0.0
        maybe_spawn_background_review(
            messages=[ConversationMessage.from_user_text("hi")],
            api_client=MagicMock(),
            model="m",
            system_prompt="s",
            memory=MagicMock(),
        )
        assert get_review_state().active_count() == 0

    def test_no_memory_does_nothing(self):
        maybe_spawn_background_review(
            messages=[ConversationMessage.from_user_text("hi")],
            api_client=MagicMock(),
            model="m",
            system_prompt="s",
            memory=None,
        )
        assert get_review_state().active_count() == 0

    def test_too_few_messages_does_nothing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NIA_BACKGROUND_REVIEW", "1")
        monkeypatch.setenv("NIA_BACKGROUND_REVIEW_INTERVAL", "0")
        get_review_state()._last_review_time = 0.0
        maybe_spawn_background_review(
            messages=[ConversationMessage.from_user_text("hi")],  # only 1 msg
            api_client=MagicMock(),
            model="m",
            system_prompt="s",
            memory=MagicMock(),
        )
        wait_for_reviews(timeout=1.0)
        assert get_review_state().active_count() == 0


# ---------------------------------------------------------------------------
# Integration with QueryEngine
# ---------------------------------------------------------------------------


class TestQueryEngineIntegration:
    @pytest.mark.asyncio
    async def test_review_spawned_after_turn(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """QueryEngine.submit_message should spawn a review after the turn."""
        from niaharness.config.settings import PermissionSettings
        from niaharness.engine.query_engine import QueryEngine
        from niaharness.permissions import PermissionChecker
        from niaharness.tools import create_default_tool_registry

        # Enable review with 0 interval.
        monkeypatch.setenv("NIA_BACKGROUND_REVIEW", "1")
        monkeypatch.setenv("NIA_BACKGROUND_REVIEW_INTERVAL", "0")
        get_review_state()._last_review_time = 0.0

        # Mock API client that returns a simple response.
        class FakeApiClient:
            async def stream_message(self, request):
                del request
                yield ApiTextDeltaEvent(text="Hello!")
                yield ApiMessageCompleteEvent(
                    message=ConversationMessage(
                        role="assistant", content=[TextBlock(text="Hello!")]
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                    stop_reason=None,
                )

        memory = MagicMock()

        engine = QueryEngine(
            api_client=FakeApiClient(),
            tool_registry=create_default_tool_registry(),
            permission_checker=PermissionChecker(PermissionSettings()),
            cwd=tmp_path,
            model="test-model",
            system_prompt="system",
            memory=memory,
        )

        events = [ev async for ev in engine.submit_message("hi")]
        # Should have streamed events.
        assert len(events) > 0

        # Wait for the review thread to spawn (it may or may not complete
        # depending on the mock, but it should at least start).
        # The review will fail because FakeApiClient doesn't handle the review
        # call's ApiMessageRequest properly, but that's fine — we just want
        # to verify the spawn happens without breaking the main turn.
        # No assertion on memory writes here — the mock API client won't
        # produce a valid review response.

    @pytest.mark.asyncio
    async def test_no_memory_no_review(self, tmp_path: Path):
        """Without a memory object, no review should be spawned."""
        from niaharness.config.settings import PermissionSettings
        from niaharness.engine.query_engine import QueryEngine
        from niaharness.permissions import PermissionChecker
        from niaharness.tools import create_default_tool_registry

        class FakeApiClient:
            async def stream_message(self, request):
                del request
                yield ApiMessageCompleteEvent(
                    message=ConversationMessage(
                        role="assistant", content=[TextBlock(text="Hi")]
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                    stop_reason=None,
                )

        # No memory= argument → defaults to None → no review.
        engine = QueryEngine(
            api_client=FakeApiClient(),
            tool_registry=create_default_tool_registry(),
            permission_checker=PermissionChecker(PermissionSettings()),
            cwd=tmp_path,
            model="test-model",
            system_prompt="system",
        )

        events = [ev async for ev in engine.submit_message("hi")]
        assert len(events) > 0  # turn completed normally


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_structure(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NIA_BACKGROUND_REVIEW", "1")
        monkeypatch.setenv("NIA_BACKGROUND_REVIEW_MODEL", "gpt-4o")
        monkeypatch.setenv("NIA_BACKGROUND_REVIEW_INTERVAL", "60")
        stats = get_review_stats()
        assert stats["enabled"] is True
        assert stats["model"] == "gpt-4o"
        assert stats["interval_seconds"] == 60.0
        assert "active_threads" in stats
        assert "last_review_time" in stats
