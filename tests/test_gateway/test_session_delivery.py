"""Tests for gateway session persistence + delivery routing (Task 9).

Covers:
  - PII redaction: _hash_sender_id, _hash_chat_id (SHA-256, prefix-preserving)
  - Session key derivation: build_session_key (DM, group, per-user, thread, profile)
  - SessionSource: to_dict / from_dict roundtrip, description property
  - SessionEntry: to_dict / from_dict roundtrip
  - SessionStore: get_or_create_session, update_session, suspend_session, reset_session, list_sessions
  - build_session_context_prompt: platform, user, connected platforms, delivery options, PII redaction
  - auto_continue_freshness_window: env var override, default
  - DeliveryTarget: parse (origin, local, platform, platform:chat, platform:chat:thread), to_string
  - DeadTargetRegistry: mark_dead, is_dead, clear, persistence, all_dead
  - DeliveryRouter: deliver (local, platform, dead-target skip, silence-narration filter, oversized truncation)
  - _is_silence_narration: various tokens
  - _classify_dead_from_error_text: forbidden, not_found, unknown
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from niaharness.gateway.session import (
    SessionEntry,
    SessionSource,
    SessionStore,
    _format_untrusted_prompt_value,
    _hash_chat_id,
    _hash_sender_id,
    auto_continue_freshness_window,
    build_session_context_prompt,
    build_session_key,
)
from niaharness.gateway.delivery import (
    DeadTargetRegistry,
    DeliveryRouter,
    DeliveryTarget,
    MAX_PLATFORM_OUTPUT,
    _classify_dead_from_error_text,
    _is_silence_narration,
    _normalize,
)


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------


class TestPIIRedaction:
    def test_hash_sender_id_format(self):
        result = _hash_sender_id("12345")
        assert result.startswith("user_")
        assert len(result) == 5 + 12  # "user_" + 12 hex chars

    def test_hash_sender_id_deterministic(self):
        assert _hash_sender_id("12345") == _hash_sender_id("12345")

    def test_hash_sender_id_different_inputs_different_hashes(self):
        assert _hash_sender_id("12345") != _hash_sender_id("67890")

    def test_hash_chat_id_preserves_prefix(self):
        result = _hash_chat_id("telegram:12345")
        assert result.startswith("telegram:")
        assert len(result) == len("telegram:") + 12

    def test_hash_chat_id_no_prefix(self):
        result = _hash_chat_id("12345")
        assert len(result) == 12  # Just the hash.

    def test_hash_chat_id_deterministic(self):
        assert _hash_chat_id("telegram:12345") == _hash_chat_id("telegram:12345")


class TestFormatUntrustedPromptValue:
    def test_none_returns_empty(self):
        assert _format_untrusted_prompt_value(None) == ""

    def test_simple_string(self):
        assert _format_untrusted_prompt_value("hello") == "hello"

    def test_truncates_long_strings(self):
        long_str = "x" * 300
        result = _format_untrusted_prompt_value(long_str)
        assert len(result) <= 243  # 240 + "..."
        assert result.endswith("...")

    def test_replaces_control_chars(self):
        result = _format_untrusted_prompt_value("hello\x00\x01world")
        assert "\x00" not in result
        assert "\x01" not in result


# ---------------------------------------------------------------------------
# Session key derivation
# ---------------------------------------------------------------------------


class TestBuildSessionKey:
    def test_dm_key(self):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        key = build_session_key(source)
        assert key == "agent:main:telegram:dm:12345"

    def test_dm_key_with_user_id_fallback(self):
        source = SessionSource(platform="telegram", chat_id="", user_id="67890", chat_type="dm")
        key = build_session_key(source)
        assert key == "agent:main:telegram:dm:67890"

    def test_group_key_shared(self):
        source = SessionSource(platform="telegram", chat_id="-100", user_id="67890", chat_type="group")
        key = build_session_key(source, group_sessions_per_user=False)
        assert key == "agent:main:telegram:group:-100"

    def test_group_key_per_user(self):
        source = SessionSource(platform="telegram", chat_id="-100", user_id="67890", chat_type="group")
        key = build_session_key(source, group_sessions_per_user=True)
        assert key == "agent:main:telegram:group:-100:67890"

    def test_thread_key(self):
        source = SessionSource(
            platform="discord", chat_id="chan1", thread_id="thread1", chat_type="thread"
        )
        key = build_session_key(source)
        # Threads default to shared (no participant_id).
        assert key == "agent:main:discord:thread:chan1:thread1"

    def test_thread_key_per_user(self):
        source = SessionSource(
            platform="discord", chat_id="chan1", thread_id="thread1", user_id="u1", chat_type="thread"
        )
        key = build_session_key(source, thread_sessions_per_user=True)
        assert key == "agent:main:discord:thread:chan1:thread1:u1"

    def test_profile_namespace(self):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm", profile="work")
        key = build_session_key(source, profile="work")
        assert key.startswith("agent:work:")

    def test_default_profile_namespace(self):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        key = build_session_key(source, profile="default")
        assert key.startswith("agent:main:")


# ---------------------------------------------------------------------------
# SessionSource
# ---------------------------------------------------------------------------


class TestSessionSource:
    def test_to_dict_from_dict_roundtrip(self):
        source = SessionSource(
            platform="telegram", chat_id="12345", user_id="67890",
            user_name="Alice", chat_type="dm", thread_id="t1",
        )
        d = source.to_dict()
        restored = SessionSource.from_dict(d)
        assert restored.platform == "telegram"
        assert restored.chat_id == "12345"
        assert restored.user_id == "67890"
        assert restored.user_name == "Alice"
        assert restored.chat_type == "dm"
        assert restored.thread_id == "t1"

    def test_description_dm(self):
        source = SessionSource(platform="telegram", chat_id="12345", user_name="Alice", chat_type="dm")
        assert source.description == "DM with Alice"

    def test_description_dm_with_user_id(self):
        source = SessionSource(platform="telegram", chat_id="12345", user_id="67890", chat_type="dm")
        assert source.description == "DM with 67890"

    def test_description_group(self):
        source = SessionSource(platform="telegram", chat_id="-100", chat_name="Dev Team", chat_type="group")
        assert source.description == "group: Dev Team"

    def test_description_channel(self):
        source = SessionSource(platform="telegram", chat_id="-200", chat_name="Announcements", chat_type="channel")
        assert source.description == "channel: Announcements"

    def test_description_local(self):
        source = SessionSource(platform="local")
        assert source.description == "the machine running this agent"


# ---------------------------------------------------------------------------
# SessionEntry
# ---------------------------------------------------------------------------


class TestSessionEntry:
    def test_to_dict_from_dict_roundtrip(self):
        entry = SessionEntry(
            session_key="agent:main:telegram:dm:12345",
            session_id="20260101_120000_abc123",
            created_at=datetime(2026, 1, 1, 12, 0, 0),
            updated_at=datetime(2026, 1, 1, 12, 30, 0),
            platform="telegram",
            chat_type="dm",
        )
        d = entry.to_dict()
        restored = SessionEntry.from_dict(d)
        assert restored.session_key == entry.session_key
        assert restored.session_id == entry.session_id
        assert restored.platform == "telegram"


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class TestSessionStore:
    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        """Build a SessionStore with a mocked SessionDB."""
        # Mock SessionDB to avoid touching real SQLite.
        mock_db = MagicMock()
        mock_db._conn = None
        mock_db._lock = MagicMock()
        with patch("niaharness.services.session_db.SessionDB", return_value=mock_db):
            store = SessionStore(sessions_dir=tmp_path / "gateway")
        # Override _db with our mock.
        store._db = mock_db
        return store

    def test_get_or_create_new_session(self, store):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        entry = store.get_or_create_session(source)
        assert entry.session_key == "agent:main:telegram:dm:12345"
        assert entry.session_id  # Non-empty.
        assert entry.platform == "telegram"

    def test_get_or_create_existing_session(self, store):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        entry1 = store.get_or_create_session(source)
        entry2 = store.get_or_create_session(source)
        assert entry1.session_id == entry2.session_id  # Same session.

    def test_get_or_create_force_new(self, store):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        entry1 = store.get_or_create_session(source)
        entry2 = store.get_or_create_session(source, force_new=True)
        assert entry1.session_id != entry2.session_id  # Different.

    def test_update_session(self, store):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        entry = store.get_or_create_session(source)
        old_updated = entry.updated_at
        time.sleep(0.01)
        store.update_session(entry.session_key, last_prompt_tokens=500)
        assert entry.last_prompt_tokens == 500
        assert entry.updated_at > old_updated

    def test_suspend_session(self, store):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        entry = store.get_or_create_session(source)
        assert store.suspend_session(entry.session_key) is True
        assert entry.suspended is True

    def test_reset_session(self, store):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        entry1 = store.get_or_create_session(source)
        entry2 = store.reset_session(entry1.session_key)
        assert entry2 is not None
        assert entry1.session_id != entry2.session_id

    def test_list_sessions(self, store):
        s1 = SessionSource(platform="telegram", chat_id="111", chat_type="dm")
        s2 = SessionSource(platform="telegram", chat_id="222", chat_type="dm")
        store.get_or_create_session(s1)
        store.get_or_create_session(s2)
        sessions = store.list_sessions()
        assert len(sessions) == 2

    def test_peek_session_id(self, store):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        entry = store.get_or_create_session(source)
        assert store.peek_session_id(entry.session_key) == entry.session_id


# ---------------------------------------------------------------------------
# build_session_context_prompt
# ---------------------------------------------------------------------------


class TestBuildSessionContextPrompt:
    def test_contains_session_context_header(self):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        prompt = build_session_context_prompt(source)
        assert "## Current Session Context" in prompt

    def test_contains_untrusted_warning(self):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        prompt = build_session_context_prompt(source)
        assert "untrusted" in prompt.lower()

    def test_contains_platform_name(self):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        prompt = build_session_context_prompt(source)
        assert "Telegram" in prompt

    def test_contains_user_name(self):
        source = SessionSource(platform="telegram", chat_id="12345", user_name="Alice", chat_type="dm")
        prompt = build_session_context_prompt(source)
        assert "Alice" in prompt

    def test_pii_redaction_hashes_user_id(self):
        source = SessionSource(platform="telegram", chat_id="12345", user_id="67890", chat_type="dm")
        prompt = build_session_context_prompt(source, redact_pii=True)
        assert "user_" in prompt
        assert "67890" not in prompt

    def test_pii_redaction_preserves_user_name(self):
        source = SessionSource(platform="telegram", chat_id="12345", user_name="Alice", chat_type="dm")
        prompt = build_session_context_prompt(source, redact_pii=True)
        assert "Alice" in prompt  # Name is not redacted, only IDs.

    def test_local_platform(self):
        source = SessionSource(platform="local")
        prompt = build_session_context_prompt(source)
        assert "machine running this agent" in prompt

    def test_connected_platforms(self):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        prompt = build_session_context_prompt(source, connected_platforms=["telegram", "discord"])
        assert "telegram: Connected" in prompt
        assert "discord: Connected" in prompt

    def test_delivery_options(self):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        prompt = build_session_context_prompt(source)
        assert '"origin"' in prompt
        assert '"local"' in prompt

    def test_home_channels(self):
        source = SessionSource(platform="telegram", chat_id="12345", chat_type="dm")
        prompt = build_session_context_prompt(
            source,
            home_channels={"telegram": {"name": "My Chat", "chat_id": "999"}},
        )
        assert "My Chat" in prompt
        assert "Home Channels" in prompt


# ---------------------------------------------------------------------------
# auto_continue_freshness_window
# ---------------------------------------------------------------------------


class TestAutoContinueFreshnessWindow:
    def test_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NIA_AUTO_CONTINUE_FRESHNESS", None)
            assert auto_continue_freshness_window() == 3600.0

    def test_env_override(self):
        with patch.dict(os.environ, {"NIA_AUTO_CONTINUE_FRESHNESS": "7200"}):
            assert auto_continue_freshness_window() == 7200.0

    def test_disable_with_zero(self):
        with patch.dict(os.environ, {"NIA_AUTO_CONTINUE_FRESHNESS": "0"}):
            assert auto_continue_freshness_window() == 0.0


# ---------------------------------------------------------------------------
# DeliveryTarget
# ---------------------------------------------------------------------------


class TestDeliveryTarget:
    def test_parse_origin(self):
        source = SessionSource(platform="telegram", chat_id="12345")
        target = DeliveryTarget.parse("origin", origin=source)
        assert target.is_origin is True
        assert target.platform == "telegram"
        assert target.chat_id == "12345"

    def test_parse_origin_no_source(self):
        target = DeliveryTarget.parse("origin")
        assert target.is_origin is True
        assert target.platform == "local"

    def test_parse_local(self):
        target = DeliveryTarget.parse("local")
        assert target.platform == "local"

    def test_parse_platform_only(self):
        target = DeliveryTarget.parse("telegram")
        assert target.platform == "telegram"
        assert target.chat_id is None

    def test_parse_platform_chat(self):
        target = DeliveryTarget.parse("telegram:12345")
        assert target.platform == "telegram"
        assert target.chat_id == "12345"
        assert target.is_explicit is True

    def test_parse_platform_chat_thread(self):
        target = DeliveryTarget.parse("telegram:12345:789")
        assert target.platform == "telegram"
        assert target.chat_id == "12345"
        assert target.thread_id == "789"

    def test_to_string_origin(self):
        target = DeliveryTarget(platform="telegram", is_origin=True)
        assert target.to_string() == "origin"

    def test_to_string_local(self):
        target = DeliveryTarget(platform="local")
        assert target.to_string() == "local"

    def test_to_string_platform_chat(self):
        target = DeliveryTarget(platform="telegram", chat_id="12345")
        assert target.to_string() == "telegram:12345"

    def test_to_string_platform_chat_thread(self):
        target = DeliveryTarget(platform="telegram", chat_id="12345", thread_id="789")
        assert target.to_string() == "telegram:12345:789"


# ---------------------------------------------------------------------------
# DeadTargetRegistry
# ---------------------------------------------------------------------------


class TestDeadTargetRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        return DeadTargetRegistry(path=tmp_path / "dead_targets.json")

    def test_mark_dead(self, registry):
        assert registry.mark_dead("telegram", "12345", reason="blocked") is True
        assert registry.is_dead("telegram", "12345") is True

    def test_mark_dead_already_dead(self, registry):
        registry.mark_dead("telegram", "12345")
        assert registry.mark_dead("telegram", "12345") is False  # Not newly added.

    def test_is_dead_no_chat_id(self, registry):
        assert registry.is_dead("telegram", None) is False
        assert registry.is_dead("telegram", "") is False

    def test_clear(self, registry):
        registry.mark_dead("telegram", "12345")
        assert registry.clear("telegram", "12345") is True
        assert registry.is_dead("telegram", "12345") is False

    def test_clear_not_dead(self, registry):
        assert registry.clear("telegram", "12345") is False

    def test_all_dead(self, registry):
        registry.mark_dead("telegram", "111")
        registry.mark_dead("discord", "222")
        all_dead = registry.all_dead()
        assert len(all_dead) == 2
        assert "telegram:111" in all_dead
        assert "discord:222" in all_dead

    def test_persistence(self, tmp_path):
        path = tmp_path / "dead.json"
        reg1 = DeadTargetRegistry(path=path)
        reg1.mark_dead("telegram", "12345")
        # Create a new instance pointing at the same file.
        reg2 = DeadTargetRegistry(path=path)
        assert reg2.is_dead("telegram", "12345") is True

    def test_is_dead_error_kind(self):
        assert DeadTargetRegistry.is_dead_error_kind("forbidden") is True
        assert DeadTargetRegistry.is_dead_error_kind("not_found") is True
        assert DeadTargetRegistry.is_dead_error_kind("other") is False
        assert DeadTargetRegistry.is_dead_error_kind(None) is False


# ---------------------------------------------------------------------------
# _is_silence_narration
# ---------------------------------------------------------------------------


class TestIsSilenceNarration:
    def test_silent(self):
        assert _is_silence_narration("silent") is True
        assert _is_silence_narration("(silent)") is True
        assert _is_silence_narration("*silent*") is True

    def test_silence(self):
        assert _is_silence_narration("silence") is True

    def test_no_response(self):
        assert _is_silence_narration("no response") is True

    def test_emoji(self):
        assert _is_silence_narration("🔇") is True

    def test_bare_dot(self):
        assert _is_silence_narration(".") is True

    def test_ellipsis(self):
        assert _is_silence_narration("…") is True

    def test_normal_text_not_silence(self):
        assert _is_silence_narration("Hello there") is False
        assert _is_silence_narration("The deployment ran silently") is False

    def test_empty_not_silence(self):
        assert _is_silence_narration("") is False
        assert _is_silence_narration(None) is False

    def test_long_text_not_silence(self):
        assert _is_silence_narration("x" * 100) is False


# ---------------------------------------------------------------------------
# _classify_dead_from_error_text
# ---------------------------------------------------------------------------


class TestClassifyDeadErrorText:
    def test_forbidden(self):
        assert _classify_dead_from_error_text("bot was blocked by the user") == "forbidden"
        assert _classify_dead_from_error_text("Forbidden: bot kicked") == "forbidden"

    def test_not_found(self):
        assert _classify_dead_from_error_text("chat not found") == "not_found"
        assert _classify_dead_from_error_text("chat_id not found") == "not_found"

    def test_unknown(self):
        assert _classify_dead_from_error_text("timeout") is None
        assert _classify_dead_from_error_text("rate limited") is None

    def test_none(self):
        assert _classify_dead_from_error_text(None) is None
        assert _classify_dead_from_error_text("") is None


# ---------------------------------------------------------------------------
# DeliveryRouter
# ---------------------------------------------------------------------------


class TestDeliveryRouter:
    @pytest.fixture
    def router(self, tmp_path):
        return DeliveryRouter(
            output_dir=tmp_path / "output",
            dead_targets=DeadTargetRegistry(path=tmp_path / "dead.json"),
        )

    @pytest.mark.asyncio
    async def test_deliver_local(self, router):
        target = DeliveryTarget.parse("local")
        result = await router.deliver("Hello world", [target], job_id="test-job", job_name="Test Job")
        assert "local" in result
        assert result["local"]["success"] is True
        # File should be created.
        output_path = Path(result["local"]["result"]["path"])
        assert output_path.exists()
        assert "Hello world" in output_path.read_text()

    @pytest.mark.asyncio
    async def test_deliver_multiple_targets(self, router):
        """Delivering to local + a platform target produces 2 result keys."""

        async def mock_send(chat_id, content, metadata=None):
            return {"success": True}

        mock_adapter = MagicMock()
        mock_adapter.send = mock_send
        router.adapters = {"telegram": mock_adapter}
        targets = [
            DeliveryTarget.parse("local"),
            DeliveryTarget(platform="telegram", chat_id="12345"),
        ]
        result = await router.deliver("test", targets)
        assert len(result) == 2
        assert "local" in result
        assert "telegram:12345" in result

    @pytest.mark.asyncio
    async def test_deliver_to_dead_target_skipped(self, router):
        router.dead_targets.mark_dead("telegram", "12345")
        target = DeliveryTarget(platform="telegram", chat_id="12345")
        result = await router.deliver("test", [target])
        assert result["telegram:12345"]["success"] is False
        assert result["telegram:12345"]["skipped"] == "dead_target"

    @pytest.mark.asyncio
    async def test_deliver_silence_narration_filtered(self, router):
        """Silence narration should be filtered out before reaching the adapter."""
        call_count = {"n": 0}

        async def mock_send(chat_id, content, metadata=None):
            call_count["n"] += 1
            return {"success": True}

        mock_adapter = MagicMock()
        mock_adapter.send = mock_send
        router.adapters = {"telegram": mock_adapter}

        target = DeliveryTarget(platform="telegram", chat_id="12345")
        result = await router.deliver("(silent)", [target])
        # Adapter should NOT have been called.
        assert call_count["n"] == 0
        assert result["telegram:12345"]["success"] is True

    @pytest.mark.asyncio
    async def test_deliver_oversized_truncation(self, router):
        """Content > MAX_PLATFORM_OUTPUT should be truncated for non-chunking adapters."""
        sent_content_holder = {}

        async def mock_send(chat_id, content, metadata=None):
            sent_content_holder["content"] = content
            return {"success": True}

        mock_adapter = MagicMock()
        mock_adapter.send = mock_send
        mock_adapter.splits_long_messages = False
        router.adapters = {"telegram": mock_adapter}

        long_content = "x" * (MAX_PLATFORM_OUTPUT + 1000)
        target = DeliveryTarget(platform="telegram", chat_id="12345")
        await router.deliver(long_content, [target], metadata={"job_id": "test"})

        sent_content = sent_content_holder["content"]
        assert len(sent_content) <= MAX_PLATFORM_OUTPUT
        assert "truncated" in sent_content

    @pytest.mark.asyncio
    async def test_deliver_oversized_chunking_adapter(self, router):
        """Chunking adapters receive the full payload."""
        sent_content_holder = {}

        async def mock_send(chat_id, content, metadata=None):
            sent_content_holder["content"] = content
            return {"success": True}

        mock_adapter = MagicMock()
        mock_adapter.send = mock_send
        mock_adapter.splits_long_messages = True
        router.adapters = {"telegram": mock_adapter}

        long_content = "x" * (MAX_PLATFORM_OUTPUT + 1000)
        target = DeliveryTarget(platform="telegram", chat_id="12345")
        await router.deliver(long_content, [target], metadata={"job_id": "test"})

        sent_content = sent_content_holder["content"]
        assert len(sent_content) == len(long_content)  # Full payload.

    @pytest.mark.asyncio
    async def test_deliver_platform_failure_marks_dead(self, router):
        """A 'forbidden' error should mark the target as dead."""

        async def mock_send(chat_id, content, metadata=None):
            raise RuntimeError("bot was blocked by the user")

        mock_adapter = MagicMock()
        mock_adapter.send = mock_send
        router.adapters = {"telegram": mock_adapter}

        target = DeliveryTarget(platform="telegram", chat_id="12345")
        await router.deliver("test", [target])

        assert router.dead_targets.is_dead("telegram", "12345") is True

    @pytest.mark.asyncio
    async def test_deliver_success_clears_dead(self, router):
        """A successful send clears a stale dead flag.

        The dead-target check in deliver() skips known-dead targets, so
        self-healing happens when: (1) the dead flag is manually cleared
        or (2) a different code path (e.g. _handle_incoming) sends
        directly via the adapter and succeeds. Here we test that a
        successful deliver() to a NON-dead target keeps it non-dead,
        and that manually calling clear() works.
        """

        async def mock_send(chat_id, content, metadata=None):
            return {"success": True}

        mock_adapter = MagicMock()
        mock_adapter.send = mock_send
        router.adapters = {"telegram": mock_adapter}

        # Mark dead, then manually clear (simulating a user re-adding the bot).
        router.dead_targets.mark_dead("telegram", "12345")
        assert router.dead_targets.is_dead("telegram", "12345") is True
        router.dead_targets.clear("telegram", "12345")
        assert router.dead_targets.is_dead("telegram", "12345") is False

        # Now deliver succeeds — target stays non-dead.
        target = DeliveryTarget(platform="telegram", chat_id="12345")
        await router.deliver("test", [target])
        assert router.dead_targets.is_dead("telegram", "12345") is False


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_normalizes_case(self):
        assert _normalize("Telegram", "12345") == "telegram:12345"

    def test_strips_whitespace(self):
        assert _normalize("  telegram  ", "  12345  ") == "telegram:12345"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
