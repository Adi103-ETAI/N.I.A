"""Tests for the P1 SessionDB extensions — gateway routing, handoff, message ops,
session meta, titles, rich listing, pruning, export, AsyncSessionDB.

Covers all the methods added in the P1 commit. Uses a temp DB so tests
don't pollute the dev environment.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from niaharness.services.session_db import SessionDB, AsyncSessionDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    """Return a SessionDB instance using a temp file."""
    db_path = tmp_path / "test_sessions.db"
    return SessionDB(db_path=db_path)


@pytest.fixture
def session_with_messages(db: SessionDB) -> str:
    """Create a session with some messages, return the session id."""
    session_id = "test-session-001"
    db.create_session(session_id, cwd="/tmp/test", model="test-model")
    db.add_message(session_id, "user", "first message")
    db.add_message(session_id, "assistant", "first reply")
    db.add_message(session_id, "user", "second message")
    db.add_message(session_id, "assistant", "second reply")
    return session_id


# ---------------------------------------------------------------------------
# Gateway routing
# ---------------------------------------------------------------------------


class TestGatewayRouting:
    def test_record_gateway_session_peer(self, db: SessionDB):
        session_id = "gw-001"
        db.create_session(session_id)
        db.record_gateway_session_peer(
            session_id,
            source="telegram",
            user_id="user123",
            session_key="tg:user123:chat456",
            chat_id="chat456",
            chat_type="private",
        )
        session = db.get_session(session_id)
        assert session["source"] == "telegram"
        assert session["user_id"] == "user123"
        assert session["session_key"] == "tg:user123:chat456"

    def test_record_gateway_session_peer_noop_without_key(self, db: SessionDB):
        session_id = "gw-002"
        db.create_session(session_id)
        # No session_key → no-op.
        db.record_gateway_session_peer(
            session_id, source="telegram", session_key=None,
        )
        session = db.get_session(session_id)
        # source stays as default 'cli'
        assert session["source"] == "cli"

    def test_set_expiry_finalized(self, db: SessionDB):
        session_id = "gw-003"
        db.create_session(session_id)
        db.set_expiry_finalized(session_id, True)
        session = db.get_session(session_id)
        assert session["expiry_finalized"] == 1
        db.set_expiry_finalized(session_id, False)
        session = db.get_session(session_id)
        assert session["expiry_finalized"] == 0

    def test_save_load_gateway_routing_entry(self, db: SessionDB):
        db.save_gateway_routing_entry("key1", '{"session_id": "s1"}', scope="tg")
        db.save_gateway_routing_entry("key2", '{"session_id": "s2"}', scope="tg")
        entries = db.load_gateway_routing_entries(scope="tg")
        assert entries == {"key1": '{"session_id": "s1"}', "key2": '{"session_id": "s2"}'}

    def test_replace_gateway_routing_entries(self, db: SessionDB):
        db.save_gateway_routing_entry("key1", "old1", scope="tg")
        db.replace_gateway_routing_entries({"key2": "new2"}, scope="tg")
        entries = db.load_gateway_routing_entries(scope="tg")
        assert "key1" not in entries
        assert entries == {"key2": "new2"}

    def test_delete_gateway_routing_entries(self, db: SessionDB):
        db.save_gateway_routing_entry("key1", "v1", scope="tg")
        db.save_gateway_routing_entry("key2", "v2", scope="tg")
        db.delete_gateway_routing_entries(["key1"], scope="tg")
        entries = db.load_gateway_routing_entries(scope="tg")
        assert "key1" not in entries
        assert entries == {"key2": "v2"}

    def test_list_gateway_sessions(self, db: SessionDB):
        db.create_session("s1")
        db.record_gateway_session_peer(
            "s1", source="telegram", session_key="k1", chat_id="c1",
        )
        db.create_session("s2")
        db.record_gateway_session_peer(
            "s2", source="telegram", session_key="k2", chat_id="c2",
        )
        # Non-gateway session (no session_key).
        db.create_session("s3")
        gw_sessions = db.list_gateway_sessions()
        assert len(gw_sessions) == 2
        gw_keys = {s["session_key"] for s in gw_sessions}
        assert gw_keys == {"k1", "k2"}

    def test_find_session_by_origin(self, db: SessionDB):
        db.create_session("s1")
        db.record_gateway_session_peer(
            "s1", source="telegram", session_key="k1",
            chat_id="chat123", user_id="user456",
        )
        found = db.find_session_by_origin(
            platform="telegram", chat_id="chat123", user_id="user456",
        )
        assert found == "s1"

    def test_find_session_by_origin_returns_none_for_unknown(self, db: SessionDB):
        found = db.find_session_by_origin(
            platform="telegram", chat_id="nonexistent",
        )
        assert found is None

    def test_find_latest_gateway_session_for_peer(self, db: SessionDB):
        db.create_session("s1")
        db.record_gateway_session_peer(
            "s1", source="telegram", session_key="k1",
            chat_id="c1", chat_type="private", user_id="u1",
        )
        db.add_message("s1", "user", "hello")
        result = db.find_latest_gateway_session_for_peer(
            source="telegram", session_key="k1",
        )
        assert result is not None
        assert result["id"] == "s1"

    def test_find_latest_gateway_session_for_peer_returns_none(self, db: SessionDB):
        result = db.find_latest_gateway_session_for_peer(
            source="telegram", session_key="nonexistent",
        )
        assert result is None


# ---------------------------------------------------------------------------
# Handoff
# ---------------------------------------------------------------------------


class TestHandoff:
    def test_request_handoff(self, db: SessionDB):
        db.create_session("s1")
        result = db.request_handoff("s1", "telegram")
        assert result is True
        state = db.get_handoff_state("s1")
        assert state["state"] == "pending"
        assert state["platform"] == "telegram"

    def test_request_handoff_returns_false_if_already_pending(self, db: SessionDB):
        db.create_session("s1")
        db.request_handoff("s1", "telegram")
        # Second request while pending → False.
        result = db.request_handoff("s1", "discord")
        assert result is False

    def test_get_handoff_state_none_if_no_handoff(self, db: SessionDB):
        db.create_session("s1")
        state = db.get_handoff_state("s1")
        assert state is not None
        assert state["state"] is None

    def test_list_pending_handoffs(self, db: SessionDB):
        db.create_session("s1")
        db.create_session("s2")
        db.request_handoff("s1", "telegram")
        db.request_handoff("s2", "discord")
        pending = db.list_pending_handoffs()
        assert len(pending) == 2

    def test_claim_handoff(self, db: SessionDB):
        db.create_session("s1")
        db.request_handoff("s1", "telegram")
        claimed = db.claim_handoff("s1")
        assert claimed is True
        state = db.get_handoff_state("s1")
        assert state["state"] == "running"

    def test_claim_handoff_returns_false_if_not_pending(self, db: SessionDB):
        db.create_session("s1")
        # No pending handoff → can't claim.
        claimed = db.claim_handoff("s1")
        assert claimed is False

    def test_complete_handoff(self, db: SessionDB):
        db.create_session("s1")
        db.request_handoff("s1", "telegram")
        db.claim_handoff("s1")
        db.complete_handoff("s1")
        state = db.get_handoff_state("s1")
        assert state["state"] == "completed"

    def test_fail_handoff(self, db: SessionDB):
        db.create_session("s1")
        db.request_handoff("s1", "telegram")
        db.claim_handoff("s1")
        db.fail_handoff("s1", "connection refused")
        state = db.get_handoff_state("s1")
        assert state["state"] == "failed"
        assert "connection refused" in state["error"]


# ---------------------------------------------------------------------------
# Message operations
# ---------------------------------------------------------------------------


class TestMessageOps:
    def test_has_archived_messages_false_initially(self, db: SessionDB, session_with_messages: str):
        assert db.has_archived_messages(session_with_messages) is False

    def test_archive_and_compact(self, db: SessionDB, session_with_messages: str):
        compacted = [{"role": "system", "content": "summary of conversation"}]
        new_count = db.archive_and_compact(session_with_messages, compacted)
        assert new_count == 1
        # Old messages are now archived (active=0).
        assert db.has_archived_messages(session_with_messages) is True
        # NIA's get_messages filters on compacted=0, so archived messages
        # (active=0, compacted=1) are excluded. The new summary message
        # (active=1, compacted=0) should be the only one returned.
        active = db.get_messages(session_with_messages)
        assert len(active) == 1
        assert active[0]["content"] == "summary of conversation"

    def test_get_messages_around(self, db: SessionDB, session_with_messages: str):
        messages = db.get_messages(session_with_messages)
        anchor_id = messages[1]["id"]  # second message
        result = db.get_messages_around(session_with_messages, anchor_id, window=2)
        assert result["anchor"] is not None
        assert result["anchor"]["id"] == anchor_id
        assert len(result["before"]) >= 1
        assert len(result["after"]) >= 1

    def test_get_messages_around_returns_empty_for_unknown(self, db: SessionDB, session_with_messages: str):
        result = db.get_messages_around(session_with_messages, 99999)
        assert result["anchor"] is None
        assert result["before"] == []
        assert result["after"] == []

    def test_rewind_to_message(self, db: SessionDB, session_with_messages: str):
        messages = db.get_messages(session_with_messages)
        # Rewind to the 3rd message (a user message).
        target_id = messages[2]["id"]
        result = db.rewind_to_message(session_with_messages, target_id)
        assert result["rewound_count"] == 2  # 3rd + 4th message
        assert result["target_message"]["id"] == target_id
        # NIA's get_messages filters on compacted=0, not active=1, so rewound
        # messages (active=0, compacted=0) still appear. Verify via the
        # active flag instead.
        all_messages = db.get_messages(session_with_messages)
        active_messages = [m for m in all_messages if m.get("active", 1) == 1]
        assert len(active_messages) == 2
        # The rewound messages should have active=0.
        rewound = [m for m in all_messages if m.get("active", 1) == 0]
        assert len(rewound) == 2

    def test_rewind_to_message_raises_for_nonexistent(self, db: SessionDB, session_with_messages: str):
        with pytest.raises(ValueError, match="not found"):
            db.rewind_to_message(session_with_messages, 99999)

    def test_rewind_to_message_raises_for_non_user(self, db: SessionDB, session_with_messages: str):
        messages = db.get_messages(session_with_messages)
        # The 2nd message is an assistant message.
        assistant_id = messages[1]["id"]
        with pytest.raises(ValueError, match="must be a 'user' message"):
            db.rewind_to_message(session_with_messages, assistant_id)

    def test_restore_rewound(self, db: SessionDB, session_with_messages: str):
        messages = db.get_messages(session_with_messages)
        target_id = messages[2]["id"]
        db.rewind_to_message(session_with_messages, target_id)
        restored = db.restore_rewound(session_with_messages, target_id)
        assert restored == 2
        # All 4 messages should be active again.
        all_messages = db.get_messages(session_with_messages)
        active = [m for m in all_messages if m.get("active", 1) == 1]
        assert len(active) == 4

    def test_clear_messages(self, db: SessionDB, session_with_messages: str):
        db.clear_messages(session_with_messages)
        messages = db.get_messages(session_with_messages)
        assert len(messages) == 0

    def test_resolve_resume_session_id_no_children(self, db: SessionDB, session_with_messages: str):
        result = db.resolve_resume_session_id(session_with_messages)
        assert result == session_with_messages

    def test_resolve_resume_session_id_follows_chain(self, db: SessionDB):
        db.create_session("parent")
        db.create_session("child", parent_session_id="parent")
        db.create_session("grandchild", parent_session_id="child")
        result = db.resolve_resume_session_id("parent")
        assert result == "grandchild"


# ---------------------------------------------------------------------------
# Session meta + update methods
# ---------------------------------------------------------------------------


class TestSessionMeta:
    def test_update_session_meta(self, db: SessionDB):
        db.create_session("s1")
        db.update_session_meta("s1", {"key1": "value1"})
        session = db.get_session("s1")
        metadata = json.loads(session["metadata"])
        assert metadata["key1"] == "value1"
        # Merge.
        db.update_session_meta("s1", {"key2": "value2"})
        session = db.get_session("s1")
        metadata = json.loads(session["metadata"])
        assert metadata["key1"] == "value1"
        assert metadata["key2"] == "value2"

    def test_update_session_meta_returns_false_for_unknown(self, db: SessionDB):
        result = db.update_session_meta("nonexistent", {"k": "v"})
        assert result is False

    def test_update_system_prompt(self, db: SessionDB):
        db.create_session("s1")
        db.update_system_prompt("s1", "new system prompt")
        session = db.get_session("s1")
        assert session["system_prompt"] == "new system prompt"

    def test_update_session_model(self, db: SessionDB):
        db.create_session("s1")
        db.update_session_model("s1", "claude-4-opus")
        session = db.get_session("s1")
        assert session["model"] == "claude-4-opus"

    def test_update_session_billing_route(self, db: SessionDB):
        db.create_session("s1")
        db.update_session_billing_route(
            "s1", provider="anthropic", base_url="https://api.anthropic.com", mode="standard",
        )
        session = db.get_session("s1")
        assert session["billing_provider"] == "anthropic"
        assert session["billing_base_url"] == "https://api.anthropic.com"
        assert session["billing_mode"] == "standard"

    def test_update_token_counts(self, db: SessionDB):
        db.create_session("s1")
        db.update_token_counts("s1", input_tokens=100, output_tokens=50)
        session = db.get_session("s1")
        assert session["input_tokens"] == 100
        assert session["output_tokens"] == 50
        # Additive.
        db.update_token_counts("s1", input_tokens=200, output_tokens=100)
        session = db.get_session("s1")
        assert session["input_tokens"] == 300
        assert session["output_tokens"] == 150

    def test_update_session_cwd(self, db: SessionDB):
        db.create_session("s1")
        db.update_session_cwd("s1", "/new/path", git_branch="feature")
        session = db.get_session("s1")
        assert session["project_path"] == "/new/path"
        assert session["git_branch"] == "feature"

    def test_backfill_repo_roots(self, db: SessionDB):
        db.create_session("s1", cwd="/repo1")
        db.create_session("s2", cwd="/repo2")
        db.backfill_repo_roots({"/repo1": "/root1", "/repo2": "/root2"})
        s1 = db.get_session("s1")
        s2 = db.get_session("s2")
        assert s1["git_repo_root"] == "/root1"
        assert s2["git_repo_root"] == "/root2"

    def test_set_session_archived(self, db: SessionDB):
        db.create_session("s1")
        assert db.set_session_archived("s1", True) is True
        session = db.get_session("s1")
        assert session["archived"] == 1
        assert db.set_session_archived("s1", False) is True
        session = db.get_session("s1")
        assert session["archived"] == 0

    def test_set_session_archived_returns_false_for_unknown(self, db: SessionDB):
        assert db.set_session_archived("nonexistent", True) is False


# ---------------------------------------------------------------------------
# Title methods
# ---------------------------------------------------------------------------


class TestTitleMethods:
    def test_sanitize_title_strips_control_chars(self):
        assert SessionDB.sanitize_title("hello\x00world") == "helloworld"
        assert SessionDB.sanitize_title("  spaced  ") == "spaced"
        assert SessionDB.sanitize_title(None) is None
        assert SessionDB.sanitize_title("") is None

    def test_sanitize_title_truncates(self):
        long = "x" * 300
        result = SessionDB.sanitize_title(long)
        assert len(result) == 200

    def test_get_session_by_title(self, db: SessionDB):
        db.create_session("s1", title="My Project")
        session = db.get_session_by_title("My Project")
        assert session is not None
        assert session["id"] == "s1"

    def test_get_session_by_title_returns_none(self, db: SessionDB):
        db.create_session("s1")
        assert db.get_session_by_title("nonexistent") is None

    def test_resolve_session_by_title(self, db: SessionDB):
        db.create_session("s1", title="My Project")
        assert db.resolve_session_by_title("My Project") == "s1"
        assert db.resolve_session_by_title("nonexistent") is None

    def test_get_next_title_in_lineage(self, db: SessionDB):
        db.create_session("s1", title="Project")
        # First call → "Project" already exists → "Project (1)"
        assert db.get_next_title_in_lineage("Project") == "Project (1)"
        db.create_session("s2", title="Project (1)")
        # Now → "Project (2)"
        assert db.get_next_title_in_lineage("Project") == "Project (2)"

    def test_get_compression_tip(self, db: SessionDB):
        db.create_session("root")
        db.create_session("child", parent_session_id="root")
        db.create_session("grandchild", parent_session_id="child")
        assert db.get_compression_tip("root") == "grandchild"


# ---------------------------------------------------------------------------
# Rich listing + pruning
# ---------------------------------------------------------------------------


class TestRichListingAndPruning:
    def test_distinct_session_cwds(self, db: SessionDB):
        db.create_session("s1", cwd="/repo1")
        db.create_session("s2", cwd="/repo1")
        db.create_session("s3", cwd="/repo2")
        cwds = db.distinct_session_cwds()
        assert len(cwds) == 2
        by_cwd = {c["cwd"]: c for c in cwds}
        assert by_cwd["/repo1"]["sessions"] == 2
        assert by_cwd["/repo2"]["sessions"] == 1

    def test_list_sessions_rich(self, db: SessionDB):
        db.create_session("s1", cwd="/proj", title="Project 1")
        db.add_message("s1", "user", "first message here")
        db.create_session("s2", cwd="/proj", title="Project 2")
        db.add_message("s2", "user", "another message")
        sessions = db.list_sessions_rich(limit=10)
        assert len(sessions) == 2
        # Should have preview + last_active.
        assert "preview" in sessions[0]
        assert "last_active" in sessions[0]

    def test_list_sessions_rich_filter_by_source(self, db: SessionDB):
        db.create_session("s1", source="cli")
        db.create_session("s2", source="telegram")
        sessions = db.list_sessions_rich(source="cli")
        assert len(sessions) == 1
        assert sessions[0]["source"] == "cli"

    def test_list_sessions_rich_search_query(self, db: SessionDB):
        db.create_session("s1", title="Authentication Bug")
        db.create_session("s2", title="Database Migration")
        sessions = db.list_sessions_rich(search_query="auth")
        assert len(sessions) == 1
        assert sessions[0]["title"] == "Authentication Bug"

    def test_list_cron_job_runs(self, db: SessionDB):
        db.create_session("s1", source="cron")
        db.create_session("s2", source="cli")
        db.create_session("s3", source="cron")
        cron_runs = db.list_cron_job_runs()
        assert len(cron_runs) == 2

    def test_list_prune_candidates(self, db: SessionDB):
        db.create_session("s1")
        db.end_session("s1", "completed")
        db.create_session("s2")  # still active
        candidates = db.list_prune_candidates(older_than_days=0)
        # Only ended sessions are candidates.
        assert len(candidates) == 1
        assert candidates[0]["id"] == "s1"

    def test_archive_sessions(self, db: SessionDB):
        db.create_session("s1")
        db.end_session("s1", "completed")
        count = db.archive_sessions(older_than_days=0)
        assert count == 1
        session = db.get_session("s1")
        assert session["archived"] == 1

    def test_prune_sessions(self, db: SessionDB):
        db.create_session("s1")
        db.end_session("s1", "completed")
        db.add_message("s1", "user", "msg")
        count = db.prune_sessions(older_than_days=0)
        assert count == 1
        # Session + messages should be gone.
        assert db.get_session("s1") is None

    def test_maybe_auto_prune_and_vacuum(self, db: SessionDB):
        db.create_session("s1")
        db.end_session("s1", "completed")
        result = db.maybe_auto_prune_and_vacuum(retention_days=0)
        assert result["skipped"] is False
        assert result["pruned"] == 1
        # Second call within 24h → skipped.
        result2 = db.maybe_auto_prune_and_vacuum(retention_days=0)
        assert result2["skipped"] is True

    def test_count_empty_sessions(self, db: SessionDB):
        db.create_session("s1")  # no messages
        db.create_session("s2")
        db.add_message("s2", "user", "msg")
        assert db.count_empty_sessions() == 1

    def test_delete_empty_sessions(self, db: SessionDB):
        db.create_session("s1")  # empty
        db.create_session("s2")
        db.add_message("s2", "user", "msg")
        count = db.delete_empty_sessions()
        assert count == 1
        assert db.get_session("s1") is None
        assert db.get_session("s2") is not None

    def test_finalize_orphaned_compression_sessions(self, db: SessionDB):
        db.create_session("parent")
        db.end_session("parent", "completed")
        db.create_session("child", parent_session_id="parent")
        # child is still active but parent has ended.
        count = db.finalize_orphaned_compression_sessions()
        assert count == 1
        child = db.get_session("child")
        assert child["ended_at"] is not None
        assert child["end_reason"] == "parent_ended"


# ---------------------------------------------------------------------------
# Export + misc
# ---------------------------------------------------------------------------


class TestExportAndMisc:
    def test_export_session(self, db: SessionDB, session_with_messages: str):
        export = db.export_session(session_with_messages)
        assert export is not None
        assert export["session"]["id"] == session_with_messages
        assert len(export["messages"]) == 4

    def test_export_session_returns_none_for_unknown(self, db: SessionDB):
        assert db.export_session("nonexistent") is None

    def test_export_all(self, db: SessionDB):
        db.create_session("s1")
        db.add_message("s1", "user", "msg1")
        db.create_session("s2")
        db.add_message("s2", "user", "msg2")
        exports = db.export_all()
        assert len(exports) == 2

    def test_session_count(self, db: SessionDB):
        db.create_session("s1")
        db.create_session("s2")
        assert db.session_count() == 2

    def test_message_count(self, db: SessionDB, session_with_messages: str):
        assert db.message_count(session_with_messages) == 4
        assert db.message_count() == 4

    def test_delete_session_if_empty(self, db: SessionDB):
        db.create_session("s1")  # empty
        assert db.delete_session_if_empty("s1") is True
        assert db.get_session("s1") is None

    def test_delete_session_if_empty_returns_false_for_non_empty(self, db: SessionDB, session_with_messages: str):
        assert db.delete_session_if_empty(session_with_messages) is False

    def test_delete_sessions(self, db: SessionDB):
        db.create_session("s1")
        db.create_session("s2")
        count = db.delete_sessions(["s1", "s2"])
        assert count == 2
        assert db.get_session("s1") is None
        assert db.get_session("s2") is None

    def test_search_sessions_by_id(self, db: SessionDB):
        db.create_session("session-abc-123")
        db.create_session("session-xyz-456")
        results = db.search_sessions_by_id("abc")
        assert len(results) == 1
        assert results[0]["id"] == "session-abc-123"

    def test_search_sessions(self, db: SessionDB):
        db.create_session("s1", title="Authentication Bug")
        db.create_session("s2", title="Database Migration")
        results = db.search_sessions("auth")
        assert len(results) == 1
        assert results[0]["title"] == "Authentication Bug"


# ---------------------------------------------------------------------------
# AsyncSessionDB
# ---------------------------------------------------------------------------


class TestAsyncSessionDB:
    @pytest.mark.asyncio
    async def test_async_get_session(self, db: SessionDB):
        db.create_session("s1")
        async_db = AsyncSessionDB(db)
        session = await async_db.get_session("s1")
        assert session is not None
        assert session["id"] == "s1"

    @pytest.mark.asyncio
    async def test_async_add_message(self, db: SessionDB):
        db.create_session("s1")
        async_db = AsyncSessionDB(db)
        msg_id = await async_db.add_message("s1", "user", "async message")
        assert msg_id > 0

    @pytest.mark.asyncio
    async def test_async_search_messages(self, db: SessionDB, session_with_messages: str):
        async_db = AsyncSessionDB(db)
        results = await async_db.search_messages("first")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_async_underlying_property(self, db: SessionDB):
        async_db = AsyncSessionDB(db)
        assert async_db.underlying is db


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
