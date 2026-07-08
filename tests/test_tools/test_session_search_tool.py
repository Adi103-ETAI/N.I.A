"""Tests for the session_search tool and FTS5 index."""

from __future__ import annotations

from pathlib import Path

import pytest

from niaharness.api.usage import UsageSnapshot
from niaharness.engine.messages import ConversationMessage, TextBlock
from niaharness.services.session_search import SessionSearchIndex
from niaharness.services.session_storage import save_session_snapshot
from niaharness.tools.base import ToolExecutionContext
from niaharness.tools.session_search_tool import SessionSearchTool, SessionSearchToolInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect all data dirs to a temp directory and reset the index singleton."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("NIAHARNESS_DATA_DIR", str(data_dir))

    # Reset the index singleton and schema cache so they pick up the new data dir.
    import niaharness.services.session_search as ss

    ss._index_singleton = None
    ss.SessionSearchIndex._init_paths.clear()
    yield data_dir
    ss._index_singleton = None
    ss.SessionSearchIndex._init_paths.clear()


@pytest.fixture
def context(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path)


def _make_messages(*texts: tuple[str, str]) -> list[ConversationMessage]:
    """Build a list of ConversationMessages from (role, text) tuples."""
    return [
        ConversationMessage(role=role, content=[TextBlock(text=text)])
        for role, text in texts
    ]


# ---------------------------------------------------------------------------
# Index tests
# ---------------------------------------------------------------------------


class TestSessionSearchIndex:
    def test_index_session_adds_to_index(self, isolated_data_dir: Path):
        index = SessionSearchIndex()
        save_session_snapshot(
            cwd="/tmp/test-project",
            model="claude-test",
            system_prompt="system",
            messages=_make_messages(
                ("user", "How do I deploy a Flask app?"),
                ("assistant", "You can use gunicorn or uwsgi."),
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=2),
            session_id="test-001",
        )

        # The save should have triggered indexing via the hook.
        stats = index.stats()
        assert stats["sessions"] == 1
        assert stats["messages"] == 2

    def test_search_finds_matching_text(self, isolated_data_dir: Path):
        index = SessionSearchIndex()
        save_session_snapshot(
            cwd="/tmp/proj-a",
            model="m",
            system_prompt="s",
            messages=_make_messages(
                ("user", "How do I deploy a Flask app?"),
                ("assistant", "Use gunicorn."),
            ),
            session_id="flask-session",
        )
        save_session_snapshot(
            cwd="/tmp/proj-b",
            model="m",
            system_prompt="s",
            messages=_make_messages(
                ("user", "What is the capital of France?"),
                ("assistant", "Paris."),
            ),
            session_id="paris-session",
        )

        results = index.search("Flask")
        assert len(results) == 1
        assert results[0]["session_id"] == "flask-session"
        assert "Flask" in results[0]["snippet"] or "flask" in results[0]["snippet"].lower()

    def test_search_returns_snippet_and_message_idx(self, isolated_data_dir: Path):
        index = SessionSearchIndex()
        save_session_snapshot(
            cwd="/tmp/proj",
            model="m",
            system_prompt="s",
            messages=_make_messages(
                ("user", "first message about dogs"),
                ("assistant", "response about cats"),
                ("user", "third message about birds"),
                ("assistant", "response about fish"),
            ),
            session_id="multi-msg",
        )

        results = index.search("birds")
        assert len(results) == 1
        assert results[0]["match_message_idx"] == 2  # third message (0-indexed)
        assert results[0]["match_role"] == "user"

    def test_search_empty_query_returns_empty(self, isolated_data_dir: Path):
        index = SessionSearchIndex()
        save_session_snapshot(
            cwd="/tmp/p",
            model="m",
            system_prompt="s",
            messages=_make_messages(("user", "hello")),
            session_id="s1",
        )
        assert index.search("") == []
        assert index.search("   ") == []

    def test_list_recent_returns_newest_first(self, isolated_data_dir: Path):
        index = SessionSearchIndex()
        save_session_snapshot(
            cwd="/tmp/p",
            model="m",
            system_prompt="s",
            messages=_make_messages(("user", "first session")),
            session_id="s1",
        )
        save_session_snapshot(
            cwd="/tmp/p",
            model="m",
            system_prompt="s",
            messages=_make_messages(("user", "second session")),
            session_id="s2",
        )

        recent = index.list_recent()
        assert len(recent) == 2
        # s2 was saved last, should be first.
        assert recent[0]["session_id"] == "s2"

    def test_get_messages_around_returns_window(self, isolated_data_dir: Path):
        index = SessionSearchIndex()
        save_session_snapshot(
            cwd="/tmp/p",
            model="m",
            system_prompt="s",
            messages=_make_messages(
                ("user", "msg0"),
                ("assistant", "msg1"),
                ("user", "msg2"),
                ("assistant", "msg3"),
                ("user", "msg4"),
                ("assistant", "msg5"),
                ("user", "msg6"),
            ),
            session_id="window-test",
        )

        result = index.get_messages_around("window-test", 3, window=2)
        assert result is not None
        assert result["anchor_idx"] == 3
        # Should return messages 1,2,3,4,5 (±2 around idx 3).
        idxs = [m["idx"] for m in result["messages"]]
        assert idxs == [1, 2, 3, 4, 5]

    def test_get_messages_around_unknown_session(self, isolated_data_dir: Path):
        index = SessionSearchIndex()
        result = index.get_messages_around("nonexistent", 0, window=5)
        assert result is None

    def test_reindexing_same_session_replaces(self, isolated_data_dir: Path):
        """Saving the same session_id twice should replace, not duplicate."""
        index = SessionSearchIndex()
        save_session_snapshot(
            cwd="/tmp/p",
            model="m",
            system_prompt="s",
            messages=_make_messages(("user", "v1")),
            session_id="replace-test",
        )
        save_session_snapshot(
            cwd="/tmp/p",
            model="m",
            system_prompt="s",
            messages=_make_messages(("user", "v2"), ("assistant", "v2-resp")),
            session_id="replace-test",
        )

        stats = index.stats()
        assert stats["sessions"] == 1  # still 1 session, not 2
        assert stats["messages"] == 2  # 2 messages from the second save

    def test_rebuild_from_sessions_dir(self, isolated_data_dir: Path):
        """Rebuild should wipe and re-index from on-disk files."""
        # Save a session (this auto-indexes).
        save_session_snapshot(
            cwd="/tmp/p",
            model="m",
            system_prompt="s",
            messages=_make_messages(("user", "rebuild me")),
            session_id="rb-1",
        )

        # Manually add junk to the index.
        index = SessionSearchIndex()
        index.index_session(
            {
                "session_id": "junk",
                "cwd": "/tmp/junk",
                "model": "m",
                "system_prompt": "s",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "junk"}]}],
                "summary": "junk",
                "message_count": 1,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
        assert index.stats()["sessions"] == 2

        # Rebuild from disk — should drop the junk session.
        count = index.rebuild_from_sessions_dir()
        assert count == 1
        stats = index.stats()
        assert stats["sessions"] == 1
        assert stats["messages"] == 1


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


class TestSessionSearchTool:
    @pytest.mark.asyncio
    async def test_search_mode(self, isolated_data_dir: Path, context: ToolExecutionContext):
        save_session_snapshot(
            cwd="/tmp/p",
            model="m",
            system_prompt="s",
            messages=_make_messages(
                ("user", "How do I deploy a Django app?"),
                ("assistant", "Use gunicorn."),
            ),
            session_id="django-s",
        )

        result = await SessionSearchTool().execute(
            SessionSearchToolInput(query="Django"),
            context,
        )
        assert result.is_error is False
        assert "django-s" in result.output
        assert "Django" in result.output or "django" in result.output.lower()

    @pytest.mark.asyncio
    async def test_browse_mode_no_args(self, isolated_data_dir: Path, context: ToolExecutionContext):
        save_session_snapshot(
            cwd="/tmp/p",
            model="m",
            system_prompt="s",
            messages=_make_messages(("user", "hello")),
            session_id="s1",
        )

        result = await SessionSearchTool().execute(
            SessionSearchToolInput(),
            context,
        )
        assert result.is_error is False
        assert "s1" in result.output
        assert "Recent sessions" in result.output

    @pytest.mark.asyncio
    async def test_browse_empty_index(self, isolated_data_dir: Path, context: ToolExecutionContext):
        result = await SessionSearchTool().execute(
            SessionSearchToolInput(),
            context,
        )
        assert result.is_error is False
        assert "No sessions" in result.output
        assert "rebuild" in result.output.lower()

    @pytest.mark.asyncio
    async def test_scroll_mode(self, isolated_data_dir: Path, context: ToolExecutionContext):
        save_session_snapshot(
            cwd="/tmp/p",
            model="m",
            system_prompt="s",
            messages=_make_messages(
                ("user", "msg0"),
                ("assistant", "msg1"),
                ("user", "msg2"),
                ("assistant", "msg3"),
                ("user", "msg4"),
            ),
            session_id="scroll-test",
        )

        result = await SessionSearchTool().execute(
            SessionSearchToolInput(
                session_id="scroll-test",
                around_message_idx=2,
                window=1,
            ),
            context,
        )
        assert result.is_error is False
        # Window ±1 around idx 2 = messages 1, 2, 3.
        # Check the anchored message markers (">>>" prefix).
        lines = result.output.splitlines()
        anchored = [l for l in lines if l.startswith(">>>")]
        assert len(anchored) == 1
        assert "[  2]" in anchored[0]  # anchor is message idx 2 (right-padded to 3 chars)
        assert "msg2" in anchored[0]
        # Messages 1 and 3 should appear (±1 window).
        non_anchor = [l for l in lines if l.strip().startswith("[") and not l.startswith(">>>")]
        window_texts = " ".join(non_anchor)
        assert "msg1" in window_texts
        assert "msg3" in window_texts
        # Message 4 and 0 should NOT be in the window.
        assert "msg4" not in window_texts
        assert "msg0" not in window_texts

    @pytest.mark.asyncio
    async def test_scroll_unknown_session(
        self, isolated_data_dir: Path, context: ToolExecutionContext
    ):
        result = await SessionSearchTool().execute(
            SessionSearchToolInput(session_id="nonexistent", around_message_idx=0),
            context,
        )
        assert result.is_error is True
        assert "not found" in result.output.lower()

    @pytest.mark.asyncio
    async def test_rebuild_action(self, isolated_data_dir: Path, context: ToolExecutionContext):
        save_session_snapshot(
            cwd="/tmp/p",
            model="m",
            system_prompt="s",
            messages=_make_messages(("user", "rebuild test")),
            session_id="rb-tool-1",
        )

        result = await SessionSearchTool().execute(
            SessionSearchToolInput(action="rebuild"),
            context,
        )
        assert result.is_error is False
        assert "Rebuilt" in result.output
        assert "1 session" in result.output

    @pytest.mark.asyncio
    async def test_stats_action(self, isolated_data_dir: Path, context: ToolExecutionContext):
        save_session_snapshot(
            cwd="/tmp/p",
            model="m",
            system_prompt="s",
            messages=_make_messages(("user", "stats test")),
            session_id="stats-1",
        )

        result = await SessionSearchTool().execute(
            SessionSearchToolInput(action="stats"),
            context,
        )
        assert result.is_error is False
        assert "Sessions indexed: 1" in result.output
        assert "Messages indexed: 1" in result.output

    @pytest.mark.asyncio
    async def test_search_no_results(
        self, isolated_data_dir: Path, context: ToolExecutionContext
    ):
        save_session_snapshot(
            cwd="/tmp/p",
            model="m",
            system_prompt="s",
            messages=_make_messages(("user", "hello world")),
            session_id="s1",
        )

        result = await SessionSearchTool().execute(
            SessionSearchToolInput(query="nonexistent-term-xyz"),
            context,
        )
        assert result.is_error is False
        assert "No sessions found" in result.output

    def test_is_read_only(self):
        tool = SessionSearchTool()
        # All actions are read-only (even rebuild, since it only touches the index).
        for action in ("search", "scroll", "browse", "rebuild", "stats"):
            args = SessionSearchToolInput(action=action)  # type: ignore[arg-type]
            assert tool.is_read_only(args) is True, f"{action} should be read-only"


# ---------------------------------------------------------------------------
# Integration: save auto-indexes
# ---------------------------------------------------------------------------


class TestAutoIndexing:
    @pytest.mark.asyncio
    async def test_save_then_search_immediately(
        self, isolated_data_dir: Path, context: ToolExecutionContext
    ):
        """Saving a session should make it immediately searchable."""
        save_session_snapshot(
            cwd="/tmp/auto-index-test",
            model="claude-sonnet",
            system_prompt="s",
            messages=_make_messages(
                ("user", "I need to set up a PostgreSQL database for my app."),
                ("assistant", "I'll help you with that. Let's start with the installation."),
            ),
            session_id="auto-1",
        )

        # Immediately search — should find the session without an explicit rebuild.
        result = await SessionSearchTool().execute(
            SessionSearchToolInput(query="PostgreSQL"),
            context,
        )
        assert result.is_error is False
        assert "auto-1" in result.output
        assert "PostgreSQL" in result.output or "postgresql" in result.output.lower()
