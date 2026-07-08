"""FTS5-backed search index over NIA's session snapshots.

Builds and maintains a SQLite FTS5 index over every session saved by
:mod:`niaharness.services.session_storage`.  The index lives at
``<data_dir>/sessions.sqlite`` and is updated incrementally whenever
:func:`save_session_snapshot` is called.

Schema
------
Two tables in a single SQLite database:

1. ``sessions`` — metadata table (one row per session):
   - ``session_id`` (PRIMARY KEY)
   - ``project_hash`` — SHA-256 prefix of the resolved cwd
   - ``cwd`` — the project directory
   - ``model``
   - ``summary``
   - ``message_count``
   - ``created_at`` — ISO 8601 timestamp
   - ``file_path`` — path to the JSON snapshot on disk

2. ``messages_fts`` — FTS5 virtual table (one row per message):
   - ``session_id`` (UNINDEXED)
   - ``message_idx`` (UNINDEXED) — position in the conversation
   - ``role`` (UNINDEXED) — "user" or "assistant"
   - ``text`` — the message's concatenated text content (indexed)
   - ``tokenize = 'porter unicode61'``

The FTS5 table lets us find the exact message that matched a query, so we
can return a ±N message window around the hit (like Hermes's session_search).

Reference: Hermes Agent's ``tools/session_search_tool.py`` + ``hermes_state.py``
FTS5 index.  NIA's version is simpler — no session lineage dedup, no source
tagging — but mirrors the three-mode calling shape (DISCOVERY / SCROLL /
BROWSE).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _get_data_dir() -> Path:
    """Return the data dir, honoring ``NIAHARNESS_DATA_DIR`` env var."""
    data_dir_env = os.environ.get("NIAHARNESS_DATA_DIR")
    if data_dir_env:
        return Path(data_dir_env)
    from niaharness.config.paths import get_data_dir

    return get_data_dir()


def get_search_db_path() -> Path:
    """Return the path to the session search SQLite database."""
    return _get_data_dir() / "sessions.sqlite"


def _project_hash(cwd: str | Path) -> str:
    """Return the 16-char SHA-256 prefix for a cwd (matches session_storage)."""
    resolved = Path(cwd).resolve()
    return hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Message text extraction
# ---------------------------------------------------------------------------


def _extract_message_text(message: dict[str, Any]) -> str:
    """Extract all text from a serialised ConversationMessage.

    A message looks like ``{"role": "user", "content": [{"type": "text", "text": "..."}, ...]}``.
    Tool-use and tool-result blocks are rendered as short placeholders so
    their content is searchable but doesn't dominate the index.
    """
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
        elif btype == "tool_use":
            name = block.get("name", "unknown")
            inp = block.get("input", {})
            parts.append(f"[tool_use {name}] {json.dumps(inp, default=str)[:200]}")
        elif btype == "tool_result":
            c = block.get("content", "")
            parts.append(f"[tool_result] {str(c)[:200]}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class SessionSearchIndex:
    """SQLite FTS5 index over NIA's session snapshots.

    Thread-safe via a single lock around all writes.  Reads are concurrent
    (SQLite handles read locking internally).

    Usage::

        index = SessionSearchIndex()
        index.index_session(payload)   # add/update a session
        results = index.search("flask route")  # FTS5 query
        results = index.list_recent(limit=10)  # browse mode
        window = index.get_messages_around(session_id, 5, window=5)
    """

    _init_paths: set[str] = set()
    _schema_lock = threading.Lock()

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or get_search_db_path()
        self._lock = threading.Lock()
        self._ensure_schema()

    # ---- schema --------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist.  Idempotent per-db-path."""
        path_str = str(self._db_path)
        with self._schema_lock:
            if path_str in SessionSearchIndex._init_paths:
                return
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id    TEXT PRIMARY KEY,
                        project_hash  TEXT,
                        cwd           TEXT,
                        model         TEXT,
                        summary       TEXT,
                        message_count INTEGER,
                        created_at    TEXT,
                        file_path     TEXT
                    );

                    CREATE INDEX IF NOT EXISTS idx_sessions_project
                        ON sessions(project_hash);

                    CREATE INDEX IF NOT EXISTS idx_sessions_created
                        ON sessions(created_at DESC);

                    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                        session_id  UNINDEXED,
                        message_idx UNINDEXED,
                        role        UNINDEXED,
                        text,
                        tokenize = 'porter unicode61'
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()
            SessionSearchIndex._init_paths.add(path_str)

    # ---- write ---------------------------------------------------------

    def index_session(self, payload: dict[str, Any]) -> None:
        """Add or update a session in the index.

        ``payload`` is the dict shape produced by
        :func:`niaharness.services.session_storage.save_session_snapshot`:
        ``{session_id, cwd, model, system_prompt, messages, usage, summary,
        message_count, created_at}``.
        """
        session_id = payload.get("session_id")
        if not session_id:
            return

        cwd = payload.get("cwd", "")
        project_hash = _project_hash(cwd) if cwd else ""
        model = payload.get("model", "")
        summary = payload.get("summary", "")
        message_count = payload.get("message_count", 0)
        created_at = payload.get("created_at", datetime.now(timezone.utc).isoformat())

        # Find the file path on disk (for later retrieval).
        file_path = ""
        try:
            from niaharness.services.session_storage import get_project_session_dir

            file_path = str(get_project_session_dir(cwd) / f"{session_id}.json")
        except Exception:
            pass

        messages = payload.get("messages", [])

        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                # Replace existing session (cascade delete messages).
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.execute(
                    "DELETE FROM messages_fts WHERE session_id = ?", (session_id,)
                )

                conn.execute(
                    """INSERT INTO sessions
                       (session_id, project_hash, cwd, model, summary,
                        message_count, created_at, file_path)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        project_hash,
                        cwd,
                        model,
                        summary,
                        message_count,
                        created_at,
                        file_path,
                    ),
                )

                for idx, msg in enumerate(messages):
                    text = _extract_message_text(msg)
                    if not text.strip():
                        continue
                    role = msg.get("role", "unknown")
                    conn.execute(
                        """INSERT INTO messages_fts
                           (session_id, message_idx, role, text)
                           VALUES (?, ?, ?, ?)""",
                        (session_id, idx, role, text),
                    )

                conn.commit()
            except Exception as exc:
                logger.warning("Failed to index session %s: %s", session_id, exc)
                conn.rollback()
            finally:
                conn.close()

    # ---- search (DISCOVERY mode) --------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        project_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run an FTS5 query and return matching sessions with snippets.

        Each result contains:
        - ``session_id``, ``cwd``, ``model``, ``summary``, ``created_at``
        - ``snippet`` — FTS5 snippet of the matching text
        - ``match_message_idx`` — the message index where the match was found
        - ``match_role`` — the role of the matching message
        - ``message_count`` — total messages in the session
        """
        if not query.strip():
            return []

        # Sanitize the query for FTS5.  Wrap each term in quotes to avoid
        # FTS5 syntax errors from special characters.
        terms = query.strip().split()
        fts_query = " ".join(f'"{t}"' for t in terms if t)

        sql = """
            SELECT s.session_id, s.cwd, s.model, s.summary, s.created_at,
                   s.message_count,
                   m.message_idx, m.role,
                   snippet(messages_fts, 3, '>>', '<<', '...', 32) AS snippet
            FROM messages_fts m
            JOIN sessions s ON s.session_id = m.session_id
            WHERE messages_fts MATCH ?
        """
        params: list[Any] = [fts_query]
        if project_hash:
            sql += " AND s.project_hash = ?"
            params.append(project_hash)
        sql += " ORDER BY s.created_at DESC LIMIT ?"
        params.append(limit * 3)  # over-fetch, then dedupe by session

        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        # Dedupe by session_id — keep the first (best) match per session.
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        for row in rows:
            sid = row["session_id"]
            if sid in seen:
                continue
            seen.add(sid)
            results.append(
                {
                    "session_id": sid,
                    "cwd": row["cwd"],
                    "model": row["model"],
                    "summary": row["summary"],
                    "created_at": row["created_at"],
                    "message_count": row["message_count"],
                    "snippet": row["snippet"],
                    "match_message_idx": row["message_idx"],
                    "match_role": row["role"],
                }
            )
            if len(results) >= limit:
                break
        return results

    # ---- browse (BROWSE mode) -----------------------------------------

    def list_recent(
        self,
        *,
        limit: int = 20,
        project_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent sessions, newest first."""
        sql = """
            SELECT session_id, cwd, model, summary, created_at, message_count
            FROM sessions
        """
        params: list[Any] = []
        if project_hash:
            sql += " WHERE project_hash = ?"
            params.append(project_hash)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        return [
            {
                "session_id": r["session_id"],
                "cwd": r["cwd"],
                "model": r["model"],
                "summary": r["summary"],
                "created_at": r["created_at"],
                "message_count": r["message_count"],
            }
            for r in rows
        ]

    # ---- scroll (SCROLL mode) -----------------------------------------

    def get_messages_around(
        self,
        session_id: str,
        message_idx: int,
        *,
        window: int = 5,
    ) -> dict[str, Any] | None:
        """Return a window of messages around ``message_idx`` in a session.

        Returns ``None`` if the session isn't in the index.  The result dict
        contains:
        - ``session_id``, ``cwd``, ``model``, ``summary``, ``created_at``
        - ``anchor_idx`` — the requested message index
        - ``messages`` — list of ``{idx, role, text}`` dicts for the window
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            # Session metadata.
            row = conn.execute(
                """SELECT session_id, cwd, model, summary, created_at,
                          message_count
                   FROM sessions WHERE session_id = ?""",
                (session_id,),
            ).fetchone()
            if row is None:
                return None

            # Window of messages.
            start = max(0, message_idx - window)
            end = message_idx + window
            msg_rows = conn.execute(
                """SELECT message_idx, role, text
                   FROM messages_fts
                   WHERE session_id = ? AND message_idx BETWEEN ? AND ?
                   ORDER BY message_idx""",
                (session_id, start, end),
            ).fetchall()
        finally:
            conn.close()

        return {
            "session_id": row["session_id"],
            "cwd": row["cwd"],
            "model": row["model"],
            "summary": row["summary"],
            "created_at": row["created_at"],
            "message_count": row["message_count"],
            "anchor_idx": message_idx,
            "messages": [
                {"idx": m["message_idx"], "role": m["role"], "text": m["text"]}
                for m in msg_rows
            ],
        }

    # ---- rebuild -------------------------------------------------------

    def rebuild_from_sessions_dir(self) -> int:
        """Rebuild the entire index from on-disk session JSON files.

        Walks every ``<data_dir>/sessions/*/*.json`` file and re-indexes it.
        Returns the number of sessions indexed.  Useful for backfilling the
        index the first time the search feature is enabled, or for recovering
        from a corrupt index.
        """
        sessions_root = _get_data_dir() / "sessions"
        if not sessions_root.exists():
            return 0

        count = 0
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                # Wipe existing index.
                conn.execute("DELETE FROM sessions")
                conn.execute("DELETE FROM messages_fts")
                conn.commit()
            finally:
                conn.close()

        # Re-index each session file (uses index_session which has its own lock).
        for json_path in sorted(sessions_root.rglob("*.json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                if "session_id" not in payload:
                    continue
                self.index_session(payload)
                count += 1
            except Exception as exc:
                logger.warning("Failed to index %s: %s", json_path, exc)
        return count

    # ---- stats ---------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return index statistics."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            messages = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        finally:
            conn.close()
        return {"sessions": sessions, "messages": messages}


# ---------------------------------------------------------------------------
# Singleton + convenience
# ---------------------------------------------------------------------------

_index_singleton: SessionSearchIndex | None = None
_index_lock = threading.Lock()


def get_search_index() -> SessionSearchIndex:
    """Return the process-wide SessionSearchIndex singleton."""
    global _index_singleton
    if _index_singleton is None:
        with _index_lock:
            if _index_singleton is None:
                _index_singleton = SessionSearchIndex()
    return _index_singleton


def index_session_on_save(payload: dict[str, Any]) -> None:
    """Hook called by session_storage.save_session_snapshot to update the index.

    Best-effort: never fails the save if indexing fails.
    """
    try:
        get_search_index().index_session(payload)
    except Exception as exc:
        logger.warning("Session search indexing failed (non-fatal): %s", exc)
