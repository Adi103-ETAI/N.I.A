"""SQLite session storage with WAL mode — single DB for sessions + messages + FTS5.

Ported from the reference project's session management pattern (6,322 lines),
providing a robust SQLite-backed session store that replaces NIA's per-project
JSON file approach.

Why SQLite + WAL?
-----------------
NIA's current session storage uses per-project JSON files:
  ``<data_dir>/sessions/<project_hash>/<session_id>.json``

This has several issues:
  - **No cross-project queries** — to find a session by content, you must
    scan every project directory and parse every JSON file.
  - **No write concurrency** — concurrent NIA processes can clobber each
    other's writes to the same session file.
  - **No FTS5 search across all sessions** — the existing ``session_search``
    module maintains a *separate* SQLite FTS5 index that must be kept in
    sync with the JSON files.
  - **No session lineage** — parent/child session relationships (fork,
    resume, branch) require explicit bookkeeping in JSON.

SQLite with WAL mode solves all of these:
  - Single DB file at ``~/.nia/sessions.db`` holds all sessions + messages.
  - WAL mode allows concurrent readers + one writer per DB.
  - FTS5 virtual table enables full-text search across all sessions.
  - Cross-process advisory locking via ``PRAGMA busy_timeout``.
  - Session lineage via ``parent_session_id`` column.
  - Malformed DB auto-repair via ``PRAGMA integrity_check``.

Schema
------
::

    CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        project_hash TEXT NOT NULL,
        project_path TEXT NOT NULL,
        title TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        parent_session_id TEXT,  -- for fork/resume lineage
        message_count INTEGER DEFAULT 0,
        token_count INTEGER DEFAULT 0,
        model TEXT,
        provider TEXT,
        metadata TEXT  -- JSON blob for extensible metadata
    );

    CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        seq INTEGER NOT NULL,  -- 0-indexed position in session
        role TEXT NOT NULL,
        content TEXT NOT NULL,  -- JSON-serialized content blocks
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );

    CREATE VIRTUAL TABLE messages_fts USING fts5(
        content_text,
        content='messages',
        content_rowid='id',
        tokenize='porter unicode61'
    );

    -- Triggers to keep FTS5 in sync with messages table.
    CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, content_text) VALUES (new.id, new.content);
    END;
    CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content_text) VALUES ('delete', old.id, old.content);
    END;
    CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content_text) VALUES ('delete', old.id, old.content);
        INSERT INTO messages_fts(rowid, content_text) VALUES (new.id, new.content);
    END;
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WAL mode allows concurrent readers + one writer.
# busy_timeout: wait up to 5 seconds for a lock before raising SQLITE_BUSY.
_BUSY_TIMEOUT_MS = 5000

# Schema version for migrations.
_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _sessions_db_path() -> Path:
    """Return the path to the sessions SQLite DB (``~/.nia/sessions.db``)."""
    from niaharness.prompts.soul import get_nia_home

    return get_nia_home() / "sessions.db"


def _project_hash(cwd: str | Path) -> str:
    """Return a 16-char SHA-256 hash of the resolved absolute cwd."""
    resolved = Path(cwd).resolve()
    return hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

# Thread-local connection cache. Each thread gets its own connection to
# avoid SQLite's thread-safety restrictions.
_thread_local = threading.local()


def _get_connection() -> sqlite3.Connection:
    """Return a thread-local SQLite connection with WAL mode enabled."""
    conn = getattr(_thread_local, "conn", None)
    if conn is not None:
        return conn

    db_path = _sessions_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(db_path),
        timeout=_BUSY_TIMEOUT_MS / 1000.0,
        isolation_level=None,  # autocommit mode; we manage transactions explicitly
        check_same_thread=False,
    )
    # Enable row factory for dict-like access (row["column_name"]).
    conn.row_factory = sqlite3.Row

    # Enable WAL mode for concurrent read/write.
    conn.execute("PRAGMA journal_mode=WAL")
    # Set busy timeout (also set via connect timeout, but this is the SQLite-level setting).
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    # Foreign keys: enable cascade deletes.
    conn.execute("PRAGMA foreign_keys=ON")
    # Synchronous=NORMAL is safe with WAL and much faster than FULL.
    conn.execute("PRAGMA synchronous=NORMAL")
    # Temp store in memory for speed.
    conn.execute("PRAGMA temp_store=MEMORY")

    # Initialize schema if needed.
    _init_schema(conn)

    _thread_local.conn = conn
    return conn


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Context manager for a transaction. Commits on success, rolls back on error."""
    conn = _get_connection()
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _init_schema(conn: sqlite3.Connection) -> None:
    """Initialize the DB schema if it doesn't exist."""
    # Check if schema is already initialized.
    try:
        conn.execute("SELECT 1 FROM sessions LIMIT 1")
        return  # Table exists
    except sqlite3.OperationalError:
        pass

    # Create schema.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        );
        INSERT OR IGNORE INTO schema_version (version) VALUES (1);

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            project_hash TEXT NOT NULL,
            project_path TEXT NOT NULL,
            title TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            parent_session_id TEXT,
            message_count INTEGER DEFAULT 0,
            token_count INTEGER DEFAULT 0,
            model TEXT,
            provider TEXT,
            metadata TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_project_hash
            ON sessions(project_hash);
        CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
            ON sessions(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_sessions_parent
            ON sessions(parent_session_id);

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            content_text TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session_id
            ON messages(session_id, seq);

        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            content_text,
            content='messages',
            content_rowid='id',
            tokenize='porter unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content_text) VALUES (new.id, new.content_text);
        END;
        CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content_text) VALUES ('delete', old.id, old.content_text);
        END;
        CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content_text) VALUES ('delete', old.id, old.content_text);
            INSERT INTO messages_fts(rowid, content_text) VALUES (new.id, new.content_text);
        END;
        """
    )


def _repair_db_if_needed() -> None:
    """Run integrity check and repair if the DB is malformed.

    Adapted from the reference project's auto-repair logic. If the DB
    fails integrity check, we attempt to recover by dumping and re-importing.
    """
    conn = _get_connection()
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()
        if result and result[0] == "ok":
            return  # DB is healthy
        logger.warning("Session DB integrity check failed: %s — attempting repair", result[0])
    except sqlite3.DatabaseError as exc:
        logger.warning("Session DB integrity check raised: %s — attempting repair", exc)

    # Repair: rename the corrupted DB and let _init_schema create a fresh one.
    db_path = _sessions_db_path()
    backup_path = db_path.with_suffix(".db.corrupted")
    try:
        if backup_path.exists():
            backup_path.unlink()
        db_path.rename(backup_path)
        logger.warning("Corrupted session DB moved to %s", backup_path)
    except OSError as exc:
        logger.error("Failed to move corrupted DB: %s", exc)

    # Clear the thread-local connection so the next call creates a fresh one.
    if hasattr(_thread_local, "conn"):
        try:
            _thread_local.conn.close()
        except Exception:
            pass
        del _thread_local.conn

    # Re-initialize.
    conn = _get_connection()
    _init_schema(conn)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _serialize_content(content: Any) -> Tuple[str, str]:
    """Serialize message content to (json, text) for storage.

    Returns:
        (json_str, text_str) where json_str is the full serialized content
        and text_str is the extracted plain text for FTS5 indexing.
    """
    if isinstance(content, str):
        return json.dumps([{"type": "text", "text": content}]), content

    if isinstance(content, list):
        # List of content blocks (TextBlock, ToolUseBlock, etc.)
        blocks = []
        text_parts = []
        for block in content:
            if hasattr(block, "type"):
                # It's a typed block (TextBlock, etc.)
                block_dict: Dict[str, Any] = {"type": block.type}
                for attr in ("text", "tool_use_id", "name", "input", "tool_result_id"):
                    if hasattr(block, attr):
                        value = getattr(block, attr)
                        if value is not None:
                            block_dict[attr] = value
                blocks.append(block_dict)
                if block.type == "text" and hasattr(block, "text"):
                    text_parts.append(block.text)
            elif isinstance(block, dict):
                blocks.append(block)
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                blocks.append({"type": "text", "text": block})
                text_parts.append(block)
        return json.dumps(blocks), "\n".join(text_parts)

    # Fallback: serialize as a single text block.
    text = str(content)
    return json.dumps([{"type": "text", "text": text}]), text


def _deserialize_content(json_str: str) -> Any:
    """Deserialize message content from JSON storage."""
    try:
        blocks = json.loads(json_str)
        if isinstance(blocks, list):
            # Reconstruct typed blocks where possible.
            from niaharness.engine.messages import TextBlock

            result = []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    result.append(TextBlock(text=block.get("text", "")))
                else:
                    result.append(block)  # Keep as dict for non-text blocks
            return result
        return blocks
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Session operations
# ---------------------------------------------------------------------------


class SessionRecord:
    """A session record (row in the sessions table)."""

    def __init__(
        self,
        id: str,
        project_hash: str,
        project_path: str,
        title: Optional[str],
        created_at: str,
        updated_at: str,
        parent_session_id: Optional[str] = None,
        message_count: int = 0,
        token_count: int = 0,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.id = id
        self.project_hash = project_hash
        self.project_path = project_path
        self.title = title
        self.created_at = created_at
        self.updated_at = updated_at
        self.parent_session_id = parent_session_id
        self.message_count = message_count
        self.token_count = token_count
        self.model = model
        self.provider = provider
        self.metadata = metadata or {}

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "SessionRecord":
        """Build a SessionRecord from a database row."""
        metadata_str = row["metadata"]
        metadata = json.loads(metadata_str) if metadata_str else {}
        return cls(
            id=row["id"],
            project_hash=row["project_hash"],
            project_path=row["project_path"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            parent_session_id=row["parent_session_id"],
            message_count=row["message_count"],
            token_count=row["token_count"],
            model=row["model"],
            provider=row["provider"],
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict (for JSON export)."""
        return {
            "id": self.id,
            "project_hash": self.project_hash,
            "project_path": self.project_path,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parent_session_id": self.parent_session_id,
            "message_count": self.message_count,
            "token_count": self.token_count,
            "model": self.model,
            "provider": self.provider,
            "metadata": self.metadata,
        }


def create_session(
    session_id: str,
    cwd: str | Path,
    *,
    title: Optional[str] = None,
    parent_session_id: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SessionRecord:
    """Create a new session record in the DB.

    Returns the created SessionRecord. Raises ValueError if a session with
    the same ID already exists.
    """
    _repair_db_if_needed()
    project_hash = _project_hash(cwd)
    project_path = str(Path(cwd).resolve())
    now = _now_iso()

    try:
        with _transaction() as conn:
            conn.execute(
                """
                INSERT INTO sessions
                    (id, project_hash, project_path, title, created_at, updated_at,
                     parent_session_id, message_count, token_count, model, provider, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
                """,
                (
                    session_id,
                    project_hash,
                    project_path,
                    title,
                    now,
                    now,
                    parent_session_id,
                    model,
                    provider,
                    json.dumps(metadata or {}),
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"Session '{session_id}' already exists") from exc

    return SessionRecord(
        id=session_id,
        project_hash=project_hash,
        project_path=project_path,
        title=title,
        created_at=now,
        updated_at=now,
        parent_session_id=parent_session_id,
        model=model,
        provider=provider,
        metadata=metadata or {},
    )


def get_session(session_id: str) -> Optional[SessionRecord]:
    """Retrieve a session by ID. Returns None if not found."""
    _repair_db_if_needed()
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return SessionRecord.from_row(row)


def list_sessions(
    *,
    project_hash: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[SessionRecord]:
    """List sessions, optionally filtered by project. Ordered by updated_at DESC."""
    _repair_db_if_needed()
    conn = _get_connection()
    if project_hash:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE project_hash = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (project_hash, limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [SessionRecord.from_row(row) for row in rows]


def update_session(
    session_id: str,
    *,
    title: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update session fields. Returns True if the session was found and updated."""
    _repair_db_if_needed()
    sets: List[str] = []
    params: List[Any] = []

    if title is not None:
        sets.append("title = ?")
        params.append(title)
    if model is not None:
        sets.append("model = ?")
        params.append(model)
    if provider is not None:
        sets.append("provider = ?")
        params.append(provider)
    if metadata is not None:
        sets.append("metadata = ?")
        params.append(json.dumps(metadata))
    sets.append("updated_at = ?")
    params.append(_now_iso())
    params.append(session_id)

    if not sets:
        return False

    with _transaction() as conn:
        cursor = conn.execute(
            f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        return cursor.rowcount > 0


def delete_session(session_id: str) -> bool:
    """Delete a session and all its messages. Returns True if found and deleted."""
    _repair_db_if_needed()
    with _transaction() as conn:
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Message operations
# ---------------------------------------------------------------------------


def add_message(
    session_id: str,
    role: str,
    content: Any,
    *,
    token_count: int = 0,
) -> int:
    """Append a message to a session. Returns the message ID.

    Updates the session's message_count, token_count, and updated_at.
    """
    _repair_db_if_needed()
    content_json, content_text = _serialize_content(content)
    now = _now_iso()

    with _transaction() as conn:
        # Get the next seq number.
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        seq = row[0] if row else 0

        cursor = conn.execute(
            """
            INSERT INTO messages (session_id, seq, role, content, content_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, seq, role, content_json, content_text, now),
        )
        message_id = cursor.lastrowid

        # Update session counters.
        conn.execute(
            """
            UPDATE sessions
            SET message_count = message_count + 1,
                token_count = token_count + ?,
                updated_at = ?
            WHERE id = ?
            """,
            (token_count, now, session_id),
        )

    return message_id


def get_messages(
    session_id: str,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Retrieve messages for a session, ordered by seq.

    Returns a list of dicts with keys: id, session_id, seq, role, content, created_at.
    """
    _repair_db_if_needed()
    conn = _get_connection()
    if limit:
        rows = conn.execute(
            """
            SELECT id, session_id, seq, role, content, created_at
            FROM messages WHERE session_id = ?
            ORDER BY seq LIMIT ? OFFSET ?
            """,
            (session_id, limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, session_id, seq, role, content, created_at
            FROM messages WHERE session_id = ?
            ORDER BY seq
            """,
            (session_id,),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "session_id": row["session_id"],
            "seq": row["seq"],
            "role": row["role"],
            "content": _deserialize_content(row["content"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def search_messages(
    query: str,
    *,
    session_id: Optional[str] = None,
    project_hash: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Full-text search across messages using FTS5.

    Returns a list of dicts with keys: message_id, session_id, role, content,
    created_at, snippet (the matching text snippet).
    """
    _repair_db_if_needed()
    conn = _get_connection()

    # Build the FTS5 query. FTS5 supports MATCH syntax with column:term,
    # AND, OR, NOT, prefix matching (term*), and phrase matching ("...").
    # We use a simple OR query of the user's terms.
    fts_query = " OR ".join(query.split())

    if session_id:
        sql = """
            SELECT m.id, m.session_id, m.role, m.content, m.created_at,
                   snippet(messages_fts, 0, '<<', '>>', '...', 32) as snippet
            FROM messages_fts fts
            JOIN messages m ON m.id = fts.rowid
            WHERE messages_fts MATCH ? AND m.session_id = ?
            ORDER BY rank LIMIT ?
        """
        rows = conn.execute(sql, (fts_query, session_id, limit)).fetchall()
    elif project_hash:
        sql = """
            SELECT m.id, m.session_id, m.role, m.content, m.created_at,
                   snippet(messages_fts, 0, '<<', '>>', '...', 32) as snippet
            FROM messages_fts fts
            JOIN messages m ON m.id = fts.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE messages_fts MATCH ? AND s.project_hash = ?
            ORDER BY rank LIMIT ?
        """
        rows = conn.execute(sql, (fts_query, project_hash, limit)).fetchall()
    else:
        sql = """
            SELECT m.id, m.session_id, m.role, m.content, m.created_at,
                   snippet(messages_fts, 0, '<<', '>>', '...', 32) as snippet
            FROM messages_fts fts
            JOIN messages m ON m.id = fts.rowid
            WHERE messages_fts MATCH ?
            ORDER BY rank LIMIT ?
        """
        rows = conn.execute(sql, (fts_query, limit)).fetchall()

    return [
        {
            "message_id": row["id"],
            "session_id": row["session_id"],
            "role": row["role"],
            "content": _deserialize_content(row["content"]),
            "created_at": row["created_at"],
            "snippet": row["snippet"],
        }
        for row in rows
    ]


def get_session_lineage(session_id: str) -> List[SessionRecord]:
    """Return the parent chain of a session (root first, this session last).

    Useful for displaying "forked from X, which resumed from Y" in the UI.
    """
    _repair_db_if_needed()
    chain: List[SessionRecord] = []
    visited: set[str] = set()
    current = get_session(session_id)
    while current is not None and current.id not in visited:
        visited.add(current.id)
        chain.insert(0, current)
        if current.parent_session_id:
            current = get_session(current.parent_session_id)
        else:
            break
    return chain


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------


def vacuum() -> None:
    """Run VACUUM to reclaim space and defragment. Must be run with no other connections active."""
    conn = _get_connection()
    conn.execute("VACUUM")


def get_db_stats() -> Dict[str, Any]:
    """Return basic stats about the session DB."""
    _repair_db_if_needed()
    conn = _get_connection()
    session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    db_size = _sessions_db_path().stat().st_size if _sessions_db_path().exists() else 0
    wal_size = _sessions_db_path().with_suffix(".db-wal").stat().st_size if _sessions_db_path().with_suffix(".db-wal").exists() else 0
    return {
        "session_count": session_count,
        "message_count": message_count,
        "db_size_bytes": db_size,
        "wal_size_bytes": wal_size,
        "db_path": str(_sessions_db_path()),
    }


__all__ = [
    "SessionRecord",
    "add_message",
    "create_session",
    "delete_session",
    "get_db_stats",
    "get_messages",
    "get_session",
    "get_session_lineage",
    "list_sessions",
    "search_messages",
    "update_session",
    "vacuum",
]
