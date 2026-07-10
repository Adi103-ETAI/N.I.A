"""SQLite session storage with WAL mode — ported from Hermes Agent's hermes_state.py.

Full 45-column sessions schema, 18-column messages schema, declarative
column reconciliation, surgical DB repair, write-contention tuning with
jitter retry, trigram FTS5 for CJK search.

Key patterns ported from Hermes:
  - ``_reconcile_columns()`` — declarative schema migration (in-memory
    SCHEMA_SQL parse + ALTER TABLE ADD COLUMN on diff). Adding a column
    to SCHEMA_SQL is the only change needed; the reconciler picks it up
    automatically on next startup.
  - ``repair_state_db_schema()`` — 3-strategy surgical repair that
    preserves sessions/messages rows (FTS rebuild → sqlite_master dedup
    → drop FTS + VACUUM). Replaces NIA's data-losing rename-and-reinit.
  - ``_db_opens_cleanly()`` — rolled-back write probe that catches FTS
    trigger corruption (reads pass, writes fail) that integrity_check
    misses.
  - ``_execute_write()`` — BEGIN IMMEDIATE + random jitter retry
    (20-150ms) with 15 retries. Breaks the convoy pattern that SQLite's
    deterministic backoff creates under multi-process contention.
  - Trigram FTS5 table for CJK substring search.
  - WAL checkpoint (TRUNCATE) every 50 writes; FTS optimize every 1000.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Constants — write-contention tuning (ported from Hermes)
# ---------------------------------------------------------------------------

_WRITE_MAX_RETRIES = 15
_WRITE_RETRY_MIN_S = 0.020   # 20ms
_WRITE_RETRY_MAX_S = 0.150   # 150ms
_CHECKPOINT_EVERY_N_WRITES = 50
_OPTIMIZE_EVERY_N_WRITES = 1000
_BUSY_TIMEOUT_S = 1.0  # Short — application-level retry handles contention

_repair_attempt_lock = threading.Lock()
_repair_attempted_paths: set[str] = set()


# ---------------------------------------------------------------------------
# Schema — 45-column sessions + 18-column messages (ported from Hermes)
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'cli',
    user_id TEXT,
    session_key TEXT,
    chat_id TEXT,
    chat_type TEXT,
    thread_id TEXT,
    display_name TEXT,
    origin_json TEXT,
    expiry_finalized INTEGER DEFAULT 0,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    project_hash TEXT,
    project_path TEXT,
    git_branch TEXT,
    git_repo_root TEXT,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    api_call_count INTEGER DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    compression_failure_cooldown_until REAL,
    compression_failure_error TEXT,
    rewind_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    metadata TEXT,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    seq INTEGER,
    platform_message_id TEXT,
    observed INTEGER DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS gateway_routing (
    scope TEXT NOT NULL DEFAULT '',
    session_key TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (scope, session_key)
);

CREATE TABLE IF NOT EXISTS compression_locks (
    session_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_source_id ON sessions(source, id);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_compression_locks_expires ON compression_locks(expires_at);
"""

# Indexes that reference columns added by _reconcile_columns on legacy DBs.
DEFERRED_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_messages_session_active
    ON messages(session_id, active, timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_session_key
    ON sessions(session_key, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_gateway_peer
    ON sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_handoff_state
    ON sessions(handoff_state, started_at);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""

FTS_TRIGRAM_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""


# ---------------------------------------------------------------------------
# DB repair functions (ported from Hermes hermes_state.py)
# ---------------------------------------------------------------------------


def _claim_repair_attempt(db_path: Path) -> bool:
    """Claim the one-shot repair attempt for *db_path* in this process."""
    key = str(db_path)
    with _repair_attempt_lock:
        if key in _repair_attempted_paths:
            return False
        _repair_attempted_paths.add(key)
        return True


def _backup_db_file(db_path: Path) -> Optional[Path]:
    """Copy a (possibly malformed) DB file to a timestamped backup."""
    import shutil

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.malformed-backup-{stamp}")
    try:
        shutil.copy2(db_path, backup_path)
        for suffix in ("-wal", "-shm"):
            sidecar = db_path.with_name(db_path.name + suffix)
            if sidecar.exists():
                shutil.copy2(sidecar, backup_path.with_name(backup_path.name + suffix))
        return backup_path
    except Exception as exc:
        logger.warning("Could not back up malformed DB %s: %s", db_path, exc)
        return None


def _db_opens_cleanly(db_path: Path) -> Optional[str]:
    """Probe a DB on a fresh connection. Returns None if healthy, else a reason.

    Ported from Hermes. Runs PRAGMA journal_mode + integrity_check + a
    rolled-back messages write probe that catches FTS trigger corruption
    (reads pass, writes fail) that integrity_check misses.
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode").fetchone()
        rows = conn.execute("PRAGMA integrity_check").fetchall()
        problems = [str(r[0]) for r in rows if r and str(r[0]).lower() != "ok"]
        if problems:
            return "; ".join(problems[:3])
        conn.execute("SELECT COUNT(*) FROM sessions").fetchone()

        # FTS write probe: drive a row through the messages_fts* triggers in a
        # transaction that is always rolled back, so a corrupt FTS index that
        # rejects writes is caught even though reads look healthy.
        probe_session_id = f"_nia_fts_health_probe_{time.time_ns()}"
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                (probe_session_id, "_health_probe", time.time()),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (probe_session_id, "user", "_fts_health_probe", time.time()),
            )
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            msg = str(exc).lower()
            if "no such table" in msg or "no such column" in msg:
                return None
            return str(exc)
        return None
    except sqlite3.DatabaseError as exc:
        return str(exc)
    finally:
        conn.close()


def repair_state_db_schema(db_path: Path, *, backup: bool = True) -> Dict[str, Any]:
    """Repair a state.db whose schema is malformed or whose FTS rejects writes.

    Ported from Hermes. 3 strategies, least-destructive first:
      1. Rebuild FTS indexes in place (FTS5 'rebuild' command)
      2. De-duplicate sqlite_master (keep lowest rowid per type/name)
      3. Drop all FTS schema + VACUUM (rebuilds on next open)

    Canonical sessions/messages rows are never modified. A timestamped
    raw backup is taken first unless backup=False.
    """
    report: Dict[str, Any] = {
        "repaired": False,
        "strategy": None,
        "backup_path": None,
        "error": None,
    }

    db_path = Path(db_path)
    if not db_path.exists():
        report["error"] = f"{db_path} does not exist"
        return report

    if _db_opens_cleanly(db_path) is None:
        report["repaired"] = True
        report["strategy"] = "already_healthy"
        return report

    if backup:
        bpath = _backup_db_file(db_path)
        report["backup_path"] = str(bpath) if bpath else None

    # Strategy 0: rebuild FTS indexes in place
    try:
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            for table_name in ("messages_fts", "messages_fts_trigram"):
                try:
                    conn.execute(
                        f"INSERT INTO {table_name}({table_name}) VALUES('rebuild')"
                    )
                except sqlite3.OperationalError:
                    continue
        finally:
            conn.close()
        if _db_opens_cleanly(db_path) is None:
            report["repaired"] = True
            report["strategy"] = "rebuild_fts"
            logger.warning("state.db FTS indexes rebuilt in place: %s", db_path)
            return report
    except sqlite3.DatabaseError as exc:
        logger.warning("state.db FTS in-place rebuild failed: %s", exc)

    # Strategy 1: de-duplicate sqlite_master
    try:
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            conn.execute("PRAGMA writable_schema=ON")
            dupes = conn.execute(
                "SELECT type, name, COUNT(*) AS c, MIN(rowid) AS keep "
                "FROM sqlite_master GROUP BY type, name HAVING c > 1"
            ).fetchall()
            for type_, name, _count, keep in dupes:
                conn.execute(
                    "DELETE FROM sqlite_master "
                    "WHERE type IS ? AND name IS ? AND rowid <> ?",
                    (type_, name, keep),
                )
            conn.execute("PRAGMA writable_schema=OFF")
            conn.commit()
        finally:
            conn.close()
        if _db_opens_cleanly(db_path) is None:
            report["repaired"] = True
            report["strategy"] = "dedup_schema"
            logger.warning("state.db schema repaired by dedup: %s", db_path)
            return report
    except sqlite3.DatabaseError as exc:
        logger.warning("state.db dedup repair failed: %s", exc)

    # Strategy 2: drop all FTS schema + VACUUM
    try:
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            conn.execute("PRAGMA writable_schema=ON")
            conn.execute("DELETE FROM sqlite_master WHERE name LIKE 'messages_fts%'")
            conn.execute("PRAGMA writable_schema=OFF")
            conn.commit()
            conn.execute("VACUUM")
        finally:
            conn.close()
        reason = _db_opens_cleanly(db_path)
        if reason is None:
            report["repaired"] = True
            report["strategy"] = "drop_fts_rebuild"
            logger.warning("state.db FTS schema dropped; will rebuild on next open: %s", db_path)
            return report
        report["error"] = reason
    except sqlite3.DatabaseError as exc:
        report["error"] = str(exc)

    if not report["repaired"]:
        logger.error(
            "state.db repair could not recover %s automatically (backup: %s); "
            "manual restore may be required.",
            db_path, report["backup_path"],
        )
    return report


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _sessions_db_path() -> Path:
    """Return the path to the sessions SQLite DB."""
    from niaharness.prompts.soul import get_nia_home

    return get_nia_home() / "sessions.db"


def _project_hash(cwd: str | Path) -> str:
    """Return a 16-char SHA-256 hash of the resolved absolute cwd."""
    resolved = Path(cwd).resolve()
    return hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# SessionDB class (ported from Hermes SessionDB)
# ---------------------------------------------------------------------------


class SessionDB:
    """SQLite-backed session storage with FTS5 search.

    Ported from Hermes Agent's hermes_state.py SessionDB class. Provides:
      - 45-column sessions table (token counts, tool counts, billing, etc.)
      - 18-column messages table (tool calls, reasoning, compaction state)
      - Declarative column reconciliation (add column to SCHEMA_SQL → done)
      - Surgical DB repair (FTS rebuild → dedup → drop FTS, never data loss)
      - Write-contention tuning (BEGIN IMMEDIATE + jitter retry)
      - Trigram FTS5 for CJK substring search
      - WAL checkpoint every 50 writes; FTS optimize every 1000
    """

    def __init__(self, db_path: Path | None = None, *, read_only: bool = False):
        self.db_path = Path(db_path) if db_path else _sessions_db_path()
        self.read_only = read_only
        self._lock = threading.Lock()
        self._write_count = 0
        self._fts_enabled = False
        self._trigram_available = False
        self._conn: Optional[sqlite3.Connection] = None

        try:
            if read_only:
                self._conn = sqlite3.connect(
                    f"file:{self.db_path}?mode=ro",
                    uri=True,
                    check_same_thread=False,
                    timeout=_BUSY_TIMEOUT_S,
                    isolation_level=None,
                )
                self._conn.row_factory = sqlite3.Row
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            def _connect_and_init():
                self._conn = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                    timeout=_BUSY_TIMEOUT_S,
                    isolation_level=None,
                )
                self._conn.row_factory = sqlite3.Row
                # Enable WAL mode for concurrent read/write.
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute(f"PRAGMA busy_timeout={int(_BUSY_TIMEOUT_S * 1000)}")
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute("PRAGMA temp_store=MEMORY")
                self._init_schema()

            try:
                _connect_and_init()
            except sqlite3.DatabaseError as exc:
                if not _claim_repair_attempt(self.db_path):
                    raise
                logger.error(
                    "sessions.db schema is malformed (%s) — attempting repair "
                    "(backup made first).", exc,
                )
                try:
                    if self._conn is not None:
                        self._conn.close()
                except Exception:
                    pass
                report = repair_state_db_schema(self.db_path)
                if not report.get("repaired"):
                    raise
                _connect_and_init()
        except Exception as exc:
            logger.error("SessionDB init failed: %s", exc)
            raise

    # ── Schema management ──────────────────────────────────────────────

    def _init_schema(self) -> None:
        """Create tables + FTS, reconcile columns, set schema version."""
        cursor = self._conn.cursor()
        cursor.executescript(SCHEMA_SQL)
        self._reconcile_columns(cursor)
        cursor.executescript(DEFERRED_INDEX_SQL)

        # FTS5 availability check
        fts5_available = self._sqlite_supports_fts5(cursor)
        if fts5_available:
            cursor.executescript(FTS_SQL)
            self._fts_enabled = True
            # Trigram tokenizer (optional — not all SQLite builds have it)
            try:
                cursor.executescript(FTS_TRIGRAM_SQL)
                self._trigram_available = True
            except sqlite3.OperationalError:
                self._trigram_available = False
                logger.debug("Trigram FTS5 tokenizer not available — CJK search degraded")

        # Schema version
        cursor.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            cursor.execute("INSERT INTO schema_version (version) VALUES (1)")
        else:
            cursor.execute("UPDATE schema_version SET version = 1")

    @staticmethod
    def _sqlite_supports_fts5(cursor: sqlite3.Cursor) -> bool:
        """Check if the SQLite build supports FTS5."""
        try:
            cursor.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS temp._nia_fts5_probe USING fts5(x)"
            )
            cursor.execute("DROP TABLE temp._nia_fts5_probe")
            return True
        except sqlite3.OperationalError:
            return False

    @staticmethod
    def _parse_schema_columns(schema_sql: str) -> Dict[str, Dict[str, str]]:
        """Extract expected columns per table from SCHEMA_SQL.

        Uses an in-memory SQLite DB to parse the DDL — handles all syntax
        (DEFAULT expressions, inline REFERENCES, CHECK constraints, etc.)
        with zero regex edge cases.
        """
        ref = sqlite3.connect(":memory:")
        try:
            ref.executescript(schema_sql)
            table_columns: Dict[str, Dict[str, str]] = {}
            for (tbl,) in ref.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall():
                cols: Dict[str, str] = {}
                for row in ref.execute(f'PRAGMA table_info("{tbl}")').fetchall():
                    col_name = row[1]
                    col_type = row[2] or ""
                    notnull = row[3]
                    default = row[4]
                    pk = row[5]
                    parts = [col_type] if col_type else []
                    if notnull and not pk:
                        parts.append("NOT NULL")
                    if default is not None:
                        parts.append(f"DEFAULT {default}")
                    cols[col_name] = " ".join(parts)
                table_columns[tbl] = cols
            return table_columns
        finally:
            ref.close()

    def _reconcile_columns(self, cursor: sqlite3.Cursor) -> None:
        """Ensure live tables have every column declared in SCHEMA_SQL.

        Declarative migration: diff live columns (PRAGMA table_info) against
        declared columns, ADD any missing ones. Adding a column to SCHEMA_SQL
        is the only change needed — the reconciler picks it up automatically.
        """
        expected = self._parse_schema_columns(SCHEMA_SQL)
        for table_name, declared_cols in expected.items():
            try:
                rows = cursor.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            live_cols = set()
            for row in rows:
                name = row[1] if isinstance(row, (tuple, list)) else row["name"]
                live_cols.add(name)

            for col_name, col_type in declared_cols.items():
                if col_name not in live_cols:
                    safe_name = col_name.replace('"', '""')
                    try:
                        cursor.execute(
                            f'ALTER TABLE "{table_name}" ADD COLUMN "{safe_name}" {col_type}'
                        )
                        logger.info("Reconciled column: %s.%s", table_name, col_name)
                    except sqlite3.OperationalError as exc:
                        logger.debug("reconcile %s.%s: %s", table_name, col_name, exc)

    # ── Write helper ───────────────────────────────────────────────────

    def _execute_write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Execute a write transaction with BEGIN IMMEDIATE and jitter retry.

        Ported from Hermes. On ``database is locked``, releases the Python
        lock, sleeps random 20-150ms, retries — breaking the convoy pattern.
        """
        last_err: Optional[Exception] = None
        for attempt in range(_WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        try:
                            self._conn.rollback()
                        except Exception:
                            pass
                        raise
                self._write_count += 1
                if self._write_count % _CHECKPOINT_EVERY_N_WRITES == 0:
                    self._try_wal_checkpoint()
                if self._write_count % _OPTIMIZE_EVERY_N_WRITES == 0:
                    self._try_optimize_fts()
                return result
            except sqlite3.OperationalError as exc:
                err_msg = str(exc).lower()
                if "locked" in err_msg or "busy" in err_msg:
                    last_err = exc
                    if attempt < _WRITE_MAX_RETRIES - 1:
                        jitter = random.uniform(_WRITE_RETRY_MIN_S, _WRITE_RETRY_MAX_S)
                        time.sleep(jitter)
                        continue
                raise
        raise last_err or sqlite3.OperationalError("database is locked after max retries")

    def _try_wal_checkpoint(self) -> None:
        """Best-effort TRUNCATE WAL checkpoint. Never raises."""
        try:
            with self._lock:
                result = self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if result and result[1] > 0:
                    logger.debug("WAL checkpoint: %d/%d pages", result[2], result[1])
        except Exception:
            pass

    def _try_optimize_fts(self) -> None:
        """Best-effort FTS5 segment merge. Never raises."""
        try:
            with self._lock:
                for table in ("messages_fts", "messages_fts_trigram"):
                    try:
                        self._conn.execute(f'INSERT INTO {table}({table}) VALUES("optimize")')
                    except sqlite3.OperationalError:
                        pass
        except Exception:
            pass

    def close(self) -> None:
        """Close the connection. Attempts WAL checkpoint first."""
        with self._lock:
            if self._conn:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception:
                    pass
                self._conn.close()
                self._conn = None

    # ── Session CRUD ───────────────────────────────────────────────────

    def create_session(
        self,
        session_id: str,
        *,
        cwd: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        source: str = "cli",
        user_id: str | None = None,
        session_key: str | None = None,
        chat_id: str | None = None,
        parent_session_id: str | None = None,
        system_prompt: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Create a new session record."""
        now = time.time()
        project_hash = _project_hash(cwd) if cwd else None
        project_path = str(Path(cwd).resolve()) if cwd else None

        def _insert(conn: sqlite3.Connection) -> dict:
            conn.execute(
                """INSERT INTO sessions
                   (id, source, user_id, session_key, chat_id, model,
                    billing_provider,
                    parent_session_id, started_at, project_hash, project_path,
                    system_prompt, title, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, source, user_id, session_key, chat_id,
                    model, provider, parent_session_id, now,
                    project_hash, project_path, system_prompt, title,
                    json.dumps(metadata or {}),
                ),
            )
            return {"id": session_id, "started_at": now}

        return self._execute_write(_insert)

    def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve a session by ID."""
        conn = self._conn
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_sessions(
        self,
        *,
        source: str | None = None,
        project_hash: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False,
    ) -> List[dict]:
        """List sessions, ordered by started_at DESC."""
        query = "SELECT * FROM sessions WHERE 1=1"
        params: list[Any] = []
        if not include_archived:
            query += " AND archived = 0"
        if source:
            query += " AND source = ?"
            params.append(source)
        if project_hash:
            query += " AND project_hash = ?"
            params.append(project_hash)
        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def update_session(
        self,
        session_id: str,
        **kwargs: Any,
    ) -> bool:
        """Update session fields. Returns True if found and updated.

        Accepts any column name as a keyword argument (e.g.,
        update_session(id, title="New Title", input_tokens=1000)).
        """
        if not kwargs:
            return False

        allowed = {
            "title", "model", "billing_provider", "billing_base_url", "billing_mode",
            "system_prompt", "ended_at",
            "end_reason", "message_count", "tool_call_count", "input_tokens",
            "output_tokens", "cache_read_tokens", "cache_write_tokens",
            "reasoning_tokens", "estimated_cost_usd", "actual_cost_usd",
            "cost_status", "cost_source", "pricing_version", "api_call_count",
            "handoff_state", "handoff_platform", "handoff_error",
            "compression_failure_cooldown_until", "compression_failure_error",
            "rewind_count", "archived", "metadata", "display_name",
            "expiry_finalized", "session_key", "chat_id", "chat_type",
            "thread_id", "user_id", "git_branch", "git_repo_root",
        }
        updates: list[str] = []
        params: list[Any] = []
        for key, value in kwargs.items():
            if key not in allowed:
                continue
            if key == "metadata" and isinstance(value, dict):
                value = json.dumps(value)
            updates.append(f"{key} = ?")
            params.append(value)

        if not updates:
            return False

        params.append(session_id)
        sql = f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?"

        def _do_update(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(sql, params)
            return cursor.rowcount > 0

        return self._execute_write(_do_update)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        def _delete(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cursor.rowcount > 0
        return self._execute_write(_delete)

    # ── Message CRUD ───────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        tool_call_id: str | None = None,
        tool_calls: str | None = None,
        tool_name: str | None = None,
        token_count: int | None = None,
        finish_reason: str | None = None,
        reasoning: str | None = None,
        reasoning_content: str | None = None,
        seq: int | None = None,
    ) -> int:
        """Append a message to a session. Returns the message ID.

        Also updates the session's message_count, token_count, and
        tool_call_count.
        """
        now = time.time()

        def _insert(conn: sqlite3.Connection) -> int:
            # Get next seq if not provided
            if seq is None:
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                next_seq = row[0] if row else 0
            else:
                next_seq = seq

            cursor = conn.execute(
                """INSERT INTO messages
                   (session_id, role, content, tool_call_id, tool_calls, tool_name,
                    timestamp, token_count, finish_reason, reasoning,
                    reasoning_content, seq)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, role, content, tool_call_id, tool_calls, tool_name,
                    now, token_count, finish_reason, reasoning, reasoning_content,
                    next_seq,
                ),
            )
            message_id = cursor.lastrowid

            # Update session counters
            updates = ["message_count = message_count + 1"]
            params: list[Any] = []
            if token_count is not None:
                updates.append("input_tokens = input_tokens + ?" if role == "user" else "output_tokens = output_tokens + ?")
                params.append(token_count)
            if tool_name is not None:
                updates.append("tool_call_count = tool_call_count + 1")
            params.append(session_id)
            conn.execute(
                f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            return message_id

        return self._execute_write(_insert)

    def get_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        include_compacted: bool = False,
    ) -> List[dict]:
        """Retrieve messages for a session, ordered by seq/timestamp."""
        query = "SELECT * FROM messages WHERE session_id = ?"
        if not include_compacted:
            query += " AND compacted = 0"
        query += " ORDER BY seq"
        if limit:
            query += " LIMIT ? OFFSET ?"
            rows = self._conn.execute(query, (session_id, limit, offset)).fetchall()
        else:
            rows = self._conn.execute(query, (session_id,)).fetchall()
        return [dict(row) for row in rows]

    def search_messages(
        self,
        query: str,
        *,
        session_id: str | None = None,
        source: str | None = None,
        project_hash: str | None = None,
        limit: int = 20,
        use_trigram: bool = True,
    ) -> List[dict]:
        """Full-text search across messages using FTS5.

        Uses trigram FTS5 for CJK substring search if available,
        falls back to porter+unicode61 FTS5.
        """
        fts_table = "messages_fts_trigram" if (use_trigram and self._trigram_available) else "messages_fts"
        if not self._fts_enabled:
            return []

        # Build FTS5 query
        fts_query = " ".join(query.split())

        sql = f"""
            SELECT m.id, m.session_id, m.role, m.content, m.tool_name,
                   m.timestamp,
                   snippet({fts_table}, 0, '<<', '>>', '...', 32) as snippet
            FROM {fts_table} fts
            JOIN messages m ON m.id = fts.rowid
        """
        params: list[Any] = [fts_query]
        where = [f"{fts_table} MATCH ?"]

        if session_id:
            sql += " JOIN sessions s ON s.id = m.session_id"
            where.append("m.session_id = ?")
            params.append(session_id)
        if source:
            where.append("s.source = ?")
            params.append(source)
        if project_hash:
            where.append("s.project_hash = ?")
            params.append(project_hash)

        sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        try:
            rows = self._conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.OperationalError as exc:
            logger.warning("FTS search failed: %s", exc)
            return []

    def get_session_lineage(self, session_id: str) -> List[dict]:
        """Return the parent chain of a session (root first, this session last)."""
        chain: List[dict] = []
        visited: set[str] = set()
        current = self.get_session(session_id)
        while current is not None and current["id"] not in visited:
            visited.add(current["id"])
            chain.insert(0, current)
            parent_id = current.get("parent_session_id")
            if parent_id:
                current = self.get_session(parent_id)
            else:
                break
        return chain

    # ── Stats ──────────────────────────────────────────────────────────

    def get_db_stats(self) -> Dict[str, Any]:
        """Return basic stats about the session DB."""
        try:
            session_count = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            message_count = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        except sqlite3.OperationalError:
            return {"session_count": 0, "message_count": 0, "db_path": str(self.db_path)}

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        wal_path = self.db_path.with_suffix(".db-wal")
        wal_size = wal_path.stat().st_size if wal_path.exists() else 0
        return {
            "session_count": session_count,
            "message_count": message_count,
            "db_size_bytes": db_size,
            "wal_size_bytes": wal_size,
            "db_path": str(self.db_path),
            "fts_enabled": self._fts_enabled,
            "trigram_available": self._trigram_available,
        }

    def vacuum(self) -> None:
        """Run VACUUM to reclaim space and defragment."""
        with self._lock:
            self._conn.execute("VACUUM")

    def optimize_fts(self) -> None:
        """Merge FTS5 segments for better query performance."""
        with self._lock:
            for table in ("messages_fts", "messages_fts_trigram"):
                try:
                    self._conn.execute(f'INSERT INTO {table}({table}) VALUES("optimize")')
                except sqlite3.OperationalError:
                    pass

    # ------------------------------------------------------------------
    # Compaction cooldown persistence (durable across restarts)
    # ------------------------------------------------------------------

    def record_compression_failure_cooldown(
        self,
        session_id: str,
        cooldown_until: float,
        error: Optional[str] = None,
    ) -> None:
        """Persist a compaction failure cooldown for a session.

        Args:
            session_id: The session that hit a compaction failure.
            cooldown_until: Wall-clock epoch seconds when the cooldown expires.
            error: Optional error message describing the failure.
        """
        import time as _time

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE sessions SET "
                "  compression_failure_cooldown_until = ?, "
                "  compression_failure_error = ? "
                "WHERE id = ?",
                (cooldown_until, error, session_id),
            )
            conn.commit()

        try:
            self._execute_write(_do)
        except Exception as exc:
            logger.debug("record_compression_failure_cooldown failed: %s", exc)

    def get_compression_failure_cooldown(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the active cooldown for a session, or None if expired/missing.

        Returns:
            Dict with ``cooldown_until``, ``remaining_seconds``, ``error``
            if an active cooldown exists; ``None`` otherwise.
        """
        import time as _time

        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT compression_failure_cooldown_until, compression_failure_error "
                    "FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
        except Exception as exc:
            logger.debug("get_compression_failure_cooldown failed: %s", exc)
            return None

        if row is None:
            return None

        cooldown_until = row[0] if row[0] else 0
        error = row[1] if len(row) > 1 else None

        if not cooldown_until:
            return None

        now = _time.time()
        remaining = cooldown_until - now
        if remaining <= 0:
            return None

        return {
            "cooldown_until": cooldown_until,
            "remaining_seconds": remaining,
            "error": error,
        }

    def clear_compression_failure_cooldown(self, session_id: str) -> None:
        """Clear the compaction failure cooldown for a session."""
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE sessions SET "
                "  compression_failure_cooldown_until = NULL, "
                "  compression_failure_error = NULL "
                "WHERE id = ?",
                (session_id,),
            )
            conn.commit()

        try:
            self._execute_write(_do)
        except Exception as exc:
            logger.debug("clear_compression_failure_cooldown failed: %s", exc)

    # ------------------------------------------------------------------
    # Compaction locks (cross-process mutex via SQLite)
    # ------------------------------------------------------------------

    def try_acquire_compression_lock(
        self,
        session_id: str,
        holder: str,
        ttl_seconds: float = 300.0,
    ) -> bool:
        """Try to acquire a compression lock for a session.

        Args:
            session_id: The session to lock.
            holder: A unique identifier for the lock holder (e.g. PID + thread ID).
            ttl_seconds: Lock auto-expires after this many seconds.

        Returns:
            True if the lock was acquired, False if another holder has it.
        """
        import time as _time

        now = _time.time()
        expires_at = now + ttl_seconds

        def _do(conn: sqlite3.Connection) -> bool:
            # Check for existing lock.
            row = conn.execute(
                "SELECT holder, expires_at FROM compression_locks WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is not None:
                existing_holder, existing_expires = row[0], row[1]
                if existing_expires and existing_expires > now and existing_holder != holder:
                    return False  # Active lock by another holder.
                # Expired or same holder — overwrite.
                conn.execute(
                    "UPDATE compression_locks SET holder = ?, acquired_at = ?, expires_at = ? "
                    "WHERE session_id = ?",
                    (holder, now, expires_at, session_id),
                )
            else:
                conn.execute(
                    "INSERT INTO compression_locks (session_id, holder, acquired_at, expires_at) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, holder, now, expires_at),
                )
            conn.commit()
            return True

        try:
            return self._execute_write(_do)
        except Exception as exc:
            logger.debug("try_acquire_compression_lock failed: %s", exc)
            return False

    def refresh_compression_lock(
        self,
        session_id: str,
        holder: str,
        ttl_seconds: float = 300.0,
    ) -> bool:
        """Refresh an existing compression lock (extend its TTL).

        Returns True if the lock was refreshed, False if it was taken by another holder.
        """
        import time as _time

        now = _time.time()
        expires_at = now + ttl_seconds

        def _do(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT holder FROM compression_locks WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None or row[0] != holder:
                return False
            conn.execute(
                "UPDATE compression_locks SET acquired_at = ?, expires_at = ? "
                "WHERE session_id = ? AND holder = ?",
                (now, expires_at, session_id, holder),
            )
            conn.commit()
            return True

        try:
            return self._execute_write(_do)
        except Exception as exc:
            logger.debug("refresh_compression_lock failed: %s", exc)
            return False

    def release_compression_lock(self, session_id: str) -> None:
        """Release the compression lock for a session."""
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "DELETE FROM compression_locks WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()

        try:
            self._execute_write(_do)
        except Exception as exc:
            logger.debug("release_compression_lock failed: %s", exc)

    def get_compression_lock_holder(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the current lock holder for a session, or None.

        Returns:
            Dict with ``holder``, ``acquired_at``, ``expires_at`` if an
            active lock exists; ``None`` otherwise.
        """
        import time as _time

        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT holder, acquired_at, expires_at "
                    "FROM compression_locks WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
        except Exception:
            return None

        if row is None:
            return None

        holder, acquired_at, expires_at = row[0], row[1], row[2]
        if expires_at and expires_at <= _time.time():
            return None  # Expired.

        return {
            "holder": holder,
            "acquired_at": acquired_at,
            "expires_at": expires_at,
        }

    # ------------------------------------------------------------------
    # Session lifecycle (end / reopen / ensure)
    # ------------------------------------------------------------------

    def end_session(self, session_id: str, reason: str = "completed") -> bool:
        """Mark a session as ended with a reason.

        Args:
            session_id: The session to end.
            reason: Why the session ended (e.g. "completed", "session_reset",
                "agent_close", "cron_complete").

        Returns:
            True if the session was found and updated.
        """
        import time as _time

        def _do(conn: sqlite3.Connection) -> bool:
            conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
                (_time.time(), reason, session_id),
            )
            conn.commit()
            return conn.total_changes > 0

        try:
            return self._execute_write(_do)
        except Exception as exc:
            logger.debug("end_session failed: %s", exc)
            return False

    def reopen_session(self, session_id: str) -> bool:
        """Reopen a previously-ended session (clears end_reason + ended_at).

        Used by /resume and session recovery to continue a conversation
        that was previously closed.

        Returns:
            True if the session was found and reopened.
        """
        def _do(conn: sqlite3.Connection) -> bool:
            conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                (session_id,),
            )
            conn.commit()
            return conn.total_changes > 0

        try:
            return self._execute_write(_do)
        except Exception as exc:
            logger.debug("reopen_session failed: %s", exc)
            return False

    def ensure_session(self, session_id: str, **kwargs: Any) -> dict:
        """Upsert a session — create if it doesn't exist, update if it does.

        Args:
            session_id: The session ID.
            **kwargs: Fields to set (source, model, started_at, etc.).

        Returns:
            The session dict.
        """
        existing = self.get_session(session_id)
        if existing:
            if kwargs:
                self.update_session(session_id, **kwargs)
            return self.get_session(session_id) or existing
        return self.create_session(session_id=session_id, **kwargs)

    def is_session_ended(self, session_id: str) -> bool:
        """True if the session has an end_reason set (i.e. it's been ended)."""
        session = self.get_session(session_id)
        if session is None:
            return True  # Doesn't exist → treat as ended.
        return bool(session.get("end_reason"))

    # ------------------------------------------------------------------
    # Title management
    # ------------------------------------------------------------------

    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set the human-readable title for a session."""
        title = (title or "").strip()[:200]  # Cap at 200 chars.

        def _do(conn: sqlite3.Connection) -> bool:
            conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id),
            )
            conn.commit()
            return conn.total_changes > 0

        try:
            return self._execute_write(_do)
        except Exception as exc:
            logger.debug("set_session_title failed: %s", exc)
            return False

    def get_session_title(self, session_id: str) -> Optional[str]:
        """Return the title for a session, or None."""
        session = self.get_session(session_id)
        if session is None:
            return None
        return session.get("title")

    def resolve_session_id(self, prefix: str) -> Optional[str]:
        """Resolve a session ID prefix to a full session ID.

        If *prefix* is already a full ID, returns it. Otherwise, finds
        the unique session whose ID starts with *prefix*.

        Returns:
            The full session ID, or None if no match / ambiguous.
        """
        if not prefix:
            return None
        # Exact match.
        session = self.get_session(prefix)
        if session:
            return session["id"]
        # Prefix match.
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id FROM sessions WHERE id LIKE ? ORDER BY started_at DESC",
                    (prefix + "%",),
                ).fetchall()
        except Exception:
            return None
        if len(rows) == 1:
            return rows[0]["id"]
        return None

    # ------------------------------------------------------------------
    # Message operations (replace / rewind / dedup)
    # ------------------------------------------------------------------

    def replace_messages(self, session_id: str, messages: list[dict]) -> bool:
        """Hard-replace all messages for a session.

        Used by /retry, /undo, /compress — deletes all existing messages
        and inserts the new list.

        Args:
            session_id: The session to replace messages for.
            messages: List of message dicts with role, content, etc.

        Returns:
            True on success.
        """
        import time as _time

        def _do(conn: sqlite3.Connection) -> bool:
            # Delete existing messages.
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            # Insert new messages.
            for i, msg in enumerate(messages):
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, tool_call_id, "
                    "tool_calls, tool_name, timestamp, seq) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        msg.get("role", "unknown"),
                        msg.get("content", ""),
                        msg.get("tool_call_id"),
                        msg.get("tool_calls"),
                        msg.get("tool_name"),
                        msg.get("timestamp", _time.time()),
                        i,
                    ),
                )
            # Update message count.
            conn.execute(
                "UPDATE sessions SET message_count = ? WHERE id = ?",
                (len(messages), session_id),
            )
            conn.commit()
            return True

        try:
            return self._execute_write(_do)
        except Exception as exc:
            logger.warning("replace_messages failed: %s", exc)
            return False

    def has_platform_message_id(
        self, session_id: str, platform_message_id: str
    ) -> bool:
        """Check if a platform message ID has already been persisted (dedup guard).

        Used by the gateway to prevent duplicate message processing on
        transient failures.
        """
        if not platform_message_id:
            return False
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT 1 FROM messages WHERE session_id = ? AND platform_message_id = ? LIMIT 1",
                    (session_id, platform_message_id),
                ).fetchone()
            return row is not None
        except Exception:
            return False

    def get_messages_as_conversation(
        self, session_id: str
    ) -> list[dict[str, Any]]:
        """Return messages in OpenAI conversation format.

        Each message has: role, content. Assistant messages also have
        tool_calls if present.
        """
        messages = self.get_messages(session_id)
        result: list[dict[str, Any]] = []
        for msg in messages:
            entry: dict[str, Any] = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            }
            if msg.get("tool_calls"):
                entry["tool_calls"] = msg["tool_calls"]
            if msg.get("tool_call_id"):
                entry["tool_call_id"] = msg["tool_call_id"]
            result.append(entry)
        return result

    def list_recent_user_messages(
        self, session_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Return the most recent user messages for /undo."""
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id, content, timestamp FROM messages "
                    "WHERE session_id = ? AND role = 'user' "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Meta store (key/value)
    # ------------------------------------------------------------------

    def get_meta(self, key: str) -> Optional[str]:
        """Get a value from the state_meta key/value store."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT value FROM state_meta WHERE key = ?", (key,)
                ).fetchone()
            return row["value"] if row else None
        except Exception:
            return None

    def set_meta(self, key: str, value: str) -> None:
        """Set a value in the state_meta key/value store."""
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            conn.commit()

        try:
            self._execute_write(_do)
        except Exception as exc:
            logger.debug("set_meta failed: %s", exc)

    # ==================================================================
    # P1: Gateway routing (ported from Hermes hermes_state.py)
    # ==================================================================

    def record_gateway_session_peer(
        self,
        session_id: str,
        *,
        source: str,
        user_id: str | None = None,
        session_key: str | None = None,
        chat_id: str | None = None,
        chat_type: str | None = None,
        thread_id: str | None = None,
        display_name: str | None = None,
        origin_json: str | None = None,
    ) -> None:
        """Persist the gateway routing peer for an existing session row."""
        if not session_id or not session_key:
            return

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                """UPDATE sessions
                   SET session_key = ?, source = ?, user_id = ?, chat_id = ?,
                       chat_type = ?, thread_id = ?,
                       display_name = COALESCE(?, display_name),
                       origin_json = COALESCE(?, origin_json)
                   WHERE id = ?""",
                (
                    session_key, source, user_id, chat_id,
                    chat_type, thread_id, display_name, origin_json,
                    session_id,
                ),
            )
        self._execute_write(_do)

    def set_expiry_finalized(self, session_id: str, finalized: bool = True) -> None:
        """Mark a gateway session's expiry-finalization flag."""
        if not session_id:
            return

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE sessions SET expiry_finalized = ? WHERE id = ?",
                (1 if finalized else 0, session_id),
            )
        self._execute_write(_do)

    def save_gateway_routing_entry(
        self, session_key: str, entry_json: str, *, scope: str = ""
    ) -> None:
        """Upsert one gateway routing entry (session_key -> entry JSON)."""
        if not session_key or not entry_json:
            return

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(scope, session_key) DO UPDATE SET
                       entry_json = excluded.entry_json,
                       updated_at = excluded.updated_at""",
                (scope, session_key, entry_json, time.time()),
            )
        self._execute_write(_do)

    def replace_gateway_routing_entries(
        self, entries: Dict[str, str], *, scope: str = ""
    ) -> None:
        """Atomically replace the routing index for *scope*."""
        now = time.time()

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "DELETE FROM gateway_routing WHERE scope = ?", (scope,)
            )
            if entries:
                conn.executemany(
                    "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    [(scope, k, v, now) for k, v in entries.items() if k and v],
                )
        self._execute_write(_do)

    def load_gateway_routing_entries(self, *, scope: str = "") -> Dict[str, str]:
        """Load routing entries for *scope* as {session_key: entry_json}."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_key, entry_json FROM gateway_routing WHERE scope = ?",
                (scope,),
            ).fetchall()
        return {
            (r["session_key"] if isinstance(r, sqlite3.Row) else r[0]):
            (r["entry_json"] if isinstance(r, sqlite3.Row) else r[1])
            for r in rows
        }

    def delete_gateway_routing_entries(
        self, session_keys: List[str], *, scope: str = ""
    ) -> None:
        """Remove routing entries for the given session keys in *scope*."""
        if not session_keys:
            return

        def _do(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "DELETE FROM gateway_routing WHERE scope = ? AND session_key = ?",
                [(scope, k) for k in session_keys],
            )
        self._execute_write(_do)

    def list_gateway_sessions(
        self,
        *,
        platform: str | None = None,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """List gateway sessions (rows with a session_key). Returns newest per key."""
        query = """
            SELECT sessions.*,
                   COALESCE(
                       (SELECT MAX(m.timestamp) FROM messages m
                        WHERE m.session_id = sessions.id),
                       sessions.started_at
                   ) AS last_active
            FROM sessions
            WHERE session_key IS NOT NULL
              AND started_at = (
                  SELECT MAX(s2.started_at) FROM sessions s2
                  WHERE s2.session_key = sessions.session_key
              )
        """
        params: list = []
        if platform:
            query += " AND LOWER(source) = LOWER(?)"
            params.append(platform)
        if active_only:
            query += " AND ended_at IS NULL"
        query += " ORDER BY last_active DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def find_session_by_origin(
        self,
        *,
        platform: str,
        chat_id: str,
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> Optional[str]:
        """Find the most recent live session_id for a platform + chat origin."""
        if not platform or chat_id in (None, ""):
            return None
        query = """
            SELECT id, user_id, started_at FROM sessions
            WHERE LOWER(source) = LOWER(?)
              AND session_key IS NOT NULL
              AND chat_id = ?
              AND ended_at IS NULL
        """
        params: list = [platform, str(chat_id)]
        if thread_id is not None:
            query += " AND COALESCE(thread_id, '') = ?"
            params.append(str(thread_id))
        query += " ORDER BY started_at DESC"
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(query, params).fetchall()]
        if not rows:
            return None
        if user_id:
            exact = [r for r in rows if str(r.get("user_id") or "") == str(user_id)]
            if exact:
                return str(exact[0]["id"])
            if len(rows) > 1:
                return None
        elif len(rows) > 1:
            distinct_users = {
                str(r.get("user_id") or "").strip()
                for r in rows
                if str(r.get("user_id") or "").strip()
            }
            if len(distinct_users) > 1:
                return None
        return str(rows[0]["id"])

    def find_latest_gateway_session_for_peer(
        self,
        *,
        source: str,
        user_id: str | None = None,
        session_key: str | None = None,
        chat_id: str | None = None,
        chat_type: str | None = None,
        thread_id: str | None = None,
    ) -> Optional[Dict[str, Any]]:
        """Find the latest recoverable gateway session for a routing peer."""
        if not session_key:
            return None
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM sessions
                   WHERE session_key = ? AND source = ?
                     AND (ended_at IS NULL OR end_reason = 'agent_close')
                     AND (COALESCE(message_count, 0) > 0 OR EXISTS (
                         SELECT 1 FROM messages WHERE messages.session_id = sessions.id LIMIT 1
                     ))
                   ORDER BY started_at DESC LIMIT 1""",
                (session_key, source),
            ).fetchone()
            if row is not None:
                return dict(row)
            if chat_id is None or chat_type is None:
                return None
            row = self._conn.execute(
                """SELECT * FROM sessions
                   WHERE source = ?
                     AND COALESCE(user_id, '') = COALESCE(?, '')
                     AND COALESCE(chat_id, '') = COALESCE(?, '')
                     AND COALESCE(chat_type, '') = COALESCE(?, '')
                     AND COALESCE(thread_id, '') = COALESCE(?, '')
                     AND (ended_at IS NULL OR end_reason = 'agent_close')
                   ORDER BY started_at DESC LIMIT 1""",
                (source, user_id or "", chat_id, chat_type, thread_id or ""),
            ).fetchone()
            return dict(row) if row is not None else None

    # ==================================================================
    # P1: Handoff (cross-platform session transfer)
    # ==================================================================

    def request_handoff(self, session_id: str, platform: str) -> bool:
        """Mark a session as pending handoff to the given platform.

        Returns True if the row was found and not already in flight.
        """
        def _do(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                "UPDATE sessions SET handoff_state = 'pending', "
                "handoff_platform = ?, handoff_error = NULL "
                "WHERE id = ? AND (handoff_state IS NULL "
                "                  OR handoff_state IN ('completed', 'failed'))",
                (platform, session_id),
            )
            return cur.rowcount > 0
        return self._execute_write(_do)

    def get_handoff_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Read the current handoff state for a session."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT handoff_state, handoff_platform, handoff_error "
                    "FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
            if not row:
                return None
            return {
                "state": row["handoff_state"] if isinstance(row, sqlite3.Row) else row[0],
                "platform": row["handoff_platform"] if isinstance(row, sqlite3.Row) else row[1],
                "error": row["handoff_error"] if isinstance(row, sqlite3.Row) else row[2],
            }
        except Exception:
            return None

    def list_pending_handoffs(self) -> List[Dict[str, Any]]:
        """Return all sessions in handoff_state='pending', oldest first."""
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM sessions WHERE handoff_state = 'pending' "
                    "ORDER BY started_at ASC"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def claim_handoff(self, session_id: str) -> bool:
        """Atomically transition pending → running. Returns True if claimed."""
        def _do(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                "UPDATE sessions SET handoff_state = 'running' "
                "WHERE id = ? AND handoff_state = 'pending'",
                (session_id,),
            )
            return cur.rowcount > 0
        return self._execute_write(_do)

    def complete_handoff(self, session_id: str) -> None:
        """Mark a handoff as completed."""
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE sessions SET handoff_state = 'completed', "
                "handoff_error = NULL WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    def fail_handoff(self, session_id: str, error: str) -> None:
        """Mark a handoff as failed and record the reason."""
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE sessions SET handoff_state = 'failed', "
                "handoff_error = ? WHERE id = ?",
                (error[:500], session_id),
            )
        self._execute_write(_do)

    # ==================================================================
    # P1: Message operations (archive_and_compact, rewind, restore, around)
    # ==================================================================

    def _insert_message_rows(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """Insert a batch of messages, return (message_count, tool_call_count)."""
        if not messages:
            return 0, 0
        now = time.time()
        rows = []
        tool_calls = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, (dict, list)):
                content = json.dumps(content)
            tool_calls_str = msg.get("tool_calls")
            if tool_calls_str and isinstance(tool_calls_str, (dict, list)):
                tool_calls_str = json.dumps(tool_calls_str)
                tool_calls += 1
            elif tool_calls_str:
                tool_calls += 1
            rows.append((
                session_id,
                msg.get("role", "user"),
                content,
                msg.get("tool_call_id"),
                tool_calls_str,
                msg.get("tool_name"),
                msg.get("timestamp", now),
                msg.get("token_count"),
                msg.get("finish_reason"),
                msg.get("reasoning"),
                msg.get("reasoning_content"),
                msg.get("reasoning_details"),
                msg.get("seq"),
                msg.get("platform_message_id"),
                msg.get("observed", 0),
                msg.get("active", 1),
                msg.get("compacted", 0),
            ))
        conn.executemany(
            """INSERT INTO messages (
                session_id, role, content, tool_call_id, tool_calls, tool_name,
                timestamp, token_count, finish_reason, reasoning,
                reasoning_content, reasoning_details, seq,
                platform_message_id, observed, active, compacted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        return len(rows), tool_calls

    def has_archived_messages(self, session_id: str) -> bool:
        """Return True if the session has any soft-archived (active=0) rows."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM messages WHERE session_id = ? AND active = 0 LIMIT 1",
                (session_id,),
            ).fetchone()
        return row is not None

    def archive_and_compact(
        self, session_id: str, compacted_messages: List[Dict[str, Any]]
    ) -> int:
        """Non-destructive in-place compaction. Soft-archives active messages
        and inserts compacted_messages as fresh active rows. Returns new active count.
        """
        def _do(conn: sqlite3.Connection) -> int:
            conn.execute(
                "UPDATE messages SET active = 0, compacted = 1 "
                "WHERE session_id = ? AND active = 1",
                (session_id,),
            )
            inserted, tool_calls_total = self._insert_message_rows(
                conn, session_id, compacted_messages
            )
            conn.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                (inserted, tool_calls_total, session_id),
            )
            return inserted
        return self._execute_write(_do)

    def get_messages_around(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
    ) -> Dict[str, Any]:
        """Load a window of messages anchored on a specific message id."""
        with self._lock:
            anchor_row = self._conn.execute(
                "SELECT * FROM messages WHERE id = ? AND session_id = ?",
                (around_message_id, session_id),
            ).fetchone()
        if anchor_row is None:
            return {"before": [], "anchor": None, "after": []}
        anchor = dict(anchor_row)

        with self._lock:
            before_rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id < ? AND active = 1 "
                "ORDER BY id DESC LIMIT ?",
                (session_id, around_message_id, window),
            ).fetchall()
            after_rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id > ? AND active = 1 "
                "ORDER BY id ASC LIMIT ?",
                (session_id, around_message_id, window),
            ).fetchall()
        return {
            "before": list(reversed([dict(r) for r in before_rows])),
            "anchor": anchor,
            "after": [dict(r) for r in after_rows],
        }

    def rewind_to_message(
        self, session_id: str, target_message_id: int
    ) -> Dict[str, Any]:
        """Soft-delete all messages with id >= target_message_id.

        Returns {"rewound_count", "target_message", "new_head_id"}.
        Raises ValueError if the target doesn't exist or isn't a user message.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM messages WHERE id = ? AND session_id = ?",
                (target_message_id, session_id),
            ).fetchone()
        if row is None:
            raise ValueError(
                f"message {target_message_id} not found in session {session_id}"
            )
        target_row = dict(row)
        if target_row.get("role") != "user":
            raise ValueError(
                f"rewind target must be a 'user' message (got role="
                f"{target_row.get('role')!r})"
            )

        def _do(conn: sqlite3.Connection) -> List[int]:
            cursor = conn.execute(
                "SELECT id FROM messages "
                "WHERE session_id = ? AND id >= ? AND active = 1",
                (session_id, target_message_id),
            )
            ids = [r[0] for r in cursor.fetchall()]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"UPDATE messages SET active = 0 WHERE id IN ({placeholders})",
                    ids,
                )
            conn.execute(
                "UPDATE sessions SET rewind_count = COALESCE(rewind_count, 0) + 1 "
                "WHERE id = ?",
                (session_id,),
            )
            return ids

        rewound = self._execute_write(_do)

        with self._lock:
            head_row = self._conn.execute(
                "SELECT MAX(id) FROM messages WHERE session_id = ? AND active = 1",
                (session_id,),
            ).fetchone()
        new_head_id = head_row[0] if head_row and head_row[0] is not None else None

        return {
            "rewound_count": len(rewound),
            "target_message": target_row,
            "new_head_id": new_head_id,
        }

    def restore_rewound(self, session_id: str, since_message_id: int) -> int:
        """Mark inactive messages with id >= since_message_id active again."""
        def _do(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "SELECT id FROM messages "
                "WHERE session_id = ? AND id >= ? AND active = 0",
                (session_id, since_message_id),
            )
            ids = [r[0] for r in cursor.fetchall()]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"UPDATE messages SET active = 1 WHERE id IN ({placeholders})",
                    ids,
                )
            return len(ids)
        return self._execute_write(_do)

    def clear_messages(self, session_id: str) -> None:
        """Delete all messages for a session."""
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    def resolve_resume_session_id(self, session_id: str) -> str:
        """Resolve a session id to its latest compression continuation.

        If the session has children (compression continuations), follow the
        chain forward to the tip. Otherwise return the input id unchanged.
        """
        current = session_id
        visited: set[str] = set()
        while current and current not in visited:
            visited.add(current)
            with self._lock:
                row = self._conn.execute(
                    "SELECT id FROM sessions WHERE parent_session_id = ? "
                    "ORDER BY started_at ASC LIMIT 1",
                    (current,),
                ).fetchone()
            if row is None:
                break
            current = str(row["id"]) if isinstance(row, sqlite3.Row) else str(row[0])
        return current

    # ==================================================================
    # P1: Session meta + update methods
    # ==================================================================

    def update_session_meta(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """Merge metadata into a session's existing metadata JSON."""
        with self._lock:
            row = self._conn.execute(
                "SELECT metadata FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return False
        existing = {}
        raw = row["metadata"] if isinstance(row, sqlite3.Row) else row[0]
        if raw:
            try:
                existing = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                existing = {}
        existing.update(metadata)

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE sessions SET metadata = ? WHERE id = ?",
                (json.dumps(existing), session_id),
            )
        self._execute_write(_do)
        return True

    def update_system_prompt(self, session_id: str, system_prompt: str) -> None:
        """Update the system prompt for a session."""
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE sessions SET system_prompt = ? WHERE id = ?",
                (system_prompt, session_id),
            )
        self._execute_write(_do)

    def update_session_model(self, session_id: str, model: str) -> None:
        """Update the model for a session."""
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE sessions SET model = ? WHERE id = ?",
                (model, session_id),
            )
        self._execute_write(_do)

    def update_session_billing_route(
        self,
        session_id: str,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        mode: str | None = None,
    ) -> None:
        """Update billing route fields for a session."""
        sets: list[str] = []
        params: list[Any] = []
        if provider is not None:
            sets.append("billing_provider = ?")
            params.append(provider)
        if base_url is not None:
            sets.append("billing_base_url = ?")
            params.append(base_url)
        if mode is not None:
            sets.append("billing_mode = ?")
            params.append(mode)
        if not sets:
            return
        params.append(session_id)

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?",
                params,
            )
        self._execute_write(_do)

    def update_token_counts(
        self,
        session_id: str,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        reasoning_tokens: int | None = None,
    ) -> None:
        """Update token count fields for a session (additive on each call)."""
        sets: list[str] = []
        params: list[Any] = []
        for col, val in [
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
            ("cache_read_tokens", cache_read_tokens),
            ("cache_write_tokens", cache_write_tokens),
            ("reasoning_tokens", reasoning_tokens),
        ]:
            if val is not None and val > 0:
                sets.append(f"{col} = COALESCE({col}, 0) + ?")
                params.append(val)
        if not sets:
            return
        params.append(session_id)

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?",
                params,
            )
        self._execute_write(_do)

    def update_session_cwd(
        self, session_id: str, cwd: str, *, git_branch: str | None = None
    ) -> None:
        """Update the cwd (and optionally git_branch) for a session."""
        sets = ["project_path = ?"]
        params: list[Any] = [cwd]
        if git_branch is not None:
            sets.append("git_branch = ?")
            params.append(git_branch)
        params.append(session_id)

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?",
                params,
            )
        self._execute_write(_do)

    def backfill_repo_roots(self, cwd_to_root: Dict[str, str]) -> None:
        """Backfill git_repo_root for sessions matching cwd_to_root keys."""
        if not cwd_to_root:
            return

        def _do(conn: sqlite3.Connection) -> None:
            for cwd, root in cwd_to_root.items():
                conn.execute(
                    "UPDATE sessions SET git_repo_root = ? "
                    "WHERE project_path = ? AND git_repo_root IS NULL",
                    (root, cwd),
                )
        self._execute_write(_do)

    def set_session_archived(self, session_id: str, archived: bool) -> bool:
        """Set the archived flag for a session. Returns True if found."""
        def _do(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                "UPDATE sessions SET archived = ? WHERE id = ?",
                (1 if archived else 0, session_id),
            )
            return cur.rowcount > 0
        return self._execute_write(_do)

    # ==================================================================
    # P1: Title methods
    # ==================================================================

    @staticmethod
    def sanitize_title(title: str | None) -> str | None:
        """Sanitize a title for storage. Strips control chars, truncates."""
        if not title:
            return None
        # Strip control characters except newlines/tabs.
        cleaned = "".join(
            c for c in title if c == "\n" or c == "\t" or (ord(c) >= 32)
        )
        cleaned = cleaned.strip()
        if not cleaned:
            return None
        return cleaned[:200]  # cap at 200 chars

    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Get a session by exact title match."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE title = ? ORDER BY started_at DESC LIMIT 1",
                (title,),
            ).fetchone()
        return dict(row) if row is not None else None

    def resolve_session_by_title(self, title: str) -> Optional[str]:
        """Resolve a title to a session id. Returns None if not found."""
        session = self.get_session_by_title(title)
        return session["id"] if session else None

    def get_next_title_in_lineage(self, base_title: str) -> str:
        """Find the next available title with a numeric suffix in the lineage.

        e.g. if "Project Chat" exists, returns "Project Chat (1)".
        If "Project Chat (1)" exists, returns "Project Chat (2)".
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT title FROM sessions WHERE title LIKE ? ORDER BY title",
                (f"{base_title}%",),
            ).fetchall()
        existing = {
            r["title"] if isinstance(r, sqlite3.Row) else r[0]
            for r in row
        }
        if base_title not in existing:
            return base_title
        i = 1
        while f"{base_title} ({i})" in existing:
            i += 1
        return f"{base_title} ({i})"

    def get_compression_tip(self, session_id: str) -> Optional[str]:
        """Walk forward through compression children to find the tip session id."""
        return self.resolve_resume_session_id(session_id)

    # ==================================================================
    # P1: Rich listing + pruning
    # ==================================================================

    def distinct_session_cwds(
        self, include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        """Distinct non-empty session cwds with usage stats."""
        where = "project_path IS NOT NULL AND TRIM(project_path) != ''"
        if not include_archived:
            where += " AND archived = 0"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT project_path AS cwd, COUNT(*) AS sessions, "
                f"MAX(COALESCE(ended_at, started_at, 0)) AS last_active "
                f"FROM sessions WHERE {where} GROUP BY project_path"
            ).fetchall()
        return [
            {
                "cwd": r["cwd"] if isinstance(r, sqlite3.Row) else r[0],
                "sessions": int((r["sessions"] if isinstance(r, sqlite3.Row) else r[1]) or 0),
                "last_active": float(
                    (r["last_active"] if isinstance(r, sqlite3.Row) else r[2]) or 0
                ),
            }
            for r in rows
        ]

    def list_sessions_rich(
        self,
        *,
        source: str | None = None,
        exclude_sources: List[str] | None = None,
        cwd_prefix: str | None = None,
        limit: int = 20,
        offset: int = 0,
        min_message_count: int = 0,
        include_archived: bool = False,
        archived_only: bool = False,
        search_query: str | None = None,
    ) -> List[Dict[str, Any]]:
        """List sessions with preview (first user message) and last active.

        Simplified port of Hermes's list_sessions_rich — omits the recursive
        compression-chain CTE (returns raw root rows).
        """
        where_clauses: list[str] = []
        params: list[Any] = []
        if source:
            where_clauses.append("s.source = ?")
            params.append(source)
        if exclude_sources:
            placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({placeholders})")
            params.extend(exclude_sources)
        if cwd_prefix:
            where_clauses.append("s.project_path LIKE ?")
            params.append(f"{cwd_prefix}%")
        if min_message_count > 0:
            where_clauses.append("s.message_count >= ?")
            params.append(min_message_count)
        if archived_only:
            where_clauses.append("s.archived = 1")
        elif not include_archived:
            where_clauses.append("s.archived = 0")
        if search_query:
            needle = f"%{search_query.lower()}%"
            where_clauses.append(
                "(LOWER(COALESCE(s.title, '')) LIKE ? OR LOWER(s.id) LIKE ?)"
            )
            params.extend([needle, needle])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = f"""
            SELECT s.*,
                   (SELECT m.content FROM messages m
                    WHERE m.session_id = s.id AND m.role = 'user'
                    ORDER BY m.id ASC LIMIT 1) AS preview,
                   COALESCE(
                       (SELECT MAX(m.timestamp) FROM messages m
                        WHERE m.session_id = s.id),
                       s.started_at
                   ) AS last_active
            FROM sessions s
            {where_sql}
            ORDER BY s.started_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            preview = d.pop("preview", None)
            if preview:
                d["preview"] = str(preview)[:60]
            else:
                d["preview"] = ""
            result.append(d)
        return result

    def list_cron_job_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List sessions created by the cron scheduler.

        Sessions with source='cron' are cron job runs.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE source = 'cron' "
                "ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def _prune_filter_where(
        self,
        *,
        started_before: float | None = None,
        started_after: float | None = None,
        source: str | None = None,
        title_like: str | None = None,
        end_reason: str | None = None,
        cwd_prefix: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        archived: bool | None = None,
        model_like: str | None = None,
        provider: str | None = None,
        user_id: str | None = None,
        chat_id: str | None = None,
        chat_type: str | None = None,
        branch_like: str | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        min_cost: float | None = None,
        max_cost: float | None = None,
        min_tool_calls: int | None = None,
        max_tool_calls: int | None = None,
    ) -> Tuple[str, list]:
        """Build the shared WHERE clause for bulk prune/archive selection."""
        clauses = ["s.ended_at IS NOT NULL"]
        params: list = []
        if started_before is not None:
            clauses.append("s.started_at < ?")
            params.append(started_before)
        if started_after is not None:
            clauses.append("s.started_at >= ?")
            params.append(started_after)
        if source:
            clauses.append("s.source = ?")
            params.append(source)
        if title_like:
            clauses.append("LOWER(COALESCE(s.title, '')) LIKE ?")
            params.append(f"%{title_like.lower()}%")
        if end_reason:
            clauses.append("s.end_reason = ?")
            params.append(end_reason)
        if cwd_prefix:
            clauses.append("s.project_path LIKE ?")
            params.append(f"{cwd_prefix}%")
        if min_messages is not None:
            clauses.append("s.message_count >= ?")
            params.append(min_messages)
        if max_messages is not None:
            clauses.append("s.message_count <= ?")
            params.append(max_messages)
        if model_like:
            clauses.append("LOWER(COALESCE(s.model, '')) LIKE ?")
            params.append(f"%{model_like.lower()}%")
        if provider:
            clauses.append("LOWER(COALESCE(s.billing_provider, '')) = ?")
            params.append(provider.lower())
        if user_id:
            clauses.append("s.user_id = ?")
            params.append(user_id)
        if chat_id:
            clauses.append("s.chat_id = ?")
            params.append(chat_id)
        if chat_type:
            clauses.append("s.chat_type = ?")
            params.append(chat_type)
        if branch_like:
            clauses.append("LOWER(COALESCE(s.git_branch, '')) LIKE ?")
            params.append(f"%{branch_like.lower()}%")
        if min_tokens is not None:
            clauses.append(
                "(COALESCE(s.input_tokens, 0) + COALESCE(s.output_tokens, 0)) >= ?"
            )
            params.append(min_tokens)
        if max_tokens is not None:
            clauses.append(
                "(COALESCE(s.input_tokens, 0) + COALESCE(s.output_tokens, 0)) <= ?"
            )
            params.append(max_tokens)
        if min_cost is not None:
            clauses.append(
                "COALESCE(s.actual_cost_usd, s.estimated_cost_usd, 0) >= ?"
            )
            params.append(min_cost)
        if max_cost is not None:
            clauses.append(
                "COALESCE(s.actual_cost_usd, s.estimated_cost_usd, 0) <= ?"
            )
            params.append(max_cost)
        if min_tool_calls is not None:
            clauses.append("COALESCE(s.tool_call_count, 0) >= ?")
            params.append(min_tool_calls)
        if max_tool_calls is not None:
            clauses.append("COALESCE(s.tool_call_count, 0) <= ?")
            params.append(max_tool_calls)
        if archived is True:
            clauses.append("s.archived = 1")
        elif archived is False:
            clauses.append("s.archived = 0")
        return " AND ".join(clauses), params

    def list_prune_candidates(
        self,
        older_than_days: float | None = None,
        source: str | None = None,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """Return sessions a matching prune_sessions call would touch."""
        if filters.get("started_before") is None and older_than_days is not None:
            filters["started_before"] = time.time() - (older_than_days * 86400)
        where, params = self._prune_filter_where(source=source, **filters)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT s.id, s.source, s.title, s.model, s.started_at,
                           s.ended_at, s.message_count, s.archived
                    FROM sessions s WHERE {where}
                    ORDER BY s.started_at ASC""",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def archive_sessions(
        self,
        older_than_days: float | None = None,
        source: str | None = None,
        **filters: Any,
    ) -> int:
        """Bulk-archive sessions matching the filters. Returns count archived."""
        filters.setdefault("archived", False)
        rows = self.list_prune_candidates(
            older_than_days=older_than_days, source=source, **filters
        )
        for row in rows:
            self.set_session_archived(row["id"], True)
        return len(rows)

    def prune_sessions(
        self,
        older_than_days: float | None = 90,
        source: str | None = None,
        **filters: Any,
    ) -> int:
        """Delete sessions matching the filters. Returns count deleted."""
        if filters.get("started_before") is None and older_than_days is not None:
            filters["started_before"] = time.time() - (older_than_days * 86400)
        where, where_params = self._prune_filter_where(source=source, **filters)

        def _do(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                f"SELECT s.id FROM sessions s WHERE {where}", where_params
            )
            session_ids = {row[0] for row in cursor.fetchall()}
            if not session_ids:
                return 0
            placeholders = ",".join("?" * len(session_ids))
            conn.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({placeholders})",
                list(session_ids),
            )
            for sid in session_ids:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
            return len(session_ids)
        return self._execute_write(_do)

    def maybe_auto_prune_and_vacuum(
        self,
        retention_days: int = 90,
        min_interval_hours: int = 24,
        vacuum: bool = True,
    ) -> Dict[str, Any]:
        """Idempotent auto-maintenance: prune old sessions + optional VACUUM."""
        result: Dict[str, Any] = {"skipped": False, "pruned": 0, "vacuumed": False}
        try:
            last_raw = self.get_meta("last_auto_prune")
            now = time.time()
            if last_raw:
                try:
                    last_ts = float(last_raw)
                    if now - last_ts < min_interval_hours * 3600:
                        result["skipped"] = True
                        return result
                except (TypeError, ValueError):
                    pass

            pruned = self.prune_sessions(older_than_days=retention_days)
            result["pruned"] = pruned

            if vacuum and pruned > 0:
                try:
                    self.vacuum()
                    result["vacuumed"] = True
                except Exception as exc:
                    logger.warning("state.db VACUUM failed: %s", exc)

            self.set_meta("last_auto_prune", str(now))
            if pruned > 0:
                logger.info(
                    "state.db auto-maintenance: pruned %d session(s) older than %d days%s",
                    pruned, retention_days,
                    " + VACUUM" if result["vacuumed"] else "",
                )
        except Exception as exc:
            logger.warning("state.db auto-maintenance failed: %s", exc)
            result["error"] = str(exc)
        return result

    def count_empty_sessions(self) -> int:
        """Count sessions with zero messages."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE message_count = 0 "
                "OR message_count IS NULL"
            ).fetchone()
        return int(row[0]) if row else 0

    def delete_empty_sessions(self) -> int:
        """Delete sessions with zero messages. Returns count deleted."""
        def _do(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "SELECT id FROM sessions WHERE message_count = 0 "
                "OR message_count IS NULL"
            )
            ids = [row[0] for row in cursor.fetchall()]
            for sid in ids:
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
            return len(ids)
        return self._execute_write(_do)

    def prune_empty_ghost_sessions(self) -> int:
        """Alias for delete_empty_sessions."""
        return self.delete_empty_sessions()

    def finalize_orphaned_compression_sessions(self) -> int:
        """End sessions whose parent has ended (orphaned compression children)."""
        def _do(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                """SELECT s.id FROM sessions s
                   WHERE s.parent_session_id IS NOT NULL
                     AND s.ended_at IS NULL
                     AND EXISTS (
                         SELECT 1 FROM sessions p
                         WHERE p.id = s.parent_session_id AND p.ended_at IS NOT NULL
                     )"""
            )
            ids = [row[0] for row in cursor.fetchall()]
            now = time.time()
            for sid in ids:
                conn.execute(
                    "UPDATE sessions SET ended_at = ?, end_reason = 'parent_ended' "
                    "WHERE id = ?",
                    (now, sid),
                )
            return len(ids)
        return self._execute_write(_do)

    # ==================================================================
    # P1: Export + misc
    # ==================================================================

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a session + all its messages as a dict."""
        session = self.get_session(session_id)
        if session is None:
            return None
        messages = self.get_messages(session_id, include_compacted=True)
        return {"session": session, "messages": messages}

    def export_all(self, source: str | None = None) -> List[Dict[str, Any]]:
        """Export all sessions (optionally filtered by source)."""
        sessions = self.list_sessions(source=source, limit=100000)
        result = []
        for s in sessions:
            export = self.export_session(s["id"])
            if export:
                result.append(export)
        return result

    def session_count(self, *, include_archived: bool = False) -> int:
        """Return the total session count."""
        where = "" if include_archived else " WHERE archived = 0"
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM sessions{where}"
            ).fetchone()
        return int(row[0]) if row else 0

    def message_count(self, session_id: str | None = None) -> int:
        """Return total message count, or count for a specific session."""
        if session_id:
            with self._lock:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
        else:
            with self._lock:
                row = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        return int(row[0]) if row else 0

    def delete_session_if_empty(self, session_id: str) -> bool:
        """Delete a session only if it has zero messages."""
        if self.message_count(session_id) > 0:
            return False
        return self.delete_session(session_id)

    def delete_sessions(self, session_ids: List[str]) -> int:
        """Delete multiple sessions. Returns count deleted."""
        if not session_ids:
            return 0

        def _do(conn: sqlite3.Connection) -> int:
            count = 0
            for sid in session_ids:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                cur = conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
                count += cur.rowcount
            return count
        return self._execute_write(_do)

    def search_sessions_by_id(self, id_query: str) -> List[Dict[str, Any]]:
        """Search sessions by id substring."""
        needle = f"%{id_query}%"
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE id LIKE ? ORDER BY started_at DESC LIMIT 50",
                (needle,),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_sessions(
        self,
        query: str,
        *,
        source: str | None = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search sessions by title or id substring."""
        needle = f"%{query.lower()}%"
        where = "LOWER(COALESCE(s.title, '')) LIKE ? OR LOWER(s.id) LIKE ?"
        params: list[Any] = [needle, needle]
        if source:
            where += " AND s.source = ?"
            params.append(source)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM sessions s WHERE {where} "
                "ORDER BY s.started_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# P1: AsyncSessionDB wrapper
# ---------------------------------------------------------------------------


class AsyncSessionDB:
    """Async wrapper around SessionDB.

    Offloads each call to a thread via asyncio.to_thread so a blocking
    SQLite call never freezes the event loop. Generic forwarder — works
    for any method on the underlying SessionDB.
    """

    def __init__(self, db: "SessionDB") -> None:
        self._db = db

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._db, name)
        if not callable(attr):
            return attr

        async def _offloaded(*args: Any, **kwargs: Any) -> Any:
            import asyncio
            return await asyncio.to_thread(attr, *args, **kwargs)

        return _offloaded

    @property
    def underlying(self) -> "SessionDB":
        """Return the underlying SessionDB instance."""
        return self._db


# ---------------------------------------------------------------------------
# Backward-compatible module-level API
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_connection() -> sqlite3.Connection:
    """Return a thread-local SessionDB connection."""
    db = getattr(_thread_local, "db", None)
    if db is not None:
        return db
    db = SessionDB()
    _thread_local.db = db
    return db


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Context manager for a transaction."""
    db = _get_connection()
    db._conn.execute("BEGIN")
    try:
        yield db._conn
        db._conn.execute("COMMIT")
    except Exception:
        db._conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------------------
# Public API (backward-compatible with old module-level functions)
# ---------------------------------------------------------------------------


def create_session(
    session_id: str,
    cwd: str | None = None,
    *,
    model: str | None = None,
    provider: str | None = None,
    source: str = "cli",
    user_id: str | None = None,
    session_key: str | None = None,
    chat_id: str | None = None,
    parent_session_id: str | None = None,
    system_prompt: str | None = None,
    title: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Create a new session record in the DB."""
    db = _get_connection()
    return db.create_session(
        session_id,
        cwd=cwd, model=model, provider=provider, source=source,
        user_id=user_id, session_key=session_key, chat_id=chat_id,
        parent_session_id=parent_session_id, system_prompt=system_prompt,
        title=title, metadata=metadata,
    )


def get_session(session_id: str) -> Optional[dict]:
    """Retrieve a session by ID. Returns None if not found."""
    db = _get_connection()
    return db.get_session(session_id)


def list_sessions(
    *,
    source: str | None = None,
    project_hash: str | None = None,
    limit: int = 50,
    offset: int = 0,
    include_archived: bool = False,
) -> List[dict]:
    """List sessions, ordered by started_at DESC."""
    db = _get_connection()
    return db.list_sessions(
        source=source, project_hash=project_hash,
        limit=limit, offset=offset, include_archived=include_archived,
    )


def update_session(session_id: str, **kwargs: Any) -> bool:
    """Update session fields. Returns True if found and updated."""
    db = _get_connection()
    return db.update_session(session_id, **kwargs)


def delete_session(session_id: str) -> bool:
    """Delete a session and all its messages."""
    db = _get_connection()
    return db.delete_session(session_id)


def add_message(
    session_id: str,
    role: str,
    content: str,
    *,
    tool_call_id: str | None = None,
    tool_calls: str | None = None,
    tool_name: str | None = None,
    token_count: int | None = None,
    finish_reason: str | None = None,
    reasoning: str | None = None,
    reasoning_content: str | None = None,
) -> int:
    """Append a message to a session. Returns the message ID."""
    db = _get_connection()
    return db.add_message(
        session_id, role, content,
        tool_call_id=tool_call_id, tool_calls=tool_calls, tool_name=tool_name,
        token_count=token_count, finish_reason=finish_reason,
        reasoning=reasoning, reasoning_content=reasoning_content,
    )


def get_messages(
    session_id: str,
    *,
    limit: int | None = None,
    offset: int = 0,
    include_compacted: bool = False,
) -> List[dict]:
    """Retrieve messages for a session, ordered by seq."""
    db = _get_connection()
    return db.get_messages(
        session_id, limit=limit, offset=offset,
        include_compacted=include_compacted,
    )


def search_messages(
    query: str,
    *,
    session_id: str | None = None,
    source: str | None = None,
    project_hash: str | None = None,
    limit: int = 20,
) -> List[dict]:
    """Full-text search across messages using FTS5."""
    db = _get_connection()
    return db.search_messages(
        query, session_id=session_id, source=source,
        project_hash=project_hash, limit=limit,
    )


def get_session_lineage(session_id: str) -> List[dict]:
    """Return the parent chain of a session (root first)."""
    db = _get_connection()
    return db.get_session_lineage(session_id)


def get_db_stats() -> Dict[str, Any]:
    """Return basic stats about the session DB."""
    try:
        db = _get_connection()
        return db.get_db_stats()
    except Exception:
        return {"session_count": 0, "message_count": 0, "db_path": str(_sessions_db_path())}


def vacuum() -> None:
    """Run VACUUM to reclaim space."""
    db = _get_connection()
    db.vacuum()


__all__ = [
    "AsyncSessionDB",
    "SessionDB",
    "SCHEMA_SQL",
    "FTS_SQL",
    "FTS_TRIGRAM_SQL",
    "DEFERRED_INDEX_SQL",
    "add_message",
    "create_session",
    "delete_session",
    "get_db_stats",
    "get_messages",
    "get_session",
    "get_session_lineage",
    "list_sessions",
    "repair_state_db_schema",
    "search_messages",
    "update_session",
    "vacuum",
]
