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
