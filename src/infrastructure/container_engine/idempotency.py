"""Sandbox Idempotency Layer — crash-safe re-execution guard.

When the Coordinator resumes after a crash, it re-dispatches subagents from
the last checkpoint.  Without idempotency, sandbox commands that already ran
would execute a second time, causing duplicate side-effects (double writes,
double installs, etc.).

``IdempotentSandbox`` wraps :class:`StaticSandbox` and records every
executed command in an SQLite checkpoint table keyed by a caller-provided
**idempotency key** (a UUID assigned per tool-call by the planner).  On
re-execution with the same key the cached result is returned instantly.

Usage::

    from src.infrastructure.container_engine.idempotency import (
        get_idempotent_sandbox,
    )

    sandbox = get_idempotent_sandbox()
    result = await sandbox.execute(
        command="pip install requests",
        idempotency_key="a1b2c3d4-...",
        manifest_id="mission-007",
    )
    assert isinstance(result, SandboxResult)
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
from pydantic import BaseModel, Field

logger = logging.getLogger("N.I.A.IdempotentSandbox")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_DIR = Path("data")
_DEFAULT_DB_PATH = _DATA_DIR / "sandbox_checkpoints.db"
_OUTPUT_MAX_BYTES = 50 * 1024  # 50 KB cap on stored output


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class SandboxResult(BaseModel):
    """Structured result returned by every idempotent sandbox execution."""

    exit_code: int = Field(description="Process exit code (0 = success)")
    output: str = Field(description="Stdout/stderr captured from the command")
    idempotency_key: str = Field(description="Caller-supplied unique key for this invocation")
    manifest_id: str = Field(description="Mission manifest that owns this execution")
    cached: bool = Field(
        default=False,
        description="True when the result was served from checkpoint, not re-executed",
    )
    executed_at: str = Field(description="ISO-8601 timestamp of original execution")


# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS sandbox_checkpoints (
    idempotency_key TEXT PRIMARY KEY,
    manifest_id     TEXT    NOT NULL,
    command         TEXT    NOT NULL,
    exit_code       INTEGER NOT NULL,
    output          TEXT    NOT NULL,
    executed_at     TEXT    NOT NULL
);
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_manifest_id
    ON sandbox_checkpoints(manifest_id);
"""


# ---------------------------------------------------------------------------
# IdempotentSandbox
# ---------------------------------------------------------------------------

class IdempotentSandbox:
    """Wraps :class:`StaticSandbox` with checkpoint-based idempotency.

    Uses an SQLite database to log executed commands by idempotency key.
    On re-execution with the same key, returns the cached result instead
    of running the command again.
    """

    _instance: Optional[IdempotentSandbox] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH

        # Ensure parent directory exists
        os.makedirs(str(self._db_path.parent), exist_ok=True)

        # Create schema synchronously (safe at init time, matching
        # MemoryManager._init_sql_sync pattern).
        self._init_db_sync()

        logger.debug("IdempotentSandbox initialised (db=%s)", self._db_path)

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> IdempotentSandbox:
        """Return (or create) the process-wide singleton."""
        if cls._instance is None:
            with cls._instance_lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = IdempotentSandbox()
        return cls._instance

    # ------------------------------------------------------------------
    # DB initialisation (sync)
    # ------------------------------------------------------------------

    def _init_db_sync(self) -> None:
        """Create the checkpoint table and index if they don't exist."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(_CREATE_TABLE)
                conn.execute(_CREATE_INDEX)
            logger.debug("Checkpoint table ready: %s", self._db_path)
        except Exception as exc:
            logger.error("Failed to initialise checkpoint DB: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def execute(
        self,
        command: str,
        idempotency_key: str,
        manifest_id: str,
        timeout: int = 120,
    ) -> SandboxResult:
        """Execute *command* in the sandbox, honouring idempotency.

        1. If *idempotency_key* already exists in the checkpoint DB the
           cached result is returned immediately (``cached=True``).
        2. Otherwise the command is delegated to
           :meth:`StaticSandbox.execute`, the result is persisted, and a
           fresh :class:`SandboxResult` is returned (``cached=False``).
        """
        # --- Fast path: already executed ---
        cached = await self.get_cached_result(idempotency_key)
        if cached is not None:
            logger.info(
                "Idempotency hit for key=%s — returning cached result",
                idempotency_key,
            )
            return cached

        # --- Slow path: delegate to StaticSandbox ---
        # Lazy import to avoid circular dependency at module level.
        from src.infrastructure.container_engine.sandbox import StaticSandbox

        sandbox = StaticSandbox.get_instance()
        exit_code, output = await sandbox.execute(command, timeout=timeout)

        executed_at = datetime.now(timezone.utc).isoformat()

        # Truncate oversized output before persisting
        stored_output = self._truncate_output(output)

        # Persist checkpoint
        await self._store_checkpoint(
            idempotency_key=idempotency_key,
            manifest_id=manifest_id,
            command=command,
            exit_code=exit_code,
            output=stored_output,
            executed_at=executed_at,
        )

        return SandboxResult(
            exit_code=exit_code,
            output=output,  # return full output to caller
            idempotency_key=idempotency_key,
            manifest_id=manifest_id,
            cached=False,
            executed_at=executed_at,
        )

    async def has_executed(self, idempotency_key: str) -> bool:
        """Return ``True`` if *idempotency_key* is already checkpointed."""
        try:
            async with aiosqlite.connect(str(self._db_path)) as db:
                async with db.execute(
                    "SELECT 1 FROM sandbox_checkpoints WHERE idempotency_key = ?",
                    (idempotency_key,),
                ) as cursor:
                    return (await cursor.fetchone()) is not None
        except Exception as exc:
            logger.error("has_executed lookup failed: %s", exc, exc_info=True)
            return False

    async def get_cached_result(self, idempotency_key: str) -> SandboxResult | None:
        """Look up a specific cached result by key.

        Returns ``None`` when the key has not been checkpointed.
        """
        try:
            async with aiosqlite.connect(str(self._db_path)) as db:
                async with db.execute(
                    "SELECT manifest_id, command, exit_code, output, executed_at "
                    "FROM sandbox_checkpoints WHERE idempotency_key = ?",
                    (idempotency_key,),
                ) as cursor:
                    row = await cursor.fetchone()
                    if row is None:
                        return None
                    return SandboxResult(
                        exit_code=row[2],
                        output=row[3],
                        idempotency_key=idempotency_key,
                        manifest_id=row[0],
                        cached=True,
                        executed_at=row[4],
                    )
        except Exception as exc:
            logger.error("get_cached_result failed: %s", exc, exc_info=True)
            return None

    async def clear_mission_checkpoints(self, manifest_id: str) -> int:
        """Delete all checkpoints belonging to *manifest_id*.

        Intended to be called after a mission completes successfully so
        the checkpoint table does not grow without bound.

        Returns:
            Number of rows deleted.
        """
        try:
            async with aiosqlite.connect(str(self._db_path)) as db:
                cursor = await db.execute(
                    "DELETE FROM sandbox_checkpoints WHERE manifest_id = ?",
                    (manifest_id,),
                )
                await db.commit()
                deleted = cursor.rowcount
                logger.info(
                    "Cleared %d checkpoint(s) for manifest_id=%s",
                    deleted,
                    manifest_id,
                )
                return deleted
        except Exception as exc:
            logger.error("clear_mission_checkpoints failed: %s", exc, exc_info=True)
            return 0

    async def get_mission_stats(self, manifest_id: str) -> dict:
        """Return execution statistics for a mission.

        Returns:
            A dict with keys ``total_executed``, ``total_cached_hits``,
            and ``unique_commands``.  ``total_cached_hits`` is not tracked
            in the DB (it lives in-memory only) so this method reports
            the persisted checkpoint count as ``total_executed`` and
            the distinct command count as ``unique_commands``.
        """
        try:
            async with aiosqlite.connect(str(self._db_path)) as db:
                async with db.execute(
                    "SELECT COUNT(*), COUNT(DISTINCT command) "
                    "FROM sandbox_checkpoints WHERE manifest_id = ?",
                    (manifest_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                    total_executed = row[0] if row else 0
                    unique_commands = row[1] if row else 0

            return {
                "total_executed": total_executed,
                "total_cached_hits": 0,  # in-memory only; not persisted
                "unique_commands": unique_commands,
            }
        except Exception as exc:
            logger.error("get_mission_stats failed: %s", exc, exc_info=True)
            return {
                "total_executed": 0,
                "total_cached_hits": 0,
                "unique_commands": 0,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _store_checkpoint(
        self,
        *,
        idempotency_key: str,
        manifest_id: str,
        command: str,
        exit_code: int,
        output: str,
        executed_at: str,
    ) -> None:
        """Persist a single checkpoint row."""
        try:
            async with aiosqlite.connect(str(self._db_path)) as db:
                await db.execute(
                    "INSERT OR IGNORE INTO sandbox_checkpoints "
                    "(idempotency_key, manifest_id, command, exit_code, output, executed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (idempotency_key, manifest_id, command, exit_code, output, executed_at),
                )
                await db.commit()
            logger.debug("Checkpoint stored: key=%s", idempotency_key)
        except Exception as exc:
            logger.error("_store_checkpoint failed: %s", exc, exc_info=True)

    @staticmethod
    def _truncate_output(output: str) -> str:
        """Truncate *output* to at most ``_OUTPUT_MAX_BYTES`` bytes (UTF-8).

        If truncation occurs, a marker is appended so reviewers know the
        full output was larger than the stored copy.
        """
        encoded = output.encode("utf-8", errors="replace")
        if len(encoded) <= _OUTPUT_MAX_BYTES:
            return output
        truncated = encoded[:_OUTPUT_MAX_BYTES].decode("utf-8", errors="ignore")
        return truncated + "\n\n[OUTPUT TRUNCATED — exceeded 50 KB checkpoint limit]"


# ---------------------------------------------------------------------------
# Factory (ServiceRegistry-backed)
# ---------------------------------------------------------------------------

def get_idempotent_sandbox() -> IdempotentSandbox:
    """Return the singleton :class:`IdempotentSandbox`, registering it in
    the :class:`ServiceRegistry` on first call.

    This is the preferred entry point for other modules.
    """
    # Lazy import to keep the module self-contained and avoid circular deps
    from src.core.di import ServiceRegistry

    existing = ServiceRegistry.get("idempotent_sandbox")
    if existing is not None:
        return existing

    instance = IdempotentSandbox.get_instance()
    ServiceRegistry.register(
        "idempotent_sandbox",
        instance,
        description="Crash-safe sandbox idempotency layer",
    )
    return instance


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "SandboxResult",
    "IdempotentSandbox",
    "get_idempotent_sandbox",
]
