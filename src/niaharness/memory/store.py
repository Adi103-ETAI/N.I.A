"""P1 Memory extensions — file-backed MemoryStore + BuiltinJsonMemoryProvider + drift detection + write-gate + tool injection.

Closes the 8 P1 Memory System gaps from AUDIT.md:

  - :class:`MemoryStore` — file-backed store with §-delimited entries,
    char limits, ``fcntl.flock`` concurrent-write safety.
  - :class:`BuiltinJsonMemoryProvider` — concrete MemoryProvider
    implementation wrapping MemoryStore + the existing in-memory
    ``identity.memory.Memory`` class.
  - Drift detection — ``MemoryStore.detect_drift()`` tracks mtime +
    hash; surfaces external edits so the agent can warn / re-load.
  - Write-gate hooks — ``WriteGate`` runs threat-scan + optional
    approval callback before any write. Blocks prompt-injection
    payloads from landing in MEMORY.md.
  - ``notify_memory_tool_write`` — ``MemoryManager.on_memory_write``
    is wired into ``NiaMemoryTool`` so built-in writes mirror to
    external providers.
  - ``inject_memory_provider_tools`` — auto-register provider tool
    schemas into a ``ToolRegistry``.
  - Memory CLI wizard — ``run_memory_setup_wizard`` for ``nia memory setup``.

Why a separate module?
----------------------
The base ``memory/`` package (manager.py, provider.py, threat_patterns.py)
is already 1243 lines. Adding 800+ more lines of new functionality
there would make it unwieldy. This module is opt-in: callers that
need the file-backed store + drift + write-gate import it explicitly.
Existing callers keep working unchanged.

Usage::

    from niaharness.memory.store import (
        MemoryStore,
        BuiltinJsonMemoryProvider,
        inject_memory_provider_tools,
    )

    store = MemoryStore(path=Path("~/.nia/MEMORY.md").expanduser())
    store.add_entry("Prefers concise replies", category="preference")

    provider = BuiltinJsonMemoryProvider(store=store)
    manager = get_memory_manager()
    manager.add_provider(provider)

    inject_memory_provider_tools(registry, manager)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from niaharness.memory.provider import MemoryProvider
from niaharness.memory.threat_patterns import first_threat_message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# §-delimited entry separator. Each entry is a paragraph bounded by lines
# of § characters. This format is human-readable AND machine-parseable.
ENTRY_SEPARATOR = "\n§\n"

# Maximum total chars in a memory file. Writes that would exceed this
# trigger consolidation (oldest entries pruned first).
DEFAULT_MAX_TOTAL_CHARS = 50_000

# Maximum chars per entry. Longer entries are truncated with a marker.
DEFAULT_MAX_ENTRY_CHARS = 4_000

# Default file lock timeout (seconds). If another process holds the lock
# longer than this, the write fails with a TimeoutError.
DEFAULT_LOCK_TIMEOUT = 5.0

# Categories that the built-in provider recognizes.
VALID_CATEGORIES = frozenset({
    "preference", "fact", "pattern", "conversation", "skill", "note", "other"
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A single entry in the MemoryStore.

    Attributes:
        content: The entry text.
        category: One of VALID_CATEGORIES.
        timestamp: Unix timestamp (creation time).
        source: Who wrote the entry ("agent", "user", "external").
        metadata: Arbitrary key-value pairs.
    """

    content: str
    category: str = "other"
    timestamp: float = field(default_factory=time.time)
    source: str = "agent"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "category": self.category,
            "timestamp": self.timestamp,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            content=data.get("content", ""),
            category=data.get("category", "other"),
            timestamp=float(data.get("timestamp", 0.0)),
            source=data.get("source", "agent"),
            metadata=data.get("metadata", {}) or {},
        )

    def to_markdown(self) -> str:
        """Render as a markdown paragraph with a header line."""
        header = f"<!-- category={self.category} source={self.source} ts={self.timestamp:.0f} -->"
        return f"{header}\n{self.content}"

    @classmethod
    def from_markdown(cls, text: str) -> "MemoryEntry":
        """Parse a markdown paragraph (with optional header comment)."""
        lines = text.strip().splitlines()
        category = "other"
        source = "agent"
        timestamp = time.time()
        metadata: Dict[str, Any] = {}

        # Check for header comment.
        if lines and lines[0].strip().startswith("<!--"):
            header_line = lines[0].strip()
            cat_match = re.search(r"category=(\S+)", header_line)
            if cat_match:
                category = cat_match.group(1)
            src_match = re.search(r"source=(\S+)", header_line)
            if src_match:
                source = src_match.group(1)
            ts_match = re.search(r"ts=(\S+)", header_line)
            if ts_match:
                try:
                    timestamp = float(ts_match.group(1))
                except ValueError:
                    pass
            lines = lines[1:]

        content = "\n".join(lines).strip()
        return cls(
            content=content,
            category=category,
            timestamp=timestamp,
            source=source,
            metadata=metadata,
        )


@dataclass
class DriftReport:
    """Result of a drift detection check.

    Attributes:
        changed: True if the file changed externally since the last check.
        old_hash: The previous content hash (or None if no prior state).
        new_hash: The current content hash.
        old_mtime: The previous mtime (or None).
        new_mtime: The current mtime.
        old_size: The previous file size in bytes (or None).
        new_size: The current file size in bytes.
    """

    changed: bool
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    old_mtime: Optional[float] = None
    new_mtime: Optional[float] = None
    old_size: Optional[int] = None
    new_size: Optional[int] = None


# ---------------------------------------------------------------------------
# Write-gate
# ---------------------------------------------------------------------------


# Type for an approval callback. Returns True to allow the write, False to block.
ApprovalCallback = Callable[[str, str, str], bool]
# Args: (action, target, content). action is "add"|"replace"|"remove"|"clear".
# target is the entry key/category. content is the proposed content.


class WriteGate:
    """Threat-scan + approval gate for memory writes.

    Every write to the MemoryStore passes through this gate. The gate:
      1. Runs threat-pattern scan (prompt-injection / exfil / C2).
      2. If threats found, blocks the write and logs.
      3. If an approval callback is set, calls it. The callback can
         veto the write (e.g. require user confirmation for "skill"
         category writes).
      4. Otherwise, allows the write.
    """

    def __init__(
        self,
        *,
        approval_callback: Optional[ApprovalCallback] = None,
        threat_scope: str = "strict",
    ) -> None:
        self._approval_callback = approval_callback
        self._threat_scope = threat_scope
        self._blocked_count = 0
        self._approved_count = 0
        self._lock = threading.Lock()

    def check(
        self,
        action: str,
        target: str,
        content: str,
    ) -> tuple[bool, str]:
        """Return (allowed, reason).

        If not allowed, reason is a human-readable explanation.
        If allowed, reason is "".
        """
        # 1. Threat scan.
        threat = first_threat_message(content, scope=self._threat_scope)
        if threat is not None:
            with self._lock:
                self._blocked_count += 1
            logger.warning(
                "Memory write-gate blocked %s on %s: %s",
                action, target, threat,
            )
            return False, threat

        # 2. Approval callback.
        if self._approval_callback is not None:
            try:
                approved = self._approval_callback(action, target, content)
            except Exception as exc:
                logger.warning("Memory write-gate approval callback failed: %s", exc)
                return False, f"approval callback error: {exc}"
            if not approved:
                with self._lock:
                    self._blocked_count += 1
                return False, "write rejected by approval callback"
            with self._lock:
                self._approved_count += 1

        return True, ""

    @property
    def blocked_count(self) -> int:
        with self._lock:
            return self._blocked_count

    @property
    def approved_count(self) -> int:
        with self._lock:
            return self._approved_count

    def reset_stats(self) -> None:
        with self._lock:
            self._blocked_count = 0
            self._approved_count = 0


# ---------------------------------------------------------------------------
# MemoryStore — file-backed, §-delimited, fcntl.flock-protected
# ---------------------------------------------------------------------------


class MemoryStore:
    """File-backed memory store with §-delimited entries.

    Each entry is a markdown paragraph separated by a line containing
    only ``§``. The store supports:

      - ``add_entry`` / ``replace_entry`` / ``remove_entry`` / ``clear``
      - ``get_entries`` (filter by category, source, or full-text search)
      - ``apply_batch`` (atomic multi-op)
      - ``detect_drift`` (external change detection)
      - Concurrent-write safety via ``fcntl.flock`` (POSIX) or a
        thread lock fallback on non-POSIX systems.
      - Char limits with automatic consolidation (oldest entries pruned).
      - Write-gate hook (threat scan + approval).

    The store is NOT async — file I/O is fast enough that blocking is
    fine, and the MemoryManager already runs sync_turn on a background
    thread.
    """

    def __init__(
        self,
        *,
        path: Path,
        max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
        max_entry_chars: int = DEFAULT_MAX_ENTRY_CHARS,
        lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
        write_gate: Optional[WriteGate] = None,
    ) -> None:
        self._path = Path(path)
        self._max_total_chars = max_total_chars
        self._max_entry_chars = max_entry_chars
        self._lock_timeout = lock_timeout
        self._write_gate = write_gate or WriteGate()
        self._thread_lock = threading.Lock()
        # Drift tracking state.
        self._last_known_hash: Optional[str] = None
        self._last_known_mtime: Optional[float] = None
        self._last_known_size: Optional[int] = None
        # Cache of parsed entries (invalidated on file change).
        self._entries_cache: Optional[List[MemoryEntry]] = None
        # Initialize the file + drift state.
        self._ensure_file()
        self._refresh_drift_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def write_gate(self) -> WriteGate:
        return self._write_gate

    def add_entry(self, entry: MemoryEntry) -> bool:
        """Add an entry. Returns True if added, False if blocked by the gate."""
        # Truncate content if needed.
        if len(entry.content) > self._max_entry_chars:
            entry.content = (
                entry.content[: self._max_entry_chars - 30]
                + "\n...[truncated]..."
            )

        allowed, reason = self._write_gate.check(
            "add", entry.category, entry.content,
        )
        if not allowed:
            return False

        with self._thread_lock, self._file_lock():
            entries = self._read_entries_unlocked()
            entries.append(entry)
            entries = self._consolidate(entries)
            self._write_entries_unlocked(entries)
        return True

    def replace_entry(
        self,
        index: int,
        new_entry: MemoryEntry,
    ) -> bool:
        """Replace the entry at `index`. Returns True if replaced, False if blocked."""
        allowed, reason = self._write_gate.check(
            "replace", str(index), new_entry.content,
        )
        if not allowed:
            return False

        with self._thread_lock, self._file_lock():
            entries = self._read_entries_unlocked()
            if index < 0 or index >= len(entries):
                return False
            entries[index] = new_entry
            entries = self._consolidate(entries)
            self._write_entries_unlocked(entries)
        return True

    def remove_entry(self, index: int) -> bool:
        """Remove the entry at `index`. Returns True if removed."""
        allowed, _ = self._write_gate.check("remove", str(index), "")
        if not allowed:
            return False

        with self._thread_lock, self._file_lock():
            entries = self._read_entries_unlocked()
            if index < 0 or index >= len(entries):
                return False
            entries.pop(index)
            self._write_entries_unlocked(entries)
        return True

    def clear(self) -> bool:
        """Remove all entries. Returns True if cleared."""
        allowed, _ = self._write_gate.check("clear", "all", "")
        if not allowed:
            return False

        with self._thread_lock, self._file_lock():
            self._write_entries_unlocked([])
        return True

    def get_entries(
        self,
        *,
        category: Optional[str] = None,
        source: Optional[str] = None,
        query: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """Return entries, optionally filtered by category/source/query."""
        with self._thread_lock:
            entries = self._read_entries_unlocked()

        if category:
            entries = [e for e in entries if e.category == category]
        if source:
            entries = [e for e in entries if e.source == source]
        if query:
            query_lower = query.lower()
            entries = [
                e for e in entries
                if query_lower in e.content.lower()
            ]
        if limit is not None:
            entries = entries[:limit]
        return entries

    def apply_batch(
        self,
        operations: List[Dict[str, Any]],
    ) -> List[bool]:
        """Apply a batch of operations atomically.

        Each operation is a dict with "action" (add|replace|remove|clear)
        and action-specific keys. Returns a list of per-op success flags.
        If any op fails (blocked by gate or invalid index), the file is
        still written with the successful ops' effects.

        Example::

            store.apply_batch([
                {"action": "add", "entry": MemoryEntry(content="...")},
                {"action": "remove", "index": 0},
                {"action": "add", "entry": MemoryEntry(content="...")},
            ])
        """
        results: List[bool] = []
        with self._thread_lock, self._file_lock():
            entries = self._read_entries_unlocked()
            for op in operations:
                action = op.get("action", "")
                if action == "add":
                    entry = op.get("entry")
                    if not isinstance(entry, MemoryEntry):
                        results.append(False)
                        continue
                    allowed, _ = self._write_gate.check(
                        "add", entry.category, entry.content,
                    )
                    if not allowed:
                        results.append(False)
                        continue
                    entries.append(entry)
                    results.append(True)
                elif action == "replace":
                    index = op.get("index", -1)
                    entry = op.get("entry")
                    if not isinstance(entry, MemoryEntry):
                        results.append(False)
                        continue
                    allowed, _ = self._write_gate.check(
                        "replace", str(index), entry.content,
                    )
                    if not allowed:
                        results.append(False)
                        continue
                    if index < 0 or index >= len(entries):
                        results.append(False)
                        continue
                    entries[index] = entry
                    results.append(True)
                elif action == "remove":
                    index = op.get("index", -1)
                    if index < 0 or index >= len(entries):
                        results.append(False)
                        continue
                    entries.pop(index)
                    results.append(True)
                elif action == "clear":
                    entries = []
                    results.append(True)
                else:
                    results.append(False)
            entries = self._consolidate(entries)
            self._write_entries_unlocked(entries)
        return results

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(self) -> DriftReport:
        """Check if the file was modified externally since the last read/write.

        Compares mtime + content hash. If the file changed outside this
        MemoryStore instance, returns a DriftReport with ``changed=True``.
        The agent can use this to warn the user or reload the cache.
        """
        if not self._path.exists():
            return DriftReport(changed=True)

        try:
            stat = self._path.stat()
            new_mtime = stat.st_mtime
            new_size = stat.st_size
        except OSError:
            return DriftReport(changed=True)

        # Cheap check: if mtime + size match, no drift.
        if (
            self._last_known_mtime == new_mtime
            and self._last_known_size == new_size
            and self._last_known_hash is not None
        ):
            return DriftReport(
                changed=False,
                old_hash=self._last_known_hash,
                new_hash=self._last_known_hash,
                old_mtime=self._last_known_mtime,
                new_mtime=new_mtime,
                old_size=self._last_known_size,
                new_size=new_size,
            )

        # mtime/size differ — compute hash to confirm.
        content = self._path.read_text(encoding="utf-8", errors="replace")
        new_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        changed = new_hash != self._last_known_hash

        report = DriftReport(
            changed=changed,
            old_hash=self._last_known_hash,
            new_hash=new_hash,
            old_mtime=self._last_known_mtime,
            new_mtime=new_mtime,
            old_size=self._last_known_size,
            new_size=new_size,
        )

        # Update the cached state so subsequent checks are cheap.
        self._last_known_hash = new_hash
        self._last_known_mtime = new_mtime
        self._last_known_size = new_size
        # Invalidate the entries cache.
        self._entries_cache = None

        if changed:
            logger.info(
                "Memory drift detected in %s (old_size=%s new_size=%d)",
                self._path,
                self._last_known_size,
                new_size,
            )

        return report

    def reload(self) -> None:
        """Force a re-read of the file + reset drift state."""
        with self._thread_lock:
            self._entries_cache = None
            self._refresh_drift_state()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return stats about the store."""
        entries = self.get_entries()
        return {
            "path": str(self._path),
            "entry_count": len(entries),
            "total_chars": sum(len(e.content) for e in entries),
            "max_total_chars": self._max_total_chars,
            "max_entry_chars": self._max_entry_chars,
            "categories": {
                cat: sum(1 for e in entries if e.category == cat)
                for cat in {e.category for e in entries}
            },
            "write_gate_blocked": self._write_gate.blocked_count,
            "write_gate_approved": self._write_gate.approved_count,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_file(self) -> None:
        """Create the file + parent dir if they don't exist."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("", encoding="utf-8")

    def _refresh_drift_state(self) -> None:
        """Read the file and update the drift-tracking state."""
        if not self._path.exists():
            self._last_known_hash = None
            self._last_known_mtime = None
            self._last_known_size = None
            return
        try:
            stat = self._path.stat()
            self._last_known_mtime = stat.st_mtime
            self._last_known_size = stat.st_size
            content = self._path.read_text(encoding="utf-8", errors="replace")
            self._last_known_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        except OSError:
            self._last_known_hash = None
            self._last_known_mtime = None
            self._last_known_size = None

    def _read_entries_unlocked(self) -> List[MemoryEntry]:
        """Read entries from disk (caller holds the lock)."""
        if not self._path.exists():
            return []
        content = self._path.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return []
        # Split on the § separator. Each chunk is one entry's markdown.
        chunks = content.split(ENTRY_SEPARATOR)
        entries: List[MemoryEntry] = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                entries.append(MemoryEntry.from_markdown(chunk))
            except Exception as exc:
                logger.debug("Skipping unparseable memory entry: %s", exc)
        return entries

    def _write_entries_unlocked(self, entries: List[MemoryEntry]) -> None:
        """Write entries to disk (caller holds the lock)."""
        chunks = [e.to_markdown() for e in entries if e.content]
        content = ENTRY_SEPARATOR.join(chunks)
        if content and not content.endswith("\n"):
            content += "\n"
        self._path.write_text(content, encoding="utf-8")
        # Update drift state.
        self._refresh_drift_state()
        self._entries_cache = None

    def _consolidate(self, entries: List[MemoryEntry]) -> List[MemoryEntry]:
        """Prune oldest entries if total chars exceed the limit.

        Keeps preferences + facts (durable) over patterns + notes (ephemeral).
        """
        total = sum(len(e.content) for e in entries)
        if total <= self._max_total_chars:
            return entries
        # Sort by priority: preferences/facts first (keep), then by timestamp
        # (newest first within the same priority tier).
        durable_categories = {"preference", "fact"}
        durable = [e for e in entries if e.category in durable_categories]
        ephemeral = [e for e in entries if e.category not in durable_categories]
        # Sort ephemeral by timestamp (newest first).
        ephemeral.sort(key=lambda e: e.timestamp, reverse=True)

        # Trim ephemeral until we're under budget.
        while ephemeral and total > self._max_total_chars:
            removed = ephemeral.pop()
            total -= len(removed.content)

        # If still over, trim durable (oldest first).
        durable.sort(key=lambda e: e.timestamp, reverse=True)
        while durable and total > self._max_total_chars:
            removed = durable.pop()
            total -= len(removed.content)

        return durable + ephemeral

    def _file_lock(self):
        """Context manager that acquires fcntl.flock on the memory file.

        Falls back to a no-op on systems without fcntl (Windows).
        """
        return _FcntlLock(self._path, timeout=self._lock_timeout)


class _FcntlLock:
    """Context manager wrapping fcntl.flock with a timeout.

    Falls back to a no-op on systems without fcntl (e.g. Windows).
    On timeout, raises TimeoutError.
    """

    def __init__(self, path: Path, timeout: float = 5.0) -> None:
        self._path = path
        self._timeout = timeout
        self._fd: Optional[int] = None

    def __enter__(self) -> "_FcntlLock":
        try:
            import fcntl  # type: ignore[import-not-found]
        except ImportError:
            # Windows — no fcntl. Use the thread lock only.
            return self

        # Open a separate fd for locking (doesn't interfere with reads/writes).
        lock_path = self._path.parent / f".{self._path.name}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)

        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(self._fd)
                    self._fd = None
                    raise TimeoutError(
                        f"Could not acquire lock on {self._path} within {self._timeout}s"
                    )
                time.sleep(0.05)

    def __exit__(self, *args: Any) -> None:
        if self._fd is None:
            return
        try:
            import fcntl  # type: ignore[import-not-found]
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None


# ---------------------------------------------------------------------------
# BuiltinJsonMemoryProvider — concrete MemoryProvider implementation
# ---------------------------------------------------------------------------


class BuiltinJsonMemoryProvider(MemoryProvider):
    """Built-in memory provider wrapping MemoryStore + identity.memory.Memory.

    This is the always-on provider that backs the ``nia_memory`` tool.
    It exposes:
      - A tool schema for the ``memory_search`` / ``memory_add`` tools.
      - Prefetch via the in-memory Memory class's relevance search.
      - Sync via the in-memory Memory's ``add_conversation``.
      - Lifecycle hooks that snapshot MemoryStore drift.

    The provider is registered first by the MemoryManager. At most one
    external provider may be registered alongside it.
    """

    def __init__(
        self,
        *,
        store: Optional[MemoryStore] = None,
        memory: Optional[Any] = None,
        store_path: Optional[Path] = None,
    ) -> None:
        # If no store was provided, create one at the default location.
        if store is None:
            from niaharness.memory.paths import get_project_memory_dir
            import os
            cwd = os.getcwd()
            store_path = store_path or (
                get_project_memory_dir(cwd) / "STORE.md"
            )
            store = MemoryStore(path=store_path)
        self._store = store
        self._memory = memory  # identity.memory.Memory instance (optional)
        self._session_id: str = ""

    @property
    def name(self) -> str:
        return "builtin"

    @property
    def store(self) -> MemoryStore:
        return self._store

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id
        # Check for drift on init.
        drift = self._store.detect_drift()
        if drift.changed and drift.old_hash is not None:
            logger.info(
                "Memory store changed externally since last session "
                "(old_size=%s new_size=%d)",
                drift.old_size, drift.new_size,
            )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas for the built-in memory tools."""
        return [
            {
                "name": "memory_search",
                "description": (
                    "Search the agent's persistent memory (preferences, "
                    "facts, patterns, notes). Returns matching entries."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category filter "
                            "(preference|fact|pattern|note|other).",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default 5).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_add",
                "description": (
                    "Add an entry to persistent memory. Use for "
                    "preferences, facts, patterns, and notes that should "
                    "survive across sessions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The memory entry content.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Entry category "
                            "(preference|fact|pattern|note|other).",
                            "default": "other",
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "memory_list",
                "description": (
                    "List memory entries, optionally filtered by category."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Optional category filter.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default 20).",
                            "default": 20,
                        },
                    },
                },
            },
        ]

    def system_prompt_block(self) -> str:
        """Return a system-prompt section describing the memory store."""
        stats = self._store.stats()
        return (
            "## Persistent Memory\n"
            f"- Location: {stats['path']}\n"
            f"- Entries: {stats['entry_count']}\n"
            f"- Categories: {stats['categories']}\n"
            "Use memory_search / memory_add / memory_list tools to "
            "access persistent memory."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return relevant memory entries for the query."""
        entries = self._store.get_entries(query=query, limit=5)
        if not entries:
            return ""
        lines = ["# Relevant Memories (prefetched)"]
        for entry in entries:
            lines.append(f"- [{entry.category}] {entry.content[:200]}")
        return "\n".join(lines)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # No-op — prefetch is already synchronous + fast.
        pass

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist a completed turn to the in-memory Memory (if available)."""
        if self._memory is not None:
            try:
                self._memory.add_conversation("user", user_content)
                self._memory.add_conversation("assistant", assistant_content)
            except Exception as exc:
                logger.debug("Builtin provider sync_turn failed: %s", exc)

        # Also check for drift — if the user edited MEMORY.md externally,
        # we want to know.
        try:
            self._store.detect_drift()
        except Exception:
            pass

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        """Dispatch a memory tool call."""
        if tool_name == "memory_search":
            query = args.get("query", "")
            category = args.get("category")
            limit = args.get("limit", 5)
            entries = self._store.get_entries(
                query=query, category=category, limit=limit,
            )
            if not entries:
                return json.dumps({"success": True, "results": []})
            return json.dumps({
                "success": True,
                "results": [
                    {
                        "content": e.content,
                        "category": e.category,
                        "timestamp": e.timestamp,
                        "source": e.source,
                    }
                    for e in entries
                ],
            })
        elif tool_name == "memory_add":
            content = args.get("content", "")
            category = args.get("category", "other")
            if not content:
                return json.dumps({"success": False, "error": "content is required"})
            entry = MemoryEntry(content=content, category=category, source="agent")
            added = self._store.add_entry(entry)
            if not added:
                return json.dumps({
                    "success": False,
                    "error": "write blocked by gate (threat scan or approval)",
                })
            # Mirror to external providers via the manager.
            self._notify_external_providers("add", category, content)
            return json.dumps({"success": True, "entry": entry.to_dict()})
        elif tool_name == "memory_list":
            category = args.get("category")
            limit = args.get("limit", 20)
            entries = self._store.get_entries(category=category, limit=limit)
            return json.dumps({
                "success": True,
                "results": [e.to_dict() for e in entries],
            })
        else:
            raise NotImplementedError(f"Builtin provider does not handle tool '{tool_name}'")

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes to the store.

        Called by the MemoryManager when the nia_memory tool writes —
        we add the entry to the store so it persists to disk.
        """
        if action in {"add", "replace"} and content:
            entry = MemoryEntry(
                content=content,
                category=target if target in VALID_CATEGORIES else "other",
                source="agent",
                metadata=metadata or {},
            )
            self._store.add_entry(entry)

    def shutdown(self) -> None:
        # Nothing to clean up — the store is file-backed.
        pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _notify_external_providers(
        self,
        action: str,
        target: str,
        content: str,
    ) -> None:
        """Best-effort: notify the MemoryManager so external providers can mirror."""
        try:
            from niaharness.memory.manager import get_memory_manager
            manager = get_memory_manager()
            manager.on_memory_write(action, target, content)
        except Exception as exc:
            logger.debug("notify_external_providers failed: %s", exc)


# ---------------------------------------------------------------------------
# Tool injection
# ---------------------------------------------------------------------------


class MemoryProviderToolAdapter:
    """Adapter that wraps a memory tool call as a BaseTool.

    The MemoryManager routes tool calls to providers via
    ``handle_tool_call``. This adapter wraps a (provider, tool_name)
    pair as a ``BaseTool`` so the ToolRegistry can register it
    alongside the built-in tools.
    """

    def __init__(
        self,
        manager: Any,
        tool_name: str,
        description: str,
        parameters: Dict[str, Any],
    ) -> None:
        self._manager = manager
        self.name = tool_name
        self.description = description
        self.input_model = _build_pydantic_model(tool_name, parameters)

    async def execute(self, arguments: Any, context: Any) -> Any:
        from niaharness.tools.base import ToolResult
        args_dict = arguments.model_dump() if hasattr(arguments, "model_dump") else dict(arguments)
        try:
            output = self._manager.handle_tool_call(self.name, args_dict)
            return ToolResult(output=output)
        except Exception as exc:
            return ToolResult(output=str(exc), is_error=True)

    def is_read_only(self, arguments: Any) -> bool:
        # memory_search / memory_list are read-only; memory_add is not.
        return self.name in {"memory_search", "memory_list"}

    def to_api_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }


def _build_pydantic_model(model_name: str, parameters: Dict[str, Any]) -> Any:
    """Build a Pydantic BaseModel from a JSON-schema parameters dict."""
    from pydantic import create_model

    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))

    fields: Dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "string")
        if prop_type == "integer":
            py_type = int
        elif prop_type == "number":
            py_type = float
        elif prop_type == "boolean":
            py_type = bool
        else:
            py_type = str

        default = ... if prop_name in required else prop_schema.get("default")
        if default is ... and prop_name not in required:
            default = None
        fields[prop_name] = (py_type, default)

    return create_model(model_name, **fields)


def inject_memory_provider_tools(registry: Any, manager: Any) -> List[str]:
    """Auto-register memory provider tools into a ToolRegistry.

    Iterates the manager's providers, collects their tool schemas,
    and registers a MemoryProviderToolAdapter for each. Returns the
    list of registered tool names.

    Call this after ``create_default_tool_registry`` + ``register_nia_tools``.
    """
    registered: List[str] = []
    for schema in manager.get_all_tool_schemas():
        tool_name = schema.get("name", "")
        if not tool_name:
            continue
        if registry.get(tool_name) is not None:
            logger.debug("Skipping memory tool %s — already registered", tool_name)
            continue
        adapter = MemoryProviderToolAdapter(
            manager=manager,
            tool_name=tool_name,
            description=schema.get("description", ""),
            parameters=schema.get("parameters", {"type": "object", "properties": {}}),
        )
        registry.register(adapter)
        registered.append(tool_name)
        logger.info("Injected memory provider tool: %s", tool_name)
    return registered


# ---------------------------------------------------------------------------
# Memory CLI setup wizard
# ---------------------------------------------------------------------------


def run_memory_setup_wizard(
    *,
    cwd: Optional[str] = None,
    interactive: bool = True,
) -> Dict[str, Any]:
    """Run the `nia memory setup` wizard.

    Creates the memory directory, an empty MEMORY.md, and optionally
    seeds it with a few starter entries (preferences, facts). Returns
    a dict with the setup result.

    Args:
        cwd: The project directory. Defaults to os.getcwd().
        interactive: If True, prompt the user for starter entries.
            If False, create an empty store non-interactively.
    """
    from niaharness.memory.paths import get_project_memory_dir, get_memory_entrypoint

    cwd = cwd or os.getcwd()
    memory_dir = get_project_memory_dir(cwd)
    entrypoint = get_memory_entrypoint(cwd)
    store_path = memory_dir / "STORE.md"

    result: Dict[str, Any] = {
        "memory_dir": str(memory_dir),
        "entrypoint": str(entrypoint),
        "store_path": str(store_path),
        "created": [],
        "seeded_entries": 0,
    }

    # Create the memory directory (already done by get_project_memory_dir,
    # but be explicit).
    memory_dir.mkdir(parents=True, exist_ok=True)
    if not entrypoint.exists():
        entrypoint.write_text("# Memory Index\n\n", encoding="utf-8")
        result["created"].append("MEMORY.md")

    # Create the store.
    store = MemoryStore(path=store_path)
    result["created"].append("STORE.md")

    # Seed starter entries.
    starter_entries: List[MemoryEntry] = []
    if interactive:
        print("NIA Memory Setup")
        print("=" * 40)
        print(f"Memory directory: {memory_dir}")
        print()
        print("Let's add a few starter entries. Press Enter to skip any.")
        print()

        pref = input("A user preference (e.g. 'prefers concise replies'): ").strip()
        if pref:
            starter_entries.append(MemoryEntry(
                content=pref, category="preference", source="user",
            ))

        fact = input("An important fact (e.g. 'project uses Python 3.12'): ").strip()
        if fact:
            starter_entries.append(MemoryEntry(
                content=fact, category="fact", source="user",
            ))

        note = input("A general note: ").strip()
        if note:
            starter_entries.append(MemoryEntry(
                content=note, category="note", source="user",
            ))
    else:
        # Non-interactive: seed a single placeholder entry.
        starter_entries.append(MemoryEntry(
            content="Memory store initialized.",
            category="note",
            source="setup_wizard",
        ))

    for entry in starter_entries:
        if store.add_entry(entry):
            result["seeded_entries"] += 1

    # Add index entries to MEMORY.md.
    if starter_entries:
        existing = entrypoint.read_text(encoding="utf-8") if entrypoint.exists() else ""
        new_lines = []
        for entry in starter_entries:
            title = entry.content[:50].replace("\n", " ")
            new_lines.append(f"- [{entry.category}] {title}")
        entrypoint.write_text(
            existing.rstrip() + "\n" + "\n".join(new_lines) + "\n",
            encoding="utf-8",
        )

    result["stats"] = store.stats()
    return result


# ---------------------------------------------------------------------------
# Convenience: initialize the default MemoryManager + BuiltinJsonMemoryProvider
# ---------------------------------------------------------------------------


def initialize_default_memory_manager(
    *,
    cwd: Optional[str] = None,
    memory: Optional[Any] = None,
) -> Any:
    """Initialize the process-wide MemoryManager with the builtin provider.

    Call this once at startup (e.g. from the CLI's main entry point).
    Returns the manager.

    Args:
        cwd: The project directory. Defaults to os.getcwd().
        memory: An optional identity.memory.Memory instance for
            in-memory conversation tracking.
    """
    from niaharness.memory.manager import get_memory_manager

    cwd = cwd or os.getcwd()
    manager = get_memory_manager()

    # Don't double-register.
    if manager.get_provider("builtin") is not None:
        return manager

    provider = BuiltinJsonMemoryProvider(memory=memory)
    manager.add_provider(provider)
    manager.initialize_all(session_id="", nia_home=cwd)
    return manager


__all__ = [
    "BuiltinJsonMemoryProvider",
    "DriftReport",
    "ENTRY_SEPARATOR",
    "MemoryEntry",
    "MemoryProviderToolAdapter",
    "MemoryStore",
    "WriteGate",
    "DEFAULT_LOCK_TIMEOUT",
    "DEFAULT_MAX_ENTRY_CHARS",
    "DEFAULT_MAX_TOTAL_CHARS",
    "VALID_CATEGORIES",
    "inject_memory_provider_tools",
    "initialize_default_memory_manager",
    "run_memory_setup_wizard",
]
