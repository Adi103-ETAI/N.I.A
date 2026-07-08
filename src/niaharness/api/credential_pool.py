"""NIA Credential Pool — provider failover with exhaustion tracking.

Adapted from the reference project's agent/credential_pool.py (2,384 lines),
streamlined for NIA's architecture while preserving the critical safety
guarantees:

  - **Multi-credential pools** per provider (env, manual, OAuth device-code)
  - **Exhaustion tracking** with TTL-based cooldowns (429, 401, 402)
  - **Terminal failure detection** (token_invalidated, token_revoked,
    invalid_grant, etc.) → STATUS_DEAD, never re-enters rotation
  - **Four rotation strategies**: fill_first (default), round_robin,
    least_used, random
  - **Lease accounting** for concurrent requests per credential
  - **Atomic write-through** to ``~/.nia/credentials/<provider>.json``
  - **Mark-and-rotate** on failure: exhaust current → select next available
  - **API-key hint matching**: when a specific key fails, exhaust *that*
    entry (not the next one), then rotate
  - **Retry-After parsing** from error responses (honor upstream cooldown)

Why this matters
----------------
Without a credential pool, a single 401/429 from the only configured
provider terminates the turn. With a pool, NIA can:

  1. Rotate to the next available credential on transient failure (429)
  2. Honor Retry-After headers instead of fixed backoff
  3. Mark permanently-revoked credentials as DEAD (skip forever)
  4. Recover automatically when another process refreshes tokens on disk
  5. Distribute load across multiple API keys (rate-limit pooling)

Persistence model
-----------------
Each provider has its own pool file at ``~/.nia/credentials/<provider>.json``:

    {
      "version": 1,
      "provider": "anthropic",
      "entries": [
        {
          "id": "a1b2c3",
          "label": "work-account",
          "auth_type": "api_key",
          "priority": 0,
          "source": "manual",
          "access_token": "sk-ant-...",
          "last_status": "ok",
          "last_status_at": null,
          "last_error_code": null,
          "last_error_reason": null,
          "last_error_message": null,
          "last_error_reset_at": null,
          "request_count": 0
        }
      ],
      "updated_at": "2026-07-08T10:30:00Z"
    }

The file is written atomically (tempfile + os.replace) and read fresh on
each ``load_pool()`` call so multiple NIA processes see each other's
updates (write-through to global state, mirroring the reference design).
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status and type constants (adapted from reference)
# ---------------------------------------------------------------------------

STATUS_OK = "ok"
STATUS_EXHAUSTED = "exhausted"
# Terminal failure — the credential will never recover on its own. Used for
# upstream-permanent OAuth states like ``token_invalidated`` / ``token_revoked``
# where retrying after a TTL cooldown is guaranteed to fail. DEAD entries are
# excluded from rotation unconditionally and only clear when an explicit
# write-side sync (e.g. a fresh device-code login) rewrites the tokens.
STATUS_DEAD = "dead"

# OAuth error reasons that indicate the credential is permanently invalid
# server-side and cannot be recovered by retry/refresh. Sourced from
# OpenAI Codex Responses API, Anthropic, xAI, and Google OAuth spec.
_TERMINAL_AUTH_REASONS = frozenset({
    "token_invalidated",    # OpenAI Codex: "Your authentication token has been invalidated."
    "token_revoked",        # OAuth 2.0 RFC 7009: token explicitly revoked
    "invalid_token",        # RFC 6750: bearer token is malformed/expired/revoked
    "invalid_grant",        # RFC 6749: refresh_token rejected during refresh
    "unauthorized_client",  # RFC 6749: client no longer authorized
    "refresh_token_reused", # Single-use refresh token consumed by another process
})

# How long a DEAD manual credential is preserved before being pruned.
# Manual entries are independent credentials with no singleton to re-seed
# from, so pruning them after a quiet window cleans up dead state without
# losing recoverability — the user always has the option to re-add via
# ``niaharness auth add``.
DEAD_MANUAL_PRUNE_TTL_SECONDS = 24 * 60 * 60  # 24 hours

# Exhaustion TTLs (seconds) by error code. 429s honor Retry-After when
# present (capped at 1 hour); other errors use these fixed TTLs.
_EXHAUSTED_TTLS: Dict[Optional[int], int] = {
    429: 5 * 60,           # 5 minutes (or Retry-After if longer)
    401: 60 * 60,          # 1 hour (likely needs re-auth, but may be transient)
    402: 60 * 60,          # 1 hour (billing — give time for payment to process)
    403: 30 * 60,          # 30 minutes (often permission/quota, sometimes permanent)
    500: 60,               # 1 minute (server error, retry quickly)
    502: 60,
    503: 60,
    504: 60,
}
DEFAULT_EXHAUSTED_TTL = 5 * 60  # 5 minutes for unknown codes

DEFAULT_MAX_CONCURRENT_PER_CREDENTIAL = 1  # soft cap on in-flight requests per key

AUTH_TYPE_OAUTH = "oauth"
AUTH_TYPE_API_KEY = "api_key"

SOURCE_MANUAL = "manual"
SOURCE_ENV = "env"
SOURCE_DEVICE_CODE = "device_code"

STRATEGY_FILL_FIRST = "fill_first"
STRATEGY_ROUND_ROBIN = "round_robin"
STRATEGY_RANDOM = "random"
STRATEGY_LEAST_USED = "least_used"
SUPPORTED_POOL_STRATEGIES = frozenset({
    STRATEGY_FILL_FIRST,
    STRATEGY_ROUND_ROBIN,
    STRATEGY_RANDOM,
    STRATEGY_LEAST_USED,
})


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _credentials_dir() -> Path:
    """Return ``~/.nia/credentials/`` (created if missing)."""
    from niaharness.prompts.soul import get_nia_home

    d = get_nia_home() / "credentials"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pool_file(provider: str) -> Path:
    """Return the pool file path for a provider."""
    # Sanitize provider name for filesystem safety.
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in provider)
    return _credentials_dir() / f"{safe}.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via tempfile + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Error-context normalization (adapted from reference)
# ---------------------------------------------------------------------------


def _extract_retry_delay_seconds(message: str) -> Optional[float]:
    """Parse a Retry-After value from an error message.

    Handles both ``Retry-After: 30`` (seconds) and HTTP-date forms.
    Returns the delay in seconds, or None if not parseable.
    """
    if not message:
        return None
    import re

    # Look for "retry after N seconds" or "retry-after: N" patterns.
    m = re.search(r"retry[-_]after[:\s]+(\d+)", message, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Try HTTP-date form.
    m = re.search(r"retry[-_]after[:\s]+([a-z]{3},\s*\d{1,2}\s+\w+\s+\d{4})", message, re.IGNORECASE)
    if m:
        try:
            from email.utils import parsedate_to_datetime

            dt = parsedate_to_datetime(m.group(1))
            if dt is not None:
                delay = dt.timestamp() - time.time()
                if delay > 0:
                    return delay
        except (ValueError, TypeError):
            pass
    return None


def _normalize_error_context(error_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize an error context dict, extracting reason/message/reset_at."""
    if not isinstance(error_context, dict):
        return {}
    result: Dict[str, Any] = {}
    # Reason: look in common keys.
    for key in ("reason", "error_reason", "code", "error_code", "type", "error_type"):
        value = error_context.get(key)
        if isinstance(value, str) and value.strip():
            result["reason"] = value.strip()
            break
    # Message: look in common keys.
    for key in ("message", "error_message", "detail", "error"):
        value = error_context.get(key)
        if isinstance(value, str) and value.strip():
            result["message"] = value.strip()
            break
    # Reset-at: Retry-After or explicit reset timestamp.
    for key in ("reset_at", "retry_after", "retry-after"):
        value = error_context.get(key)
        if value is not None:
            try:
                if isinstance(value, (int, float)):
                    result["reset_at"] = float(value)
                elif isinstance(value, str):
                    delay = _extract_retry_delay_seconds(value)
                    if delay is not None:
                        result["reset_at"] = time.time() + delay
            except (ValueError, TypeError):
                pass
            break
    # If no explicit reset_at but the message has a retry hint, parse it.
    if "reset_at" not in result and "message" in result:
        delay = _extract_retry_delay_seconds(result["message"])
        if delay is not None:
            result["reset_at"] = time.time() + delay
    return result


def _exhausted_ttl(error_code: Optional[int]) -> int:
    """Return the exhaustion TTL (seconds) for an error code."""
    return _EXHAUSTED_TTLS.get(error_code, DEFAULT_EXHAUSTED_TTL)


def _exhausted_until(entry: "PooledCredential") -> Optional[float]:
    """Return the epoch time when an exhausted entry becomes available again."""
    if entry.last_status != STATUS_EXHAUSTED:
        return None
    # Prefer explicit reset_at (from Retry-After header).
    if entry.last_error_reset_at is not None:
        return entry.last_error_reset_at
    # Fall back to TTL from error code.
    if entry.last_status_at is not None:
        return entry.last_status_at + _exhausted_ttl(entry.last_error_code)
    return None


# ---------------------------------------------------------------------------
# PooledCredential dataclass
# ---------------------------------------------------------------------------


@dataclass
class PooledCredential:
    """A single credential in a provider's pool.

    Adapted from reference PooledCredential, simplified to NIA's needs:
    no JWT claim decoding, no provider-specific sync hooks (claude_code,
    codex, nous, xai). Those can be added later as NIA grows OAuth providers.
    """

    provider: str
    id: str
    label: str
    auth_type: str  # "api_key" | "oauth"
    priority: int
    source: str  # "manual" | "env" | "device_code"
    access_token: str
    refresh_token: Optional[str] = None
    last_status: str = STATUS_OK
    last_status_at: Optional[float] = None
    last_error_code: Optional[int] = None
    last_error_reason: Optional[str] = None
    last_error_message: Optional[str] = None
    last_error_reset_at: Optional[float] = None
    base_url: Optional[str] = None
    expires_at: Optional[str] = None
    request_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, provider: str, payload: Dict[str, Any]) -> "PooledCredential":
        """Reconstruct from a dict (pool file or env singleton)."""
        # Normalize last_status_at: ISO string → epoch float.
        lsa = payload.get("last_status_at")
        if isinstance(lsa, str):
            try:
                dt = datetime.fromisoformat(lsa.replace("Z", "+00:00"))
                lsa = dt.timestamp()
            except ValueError:
                lsa = None

        return cls(
            provider=provider,
            id=payload.get("id") or uuid.uuid4().hex[:6],
            label=payload.get("label") or payload.get("source", provider),
            auth_type=payload.get("auth_type", AUTH_TYPE_API_KEY),
            priority=int(payload.get("priority", 0)),
            source=payload.get("source", SOURCE_MANUAL),
            access_token=payload.get("access_token", ""),
            refresh_token=payload.get("refresh_token"),
            last_status=payload.get("last_status", STATUS_OK),
            last_status_at=lsa,
            last_error_code=payload.get("last_error_code"),
            last_error_reason=payload.get("last_error_reason"),
            last_error_message=payload.get("last_error_message"),
            last_error_reset_at=payload.get("last_error_reset_at"),
            base_url=payload.get("base_url"),
            expires_at=payload.get("expires_at"),
            request_count=int(payload.get("request_count", 0)),
            extra={k: v for k, v in payload.items() if k not in {
                "provider", "id", "label", "auth_type", "priority", "source",
                "access_token", "refresh_token", "last_status", "last_status_at",
                "last_error_code", "last_error_reason", "last_error_message",
                "last_error_reset_at", "base_url", "expires_at", "request_count",
            } and v is not None},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence. Always emits status fields for audit trail."""
        result: Dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "auth_type": self.auth_type,
            "priority": self.priority,
            "source": self.source,
            "access_token": self.access_token,
            "last_status": self.last_status,
            "last_status_at": self.last_status_at,
            "last_error_code": self.last_error_code,
            "last_error_reason": self.last_error_reason,
            "last_error_message": self.last_error_message,
            "last_error_reset_at": self.last_error_reset_at,
            "request_count": self.request_count,
        }
        if self.refresh_token is not None:
            result["refresh_token"] = self.refresh_token
        if self.base_url is not None:
            result["base_url"] = self.base_url
        if self.expires_at is not None:
            result["expires_at"] = self.expires_at
        for k, v in self.extra.items():
            if v is not None:
                result[k] = v
        return result

    @property
    def runtime_api_key(self) -> str:
        """The API key/token to use for this credential at runtime."""
        return str(self.access_token or "")

    @property
    def runtime_base_url(self) -> Optional[str]:
        """The base URL to use for this credential at runtime."""
        return self.base_url


# ---------------------------------------------------------------------------
# CredentialPool
# ---------------------------------------------------------------------------


class CredentialPool:
    """A pool of credentials for a single provider, with rotation and exhaustion tracking.

    Thread-safe. Acquire via ``load_pool(provider)`` (which seeds from env
    singletons + the pool file). Do not construct directly unless you have
    a pre-built entries list.

    Usage pattern (in an API client)::

        pool = load_pool("anthropic")
        cred = pool.select()
        if cred is None:
            raise RuntimeError("No available anthropic credentials")
        try:
            response = client.post(..., headers={"Authorization": f"Bearer {cred.runtime_api_key}"})
            if response.status_code in (401, 429):
                next_cred = pool.mark_exhausted_and_rotate(
                    status_code=response.status_code,
                    error_context=response.json(),
                    api_key_hint=cred.runtime_api_key,
                )
                if next_cred is not None:
                    # Retry with the new credential.
                    ...
            return response
        except Exception:
            pool.release_lease(cred.id)
            raise
    """

    def __init__(self, provider: str, entries: List[PooledCredential]):
        self.provider = provider
        self._entries: List[PooledCredential] = sorted(entries, key=lambda e: e.priority)
        self._current_id: Optional[str] = None
        self._strategy: str = _get_pool_strategy(provider)
        self._lock = threading.Lock()
        self._active_leases: Dict[str, int] = {}
        self._max_concurrent = DEFAULT_MAX_CONCURRENT_PER_CREDENTIAL

    # ---- read access ------------------------------------------------------

    def has_credentials(self) -> bool:
        """True if the pool has any entries (regardless of status)."""
        return bool(self._entries)

    def has_available(self) -> bool:
        """True if at least one entry is not in exhaustion cooldown."""
        return bool(self._available_entries())

    def entries(self) -> List[PooledCredential]:
        """Return a copy of all entries."""
        return list(self._entries)

    def current(self) -> Optional[PooledCredential]:
        """Return the currently-selected entry, or None."""
        if not self._current_id:
            return None
        return next((e for e in self._entries if e.id == self._current_id), None)

    def peek(self) -> Optional[PooledCredential]:
        """Return the current entry, or the first available, without selecting."""
        cur = self.current()
        if cur is not None:
            return cur
        available = self._available_entries()
        return available[0] if available else None

    # ---- write access -----------------------------------------------------

    def _replace_entry(self, old: PooledCredential, new: PooledCredential) -> None:
        """Swap an entry in-place by id, preserving sort order."""
        for idx, entry in enumerate(self._entries):
            if entry.id == old.id:
                self._entries[idx] = new
                return

    def _persist(self) -> None:
        """Write-through to the pool file (atomic)."""
        data = {
            "version": 1,
            "provider": self.provider,
            "entries": [e.to_dict() for e in self._entries],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            _atomic_write_json(_pool_file(self.provider), data)
        except OSError as exc:
            logger.warning("Failed to persist credential pool for %s: %s", self.provider, exc)

    # ---- exhaustion tracking ----------------------------------------------

    def _is_terminal_auth_failure(
        self,
        status_code: Optional[int],
        normalized_error: Dict[str, Any],
    ) -> bool:
        """Detect upstream-permanent OAuth failures that won't recover on TTL.

        Only fires for 401 responses whose error code/reason matches a known
        terminal OAuth state. Returns False for non-401 status codes — 429
        rate limits and 402 billing failures are transient by nature.
        """
        if status_code != 401:
            return False
        reason = normalized_error.get("reason")
        if not isinstance(reason, str):
            return False
        return reason.strip().lower() in _TERMINAL_AUTH_REASONS

    def _mark_exhausted(
        self,
        entry: PooledCredential,
        status_code: Optional[int],
        error_context: Optional[Dict[str, Any]] = None,
    ) -> PooledCredential:
        """Mark an entry as exhausted (or DEAD for terminal OAuth failures)."""
        normalized = _normalize_error_context(error_context)
        # Permanent OAuth failures transition to STATUS_DEAD instead of
        # STATUS_EXHAUSTED. Without this, a revoked credential gets a 1-hour
        # TTL cooldown and then re-enters rotation, failing immediately every
        # hour until the user manually removes it.
        if self._is_terminal_auth_failure(status_code, normalized):
            terminal_status = STATUS_DEAD
        else:
            terminal_status = STATUS_EXHAUSTED

        reset_at = normalized.get("reset_at")
        # If no explicit reset_at, compute from TTL.
        if reset_at is None and terminal_status == STATUS_EXHAUSTED:
            reset_at = time.time() + _exhausted_ttl(status_code)

        updated = replace(
            entry,
            last_status=terminal_status,
            last_status_at=time.time(),
            last_error_code=status_code,
            last_error_reason=normalized.get("reason"),
            last_error_message=normalized.get("message"),
            last_error_reset_at=reset_at,
        )
        self._replace_entry(entry, updated)
        self._persist()
        return updated

    # ---- selection --------------------------------------------------------

    def select(self) -> Optional[PooledCredential]:
        """Select and return the best available credential.

        Returns None if no credentials are available (all exhausted or pool
        is empty). The selected credential becomes ``current()`` until
        ``mark_exhausted_and_rotate`` or another ``select`` is called.
        """
        with self._lock:
            return self._select_unlocked()

    def _available_entries(self, *, clear_expired: bool = False) -> List[PooledCredential]:
        """Return entries not currently in exhaustion cooldown.

        When *clear_expired* is True, entries whose cooldown has elapsed are
        reset to STATUS_OK and persisted.
        """
        now = time.time()
        cleared_any = False
        entries_to_prune: List[str] = []
        available: List[PooledCredential] = []

        for entry in self._entries:
            # DEAD entries: prune manual ones after 24h, never rotate.
            if entry.last_status == STATUS_DEAD:
                if _is_manual_source(entry.source):
                    dead_at = entry.last_status_at or 0
                    if dead_at and now - dead_at > DEAD_MANUAL_PRUNE_TTL_SECONDS:
                        entries_to_prune.append(entry.id)
                        cleared_any = True
                continue

            # EXHAUSTED entries: skip if still in cooldown, clear if expired.
            if entry.last_status == STATUS_EXHAUSTED:
                until = _exhausted_until(entry)
                if until is not None and now < until:
                    continue
                if clear_expired:
                    cleared = replace(
                        entry,
                        last_status=STATUS_OK,
                        last_status_at=None,
                        last_error_code=None,
                        last_error_reason=None,
                        last_error_message=None,
                        last_error_reset_at=None,
                    )
                    self._replace_entry(entry, cleared)
                    entry = cleared
                    cleared_any = True

            available.append(entry)

        if entries_to_prune:
            prune_set = set(entries_to_prune)
            self._entries = [e for e in self._entries if e.id not in prune_set]
        if cleared_any:
            self._persist()

        return available

    def _select_unlocked(self) -> Optional[PooledCredential]:
        """Select the best available entry, applying the rotation strategy."""
        available = self._available_entries(clear_expired=True)
        if not available:
            self._current_id = None
            logger.info("credential pool[%s]: no available entries", self.provider)
            return None

        if self._strategy == STRATEGY_RANDOM:
            entry = random.choice(available)
        elif self._strategy == STRATEGY_LEAST_USED and len(available) > 1:
            entry = min(available, key=lambda e: e.request_count)
            # Increment usage counter for load distribution.
            updated = replace(entry, request_count=entry.request_count + 1)
            self._replace_entry(entry, updated)
            entry = updated
        elif self._strategy == STRATEGY_ROUND_ROBIN and len(available) > 1:
            # Rotate: take the first, move it to the end (highest priority).
            entry = available[0]
            rotated = [e for e in self._entries if e.id != entry.id]
            rotated.append(replace(entry, priority=len(self._entries) - 1))
            self._entries = [replace(e, priority=idx) for idx, e in enumerate(rotated)]
            self._persist()
        else:  # STRATEGY_FILL_FIRST or fallback
            entry = available[0]

        self._current_id = entry.id
        return entry

    def mark_exhausted_and_rotate(
        self,
        *,
        status_code: Optional[int],
        error_context: Optional[Dict[str, Any]] = None,
        api_key_hint: Optional[str] = None,
    ) -> Optional[PooledCredential]:
        """Mark the current (or hinted) entry as exhausted and select the next.

        Returns the next available credential, or None if the pool is now
        fully exhausted. The ``api_key_hint`` parameter is important when
        the pool was just loaded from disk (another process may have
        already rotated) — it ensures we exhaust the *specific* key that
        failed, not the next one in rotation.
        """
        with self._lock:
            entry = None
            if api_key_hint:
                entry = next(
                    (e for e in self._entries if e.runtime_api_key == api_key_hint),
                    None,
                )
            if entry is None:
                entry = self.current() or self._select_unlocked()
            if entry is None:
                return None

            label = entry.label or entry.id[:8]
            self._mark_exhausted(entry, status_code, error_context)
            # Re-read updated entry to log the correct terminal state.
            updated = next((e for e in self._entries if e.id == entry.id), entry)
            if updated.last_status == STATUS_DEAD:
                logger.warning(
                    "credential pool[%s]: marking %s DEAD (status=%s, reason=%s) — "
                    "permanently failed, will NOT re-enter rotation until re-auth",
                    self.provider, label, status_code,
                    updated.last_error_reason or "unknown",
                )
            else:
                logger.info(
                    "credential pool[%s]: marking %s exhausted (status=%s), rotating",
                    self.provider, label, status_code,
                )
            self._current_id = None
            next_entry = self._select_unlocked()
            if next_entry:
                next_label = next_entry.label or next_entry.id[:8]
                logger.info("credential pool[%s]: rotated to %s", self.provider, next_label)
            return next_entry

    # ---- lease accounting -------------------------------------------------

    def acquire_lease(self, credential_id: Optional[str] = None) -> Optional[str]:
        """Acquire a soft lease on a credential for concurrent-request tracking.

        If a specific credential_id is provided, lease that entry directly.
        Otherwise prefer the least-leased available credential, using
        priority as a stable tie-breaker. When every credential is already
        at the soft cap, still return the least-leased one instead of blocking.
        """
        with self._lock:
            if credential_id:
                self._active_leases[credential_id] = self._active_leases.get(credential_id, 0) + 1
                self._current_id = credential_id
                return credential_id

            available = self._available_entries(clear_expired=True)
            if not available:
                return None

            below_cap = [
                e for e in available
                if self._active_leases.get(e.id, 0) < self._max_concurrent
            ]
            candidates = below_cap if below_cap else available
            chosen = min(
                candidates,
                key=lambda e: (self._active_leases.get(e.id, 0), e.priority),
            )
            self._active_leases[chosen.id] = self._active_leases.get(chosen.id, 0) + 1
            self._current_id = chosen.id
            return chosen.id

    def release_lease(self, credential_id: str) -> None:
        """Release a previously acquired credential lease."""
        with self._lock:
            count = self._active_leases.get(credential_id, 0)
            if count > 0:
                self._active_leases[credential_id] = count - 1
                if self._active_leases[credential_id] <= 0:
                    self._active_leases.pop(credential_id, None)

    # ---- maintenance ------------------------------------------------------

    def reset_statuses(self) -> int:
        """Reset all exhausted/dead entries back to OK. Returns the count reset."""
        with self._lock:
            count = 0
            for idx, entry in enumerate(self._entries):
                if entry.last_status != STATUS_OK:
                    self._entries[idx] = replace(
                        entry,
                        last_status=STATUS_OK,
                        last_status_at=None,
                        last_error_code=None,
                        last_error_reason=None,
                        last_error_message=None,
                        last_error_reset_at=None,
                    )
                    count += 1
            if count:
                self._persist()
            return count

    def add_entry(self, entry: PooledCredential) -> PooledCredential:
        """Add a new credential to the pool. Returns the added entry."""
        with self._lock:
            self._entries.append(entry)
            self._entries.sort(key=lambda e: e.priority)
            self._persist()
            return entry


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _is_manual_source(source: str) -> bool:
    """True if the source is 'manual' or starts with 'manual:'."""
    normalized = (source or "").strip().lower()
    return normalized == SOURCE_MANUAL or normalized.startswith(f"{SOURCE_MANUAL}:")


def _get_pool_strategy(provider: str) -> str:
    """Get the rotation strategy for a provider from config.

    Reads ``credential_pool_strategies.<provider>`` from settings.
    Defaults to ``fill_first``.
    """
    try:
        from niaharness.config.settings import load_settings

        settings = load_settings()
        strategies = (
            settings.extra.get("credential_pool_strategies", {})
            if hasattr(settings, "extra")
            else {}
        )
        if isinstance(strategies, dict):
            strategy = strategies.get(provider, STRATEGY_FILL_FIRST)
            if strategy in SUPPORTED_POOL_STRATEGIES:
                return strategy
    except Exception:
        pass
    return STRATEGY_FILL_FIRST


# ---------------------------------------------------------------------------
# Seeding: env + manual + pool file
# ---------------------------------------------------------------------------


# Provider → env var name(s) that hold the API key.
# Multiple env vars are tried in order (the first non-empty one wins).
_PROVIDER_ENV_KEYS: Dict[str, List[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "openai-codex": ["OPENAI_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY", "VERTEX_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "together": ["TOGETHER_API_KEY"],
    "fireworks": ["FIREWORKS_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "azure": ["AZURE_OPENAI_API_KEY"],
    "bedrock": ["AWS_ACCESS_KEY_ID"],  # Bedrock uses AWS creds
    "vertex": ["VERTEX_API_KEY", "GOOGLE_API_KEY"],
    "cerebras": ["CEREBRAS_API_KEY"],
    "nvidia": ["NVIDIA_API_KEY"],
    "ollama": [],  # local, no key
    "xai": ["XAI_API_KEY", "GROK_API_KEY"],
    "perplexity": ["PERPLEXITY_API_KEY"],
    "deepinfra": ["DEEPINFRA_API_KEY"],
    "huggingface": ["HUGGINGFACE_API_KEY", "HF_API_TOKEN"],
    "opencode-zen": ["OPENCODE_ZEN_API_KEY", "ZEN_API_KEY"],
}


def _seed_from_env(provider: str) -> List[PooledCredential]:
    """Seed credential entries from environment variables.

    Each env var becomes a single pool entry with source="env". Env entries
    are always present (re-seeded on every load) and have priority 0 so
    manual entries (priority 1+) take precedence when both are configured.
    """
    entries: List[PooledCredential] = []
    env_keys = _PROVIDER_ENV_KEYS.get(provider, [])
    for key in env_keys:
        value = os.environ.get(key, "").strip()
        if not value:
            continue
        entries.append(
            PooledCredential(
                provider=provider,
                id=f"env-{key.lower()}",
                label=f"env:{key}",
                auth_type=AUTH_TYPE_API_KEY,
                priority=0,
                source=SOURCE_ENV,
                access_token=value,
            )
        )
        break  # Only one env entry per provider (first non-empty wins)
    return entries


def _read_pool_file(provider: str) -> List[PooledCredential]:
    """Read persisted entries from the pool file."""
    path = _pool_file(provider)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, dict):
        return []
    entries_data = data.get("entries", [])
    if not isinstance(entries_data, list):
        return []
    result: List[PooledCredential] = []
    for payload in entries_data:
        if not isinstance(payload, dict):
            continue
        try:
            result.append(PooledCredential.from_dict(provider, payload))
        except Exception as exc:
            logger.warning("Failed to load pool entry for %s: %s", provider, exc)
    return result


# Cache of loaded pools, invalidated by file mtime.
_pool_cache: Dict[str, Tuple[CredentialPool, float]] = {}
_pool_cache_lock = threading.Lock()


def load_pool(provider: str) -> CredentialPool:
    """Load a credential pool for a provider, seeding from env + pool file.

    The pool is cached for the lifetime of the process, but the cache is
    invalidated if the pool file's mtime changes (so multiple NIA processes
    can see each other's updates).

    Env entries are always re-seeded (they may change between calls if the
    process environment is modified). Manual entries are loaded from the
    pool file and merged with env entries (env entries first, then manual
    entries sorted by priority).
    """
    cache_key = provider
    pool_path = _pool_file(provider)
    try:
        current_mtime = pool_path.stat().st_mtime if pool_path.exists() else 0.0
    except OSError:
        current_mtime = 0.0

    with _pool_cache_lock:
        cached = _pool_cache.get(cache_key)
        if cached is not None:
            cached_pool, cached_mtime = cached
            if cached_mtime == current_mtime:
                # Refresh env entries in the cached pool (env may have changed).
                env_entries = _seed_from_env(provider)
                # Replace env entries in the cached pool.
                cached_pool._entries = [
                    e for e in cached_pool._entries if e.source != SOURCE_ENV
                ] + env_entries
                cached_pool._entries.sort(key=lambda e: e.priority)
                return cached_pool

        # Build a fresh pool.
        env_entries = _seed_from_env(provider)
        file_entries = _read_pool_file(provider)

        # Merge: env entries + file entries (file entries override env entries
        # with the same id, which is rare but possible if a user manually
        # edits the pool file).
        seen_ids = {e.id for e in env_entries}
        merged = list(env_entries)
        for entry in file_entries:
            if entry.id not in seen_ids:
                merged.append(entry)
                seen_ids.add(entry.id)

        pool = CredentialPool(provider, merged)
        _pool_cache[cache_key] = (pool, current_mtime)
        return pool


def write_credential_pool(provider: str, entries: List[Dict[str, Any]]) -> None:
    """Write a list of entries to the pool file (atomic)."""
    data = {
        "version": 1,
        "provider": provider,
        "entries": entries,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(_pool_file(provider), data)
    # Invalidate cache.
    with _pool_cache_lock:
        _pool_cache.pop(provider, None)


def invalidate_pool_cache(provider: Optional[str] = None) -> None:
    """Invalidate the pool cache for a provider (or all providers)."""
    with _pool_cache_lock:
        if provider is None:
            _pool_cache.clear()
        else:
            _pool_cache.pop(provider, None)


__all__ = [
    "AUTH_TYPE_API_KEY",
    "AUTH_TYPE_OAUTH",
    "SOURCE_DEVICE_CODE",
    "SOURCE_ENV",
    "SOURCE_MANUAL",
    "STATUS_DEAD",
    "STATUS_EXHAUSTED",
    "STATUS_OK",
    "STRATEGY_FILL_FIRST",
    "STRATEGY_LEAST_USED",
    "STRATEGY_RANDOM",
    "STRATEGY_ROUND_ROBIN",
    "SUPPORTED_POOL_STRATEGIES",
    "CredentialPool",
    "PooledCredential",
    "invalidate_pool_cache",
    "load_pool",
    "write_credential_pool",
]
