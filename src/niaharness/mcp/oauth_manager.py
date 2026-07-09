"""MCP OAuth manager — singleton with cross-process token reload + 401 dedup.

Ported from Hermes Agent's ``tools/mcp_oauth_manager.py`` (720 LOC),
adapted to NIA's architecture. Provides:

  - :class:`MCPOAuthManager` — process-wide singleton that caches per-server
    OAuth providers, watches the token file mtime for cross-process changes,
    and deduplicates concurrent 401-handlers via in-flight futures.
  - Cross-process token reload — if another process (e.g. ``nia mcp login``)
    refreshes the token, the manager detects the mtime change on the next
    request and forces the SDK to re-read from disk.
  - 401 dedup — when N concurrent tool calls hit 401 with the same access
    token, only ONE refresh handler runs; all callers await the same future.
  - ``invalid_client`` auto-heal — when the token endpoint returns
    ``invalid_client``, the cached client registration is backed up + deleted,
    forcing RFC 7591 dynamic registration on the next flow.

The manager is thread-safe: the ``_entries`` dict is guarded by a
``threading.Lock`` for get-or-create semantics. Per-entry state is guarded
by the entry's own ``asyncio.Lock`` (used from the MCP event loop thread).
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-server provider entry
# ---------------------------------------------------------------------------


@dataclass
class _ProviderEntry:
    """Per-server OAuth state."""

    server_url: str
    oauth_config: Optional[dict]
    provider: Optional[Any] = None
    last_mtime_ns: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_401: dict[str, "asyncio.Future[bool]"] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _same_endpoint(a: str, b: str) -> bool:
    """Return True if two URLs have the same scheme + host + path."""
    try:
        pa = urlsplit(a)
        pb = urlsplit(b)
        return (
            pa.scheme.lower() == pb.scheme.lower()
            and pa.netloc.lower() == pb.netloc.lower()
            and pa.path.rstrip("/") == pb.path.rstrip("/")
        )
    except Exception:
        return a == b


# ---------------------------------------------------------------------------
# MCPOAuthManager — the singleton
# ---------------------------------------------------------------------------


class MCPOAuthManager:
    """Single source of truth for per-server MCP OAuth state.

    Thread-safe: the ``_entries`` dict is guarded by ``_entries_lock`` for
    get-or-create semantics. Per-entry state is guarded by the entry's own
    ``asyncio.Lock`` (used from the MCP event loop thread).
    """

    def __init__(self) -> None:
        self._entries: dict[str, _ProviderEntry] = {}
        self._entries_lock = threading.Lock()
        # Strong references to in-flight 401 handler tasks so the event
        # loop's weak-ref bookkeeping can't GC them mid-run.
        self._inflight_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Provider construction / caching
    # ------------------------------------------------------------------

    def get_or_build_provider(
        self,
        server_name: str,
        server_url: str,
        oauth_config: Optional[dict],
    ) -> Optional[Any]:
        """Return a cached OAuth provider for *server_name* or build one.

        Discards the cache if *server_url* changes (the old provider's
        tokens are for a different server).
        """
        with self._entries_lock:
            entry = self._entries.get(server_name)
            if entry is not None and entry.server_url != server_url:
                logger.info(
                    "MCP OAuth '%s': URL changed from %s to %s, discarding cache",
                    server_name, entry.server_url, server_url,
                )
                entry = None

            if entry is None:
                entry = _ProviderEntry(
                    server_url=server_url,
                    oauth_config=oauth_config,
                )
                self._entries[server_name] = entry

            if entry.provider is None:
                entry.provider = self._build_provider(server_name, entry)

            return entry.provider

    def _build_provider(
        self,
        server_name: str,
        entry: _ProviderEntry,
    ) -> Optional[Any]:
        """Build the underlying OAuth provider via :func:`build_oauth_auth`."""
        from niaharness.mcp.oauth import (
            OAuthNonInteractiveError,
            _OAUTH_AVAILABLE,
            _is_interactive,
            build_oauth_auth,
        )

        if not _OAUTH_AVAILABLE:
            logger.warning("MCP OAuth '%s': SDK auth module unavailable", server_name)
            return None

        # Check interactivity before building — if non-interactive and no
        # cached tokens, raise so the caller can surface a helpful error.
        storage = _get_storage(server_name)
        if not _is_interactive() and not storage.has_cached_tokens():
            raise OAuthNonInteractiveError(
                f"MCP OAuth for '{server_name}': non-interactive environment and "
                "no cached tokens. Run `nia mcp login <server>` interactively first."
            )

        return build_oauth_auth(
            server_name=server_name,
            server_url=entry.server_url,
            oauth_config=entry.oauth_config,
        )

    def remove(self, server_name: str) -> None:
        """Evict the provider from cache AND delete tokens from disk."""
        with self._entries_lock:
            self._entries.pop(server_name, None)
        from niaharness.mcp.oauth import remove_oauth_tokens

        remove_oauth_tokens(server_name)
        logger.info("MCP OAuth '%s': evicted from cache and removed from disk", server_name)

    # ------------------------------------------------------------------
    # Cross-process disk watch
    # ------------------------------------------------------------------

    async def invalidate_if_disk_changed(self, server_name: str) -> bool:
        """If the tokens file on disk has a newer mtime, force a reload.

        Called from the MCP event loop before each request. If another
        process refreshed the token (e.g. ``nia mcp login``), the mtime
        changes and we set ``provider._initialized = False`` so the SDK
        re-reads tokens from disk on the next ``async_auth_flow``.

        Returns True if the disk changed (caller should retry).
        """
        from niaharness.mcp.oauth import _get_token_dir, _safe_filename

        entry = self._entries.get(server_name)
        if entry is None or entry.provider is None:
            return False

        async with entry.lock:
            tokens_path = _get_token_dir() / f"{_safe_filename(server_name)}.json"
            try:
                mtime_ns = tokens_path.stat().st_mtime_ns
            except (FileNotFoundError, OSError):
                return False

            if mtime_ns != entry.last_mtime_ns:
                old = entry.last_mtime_ns
                entry.last_mtime_ns = mtime_ns
                # Force the SDK to re-initialize (re-read tokens from disk).
                if hasattr(entry.provider, "_initialized"):
                    entry.provider._initialized = False  # type: ignore[attr-defined]
                logger.info(
                    "MCP OAuth '%s': tokens file changed (mtime %d -> %d), forcing reload",
                    server_name, old, mtime_ns,
                )
                return True
            return False

    # ------------------------------------------------------------------
    # 401 handler (deduplicated via in-flight futures)
    # ------------------------------------------------------------------

    async def handle_401(
        self,
        server_name: str,
        failed_access_token: Optional[str] = None,
    ) -> bool:
        """Handle a 401 from a tool call, deduplicated across concurrent callers.

        When N concurrent tool calls hit 401 with the same access token,
        only the FIRST caller runs the refresh handler; all others await
        the same future. The handler:
          1. Checks if the disk changed (another process refreshed).
          2. If not, checks if the SDK can self-refresh.

        Returns True if the caller should retry (disk changed or SDK can
        refresh). Returns False if re-auth is needed (surface to the user).
        """
        entry = self._entries.get(server_name)
        if entry is None or entry.provider is None:
            return False

        key = failed_access_token or "<unknown>"
        loop = asyncio.get_running_loop()

        async with entry.lock:
            pending = entry.pending_401.get(key)
            if pending is None:
                # This caller is the FIRST — create the future + schedule the handler.
                pending = loop.create_future()
                entry.pending_401[key] = pending

                async def _do_handle() -> None:
                    try:
                        # Step 1: Did the disk change? (Another process refreshed.)
                        disk_changed = await self.invalidate_if_disk_changed(server_name)
                        if disk_changed:
                            if not pending.done():
                                pending.set_result(True)
                            return

                        # Step 2: No disk change — can the SDK self-refresh?
                        provider = entry.provider
                        ctx = getattr(provider, "context", None)
                        can_refresh = False
                        if ctx is not None:
                            can_refresh_fn = getattr(ctx, "can_refresh_token", None)
                            if callable(can_refresh_fn):
                                try:
                                    can_refresh = bool(can_refresh_fn())
                                except Exception:
                                    can_refresh = False
                        if not pending.done():
                            pending.set_result(can_refresh)
                    except Exception as exc:
                        logger.warning("MCP OAuth '%s': 401 handler failed: %s", server_name, exc)
                        if not pending.done():
                            pending.set_result(False)
                    finally:
                        entry.pending_401.pop(key, None)

                task = asyncio.create_task(_do_handle())
                # Strong-reference guard: prevent the event loop's weak-ref
                # bookkeeping from GC-ing the task mid-run.
                self._inflight_tasks.add(task)
                task.add_done_callback(self._inflight_tasks.discard)

        # Await OUTSIDE the lock — concurrent callers don't serialize.
        try:
            return await pending
        except Exception as exc:
            logger.warning("MCP OAuth '%s': awaiting 401 handler failed: %s", server_name, exc)
            return False

    # ------------------------------------------------------------------
    # invalid_client auto-heal
    # ------------------------------------------------------------------

    async def maybe_flag_poisoned_client(
        self,
        server_name: str,
        response: Any,
        token_endpoint: Optional[str],
    ) -> bool:
        """Detect ``invalid_client`` from the token endpoint and auto-heal.

        When the token endpoint returns 400/401 with ``invalid_client`` in
        the body, the cached client registration is stale. This backs up +
        deletes ``client.json`` + ``meta.json``, forcing RFC 7591 dynamic
        registration on the next flow.

        Returns True if the client was poisoned (caller should retry).
        """
        from niaharness.mcp.oauth import NiaTokenStorage

        try:
            if response is None or token_endpoint is None:
                return False
            status_code = getattr(response, "status_code", None)
            if status_code not in (400, 401):
                return False
            # Check if the request URL matches the token endpoint.
            request = getattr(response, "request", None)
            if request is None:
                return False
            request_url = str(getattr(request, "url", ""))
            if not _same_endpoint(request_url, token_endpoint):
                return False
            # Check the response body for invalid_client (word-boundary match
            # so invalid_client_metadata doesn't trip it).
            body = b""
            try:
                content = getattr(response, "content", None)
                if content:
                    body = content if isinstance(content, bytes) else str(content).encode()
                else:
                    body = getattr(response, "_content", b"") or b""
            except Exception:
                pass
            if not re.search(rb"\binvalid_client\b", body.lower()):
                return False
            # Auto-heal: poison the client registration.
            storage = NiaTokenStorage(server_name)
            poisoned = storage.poison_client_registration()
            if poisoned:
                # Force re-initialization on the next flow.
                entry = self._entries.get(server_name)
                if entry and entry.provider and hasattr(entry.provider, "_initialized"):
                    entry.provider._initialized = False  # type: ignore[attr-defined]
            return poisoned
        except Exception as exc:
            logger.debug("maybe_flag_poisoned_client failed (non-fatal): %s", exc)
            return False


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------


_MANAGER: Optional[MCPOAuthManager] = None
_MANAGER_LOCK = threading.Lock()


def get_manager() -> MCPOAuthManager:
    """Return the process-wide :class:`MCPOAuthManager` singleton."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = MCPOAuthManager()
        return _MANAGER


def reset_manager_for_tests() -> None:
    """Test-only helper: drop the singleton so fixtures start clean."""
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = None


# ---------------------------------------------------------------------------
# Internal helper (used by _build_provider)
# ---------------------------------------------------------------------------


def _get_storage(server_name: str):
    """Build a NiaTokenStorage for *server_name*."""
    from niaharness.mcp.oauth import NiaTokenStorage

    return NiaTokenStorage(server_name)


__all__ = [
    "MCPOAuthManager",
    "get_manager",
    "reset_manager_for_tests",
]
