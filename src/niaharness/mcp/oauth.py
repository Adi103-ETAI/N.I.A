"""MCP OAuth 2.1 + PKCE — token storage, browser flow, and auth provider.

Ported from Hermes Agent's ``tools/mcp_oauth.py`` (948 LOC), adapted to
NIA's architecture. Provides:

  - :class:`NiaTokenStorage` — disk persistence under
    ``~/.nia/mcp-tokens/`` (or ``~/.nia/profiles/<name>/mcp-tokens/`` for
    named profiles). Three files per server: ``<name>.json`` (tokens),
    ``<name>.client.json`` (client registration), ``<name>.meta.json``
    (OAuth server metadata). Atomic writes at 0o600.
  - :func:`build_oauth_auth` — builds an ``httpx.Auth``-compatible
    :class:`OAuthClientProvider` for an MCP server. The MCP SDK drives
    the PKCE flow internally; we supply the storage + redirect handler +
    callback handler.
  - Browser flow + stdin paste fallback for SSH/headless environments.
  - :class:`OAuthNonInteractiveError` — raised when the flow needs a
    human but none is reachable (background contexts, cron, etc.).

Token file format:
  ``<name>.json`` contains ``access_token``, ``token_type``, ``expires_in``,
  ``refresh_token`` (if present), ``scope`` (if present), plus a custom
  ``expires_at`` field (absolute wall-clock epoch seconds) so the SDK's
  ``is_token_valid()`` correctly reports False after a process restart.

Usage::

    from niaharness.mcp.oauth import build_oauth_auth

    auth = build_oauth_auth("notion", "https://api.notion.com/mcp", oauth_config={
        "client_id": "my-client-id",
        "scope": "read write",
    })
    # Pass `auth` to the MCP SDK client as the `auth=` parameter.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import re
import secrets
import socket
import stat
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy SDK imports — OAuth is optional (requires mcp>=1.26.0)
# ---------------------------------------------------------------------------

_OAUTH_AVAILABLE = False
OAuthClientProvider = None
OAuthClientMetadata = None
OAuthClientInformationFull = None
OAuthMetadata = None
OAuthToken = None
AnyUrl = None

try:
    from mcp.client.auth import OAuthClientProvider  # type: ignore[assignment]
    from mcp.shared.auth import (  # type: ignore[assignment]
        OAuthClientInformationFull,
        OAuthClientMetadata,
        OAuthMetadata,
        OAuthToken,
    )
    _OAUTH_AVAILABLE = True
except ImportError:
    pass

try:
    from pydantic import AnyUrl  # type: ignore[assignment]
except ImportError:
    AnyUrl = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants + ContextVars
# ---------------------------------------------------------------------------

# Skip tokens accepted at the paste prompt.
_SKIP_TOKENS = frozenset({"skip", "cancel", "s", "n", "no", "q", "quit"})
_USER_SKIPPED_SENTINEL = "__nia_user_skipped__"

# Per-context interactivity flags (ContextVar, not threading.local, so they
# propagate across run_coroutine_threadsafe boundaries).
_oauth_interactive_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "nia_oauth_interactive_enabled", default=True
)
_oauth_interactive_forced: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "nia_oauth_interactive_forced", default=False
)


class OAuthNonInteractiveError(RuntimeError):
    """Raised when the OAuth flow needs a human but none is reachable."""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _get_token_dir() -> Path:
    """Return the token storage directory.

    Profile-aware: uses ``~/.nia/mcp-tokens/`` for the default profile,
    ``~/.nia/profiles/<name>/mcp-tokens/`` for named profiles.
    """
    try:
        from niaharness.profiles import get_profile

        profile = get_profile()
        return profile.mcp_tokens_dir
    except Exception:
        try:
            from niaharness.prompts.soul import get_nia_home

            return get_nia_home() / "mcp-tokens"
        except Exception:
            return Path(os.path.expanduser("~/.nia/mcp-tokens"))


def _safe_filename(name: str) -> str:
    """Sanitize a server name for use as a filename."""
    if not name:
        return "default"
    safe = re.sub(r"[^\w\-]", "_", name)
    safe = safe.strip("_")
    if not safe:
        return "default"
    return safe[:128]


def _read_json(path: Path) -> Optional[dict]:
    """Read a JSON file, returning None on missing/corrupt."""
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def _write_json(path: Path, data: dict) -> None:
    """Atomically write JSON at 0o600 (owner-only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(stat.S_IRWXU)  # 0o700
    except OSError:
        pass

    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        fd = os.open(
            str(tmp),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,  # 0o600
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(path))
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Token storage
# ---------------------------------------------------------------------------


class NiaTokenStorage:
    """Persist OAuth tokens and client registration to JSON files.

    File layout::

        <token_dir>/<server_name>.json         — tokens (OAuthToken + expires_at)
        <token_dir>/<server_name>.client.json  — client registration (RFC 7591)
        <token_dir>/<server_name>.meta.json    — OAuth server metadata
    """

    def __init__(self, server_name: str) -> None:
        self._server_name = _safe_filename(server_name)

    def _tokens_path(self) -> Path:
        return _get_token_dir() / f"{self._server_name}.json"

    def _client_info_path(self) -> Path:
        return _get_token_dir() / f"{self._server_name}.client.json"

    def _meta_path(self) -> Path:
        return _get_token_dir() / f"{self._server_name}.meta.json"

    # -- tokens ------------------------------------------------------------

    async def get_tokens(self) -> Any:
        """Read tokens from disk, recomputing ``expires_in`` from ``expires_at``."""
        if OAuthToken is None:
            return None
        data = _read_json(self._tokens_path())
        if data is None:
            return None
        # Recompute expires_in from the persisted absolute expires_at.
        absolute_expiry = data.pop("expires_at", None)
        if absolute_expiry is not None:
            try:
                data["expires_in"] = int(max(float(absolute_expiry) - time.time(), 0))
            except (TypeError, ValueError):
                pass
        elif data.get("expires_in") is not None:
            # Legacy file without expires_at — estimate from file mtime.
            try:
                file_mtime = self._tokens_path().stat().st_mtime
                data["expires_in"] = int(max(file_mtime + int(data["expires_in"]) - time.time(), 0))
            except (OSError, TypeError, ValueError):
                pass
        try:
            return OAuthToken.model_validate(data)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Corrupt tokens at %s — ignoring: %s", self._tokens_path(), exc)
            return None

    async def set_tokens(self, tokens: Any) -> None:
        """Write tokens to disk with an absolute ``expires_at`` timestamp."""
        payload = tokens.model_dump(mode="json", exclude_none=True)
        expires_in = payload.get("expires_in")
        if expires_in is not None:
            try:
                payload["expires_at"] = time.time() + int(expires_in)
            except (TypeError, ValueError):
                pass
        _write_json(self._tokens_path(), payload)
        logger.debug("OAuth tokens saved for %s", self._server_name)

    # -- client info -------------------------------------------------------

    async def get_client_info(self) -> Any:
        if OAuthClientInformationFull is None:
            return None
        data = _read_json(self._client_info_path())
        if data is None:
            return None
        try:
            return OAuthClientInformationFull.model_validate(data)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Corrupt client info at %s — ignoring: %s", self._client_info_path(), exc)
            return None

    async def set_client_info(self, client_info: Any) -> None:
        _write_json(
            self._client_info_path(),
            client_info.model_dump(mode="json", exclude_none=True),
        )

    # -- OAuth server metadata --------------------------------------------

    def save_oauth_metadata(self, metadata: Any) -> None:
        _write_json(self._meta_path(), metadata.model_dump(exclude_none=True, mode="json"))

    def load_oauth_metadata(self) -> Any:
        if OAuthMetadata is None:
            return None
        data = _read_json(self._meta_path())
        if data is None:
            return None
        try:
            return OAuthMetadata.model_validate(data)
        except (ValueError, TypeError, KeyError):
            return None

    # -- cleanup -----------------------------------------------------------

    def remove(self) -> None:
        """Delete all stored OAuth state for this server."""
        for p in (self._tokens_path(), self._client_info_path(), self._meta_path()):
            p.unlink(missing_ok=True)

    def has_cached_tokens(self) -> bool:
        """Return True if tokens exist on disk (may be expired)."""
        return self._tokens_path().exists()

    def poison_client_registration(self) -> bool:
        """Back up + delete client.json + meta.json to force re-registration.

        Used by the ``invalid_client`` auto-heal: when the token endpoint
        returns ``invalid_client``, the cached client registration is
        stale (revoked, rotated, or never valid). Backing up to ``.bak``
        and deleting forces RFC 7591 dynamic registration on the next
        flow. Tokens are left intact — they may still be valid.

        Returns True if a client file existed.
        """
        client_path = self._client_info_path()
        if not client_path.exists():
            return False
        backup = client_path.with_name(client_path.name + ".bak")
        try:
            backup.write_bytes(client_path.read_bytes())
        except OSError as exc:
            logger.warning("Could not back up client info at %s: %s", client_path, exc)
        client_path.unlink(missing_ok=True)
        self._meta_path().unlink(missing_ok=True)
        logger.warning(
            "MCP OAuth '%s': cached client registration rejected as invalid_client; "
            "removed client.json + meta.json (backup at %s) to force re-registration",
            self._server_name, backup.name,
        )
        return True


# ---------------------------------------------------------------------------
# Callback flow — browser + stdin paste fallback
# ---------------------------------------------------------------------------


def _make_callback_handler() -> tuple[type, dict]:
    """Build a fresh BaseHTTPRequestHandler + result dict for one flow.

    Returns a NEW pair each call so concurrent flows don't stomp on each
    other's result dict.
    """
    result: dict[str, Any] = {"auth_code": None, "state": None, "error": None}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — http.server API
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            if "code" in params:
                result["auth_code"] = params["code"][0]
            if "state" in params:
                result["state"] = params["state"][0]
            if "error" in params:
                result["error"] = params["error"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Authorization complete.</h2>"
                b"<p>You can close this tab and return to NIA.</p>"
                b"</body></html>"
            )

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("OAuth callback: " + fmt, *args)

    return _Handler, result


def _find_free_port() -> int:
    """Pick a free localhost port for the callback server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _can_open_browser() -> bool:
    """Heuristic: can we open a browser on this system?"""
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"):
        return False
    if sys.platform == "darwin" or sys.platform == "win32":
        return True
    # Linux: need a display.
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _is_interactive() -> bool:
    """Return True if we can interact with the user (TTY or forced)."""
    if not _oauth_interactive_enabled.get():
        return False
    if _oauth_interactive_forced.get():
        return True
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _raise_if_non_interactive(lead: str) -> None:
    """Raise OAuthNonInteractiveError if we can't interact with the user."""
    if not _is_interactive():
        raise OAuthNonInteractiveError(
            f"{lead}. Run `nia mcp login <server>` interactively first "
            "to complete initial authorization, then cached tokens will be reused."
        )


async def _redirect_handler(authorization_url: str) -> None:
    """Display / open the authorization URL in a browser."""
    _raise_if_non_interactive("OAuth authorization requires an interactive session")
    print(f"\n  Authorization URL: {authorization_url}\n", file=sys.stderr, flush=True)
    if _can_open_browser():
        try:
            webbrowser.open(authorization_url)
        except Exception:
            pass  # Best-effort.


async def _wait_for_callback(port: int) -> tuple[str, Optional[str]]:
    """Wait for the OAuth callback on localhost:*port*.

    Returns ``(auth_code, state)``. Raises ``OAuthNonInteractiveError`` on
    timeout or if non-interactive.
    """
    _raise_if_non_interactive("OAuth callback requires an interactive session")

    handler_cls, result = _make_callback_handler()
    try:
        server = HTTPServer(("127.0.0.1", port), handler_cls)
    except OSError:
        raise OAuthNonInteractiveError(
            f"Could not bind OAuth callback port {port} — it may be in use."
        )

    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    # Stdin paste fallback for SSH/headless.
    paste_thread: Optional[threading.Thread] = None
    if _is_interactive():
        print(
            "\n  Or paste the redirect URL here (or the ``?code=...&state=...`` "
            "portion) and press Enter. Type ``skip`` + Enter to continue without "
            "this server:",
            file=sys.stderr,
            flush=True,
        )
        paste_thread = threading.Thread(
            target=_paste_callback_reader, args=(result,), daemon=True
        )
        paste_thread.start()

    timeout = 300.0
    poll_interval = 0.5
    elapsed = 0.0
    try:
        while elapsed < timeout:
            if result["auth_code"] is not None or result["error"] is not None:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
    finally:
        server.server_close()

    if result["error"] == _USER_SKIPPED_SENTINEL:
        raise OAuthNonInteractiveError("user_skipped")
    if result["error"]:
        raise RuntimeError(f"OAuth authorization failed: {result['error']}")
    if result["auth_code"] is None:
        raise OAuthNonInteractiveError(
            "OAuth callback timed out — no authorization code received."
        )
    return result["auth_code"], result["state"]


def _paste_callback_reader(result: dict) -> None:
    """Read one line from stdin as a fallback for the callback server."""
    try:
        line = sys.stdin.readline().strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not line:
        return
    if line.lower() in _SKIP_TOKENS:
        result["error"] = _USER_SKIPPED_SENTINEL
        return

    # Accept: full redirect URL, ?code=...&state=..., bare code=...&state=...
    if "?" in line:
        line = line.split("?", 1)[1]
    elif line.startswith("code="):
        pass  # Already bare.
    else:
        # Try parsing as a URL.
        try:
            parsed = urlparse(line)
            line = parsed.query
        except Exception:
            return

    params = parse_qs(line)
    # Only write if the HTTP listener hasn't already won.
    if result["auth_code"] is None and result["error"] is None:
        if "code" in params:
            result["auth_code"] = params["code"][0]
        if "state" in params:
            result["state"] = params["state"][0]
        if "error" in params:
            result["error"] = params["error"][0]


# ---------------------------------------------------------------------------
# Context managers for interactivity control
# ---------------------------------------------------------------------------


def suppress_interactive_oauth() -> contextvars.Token[bool]:
    """Suppress stdin-based OAuth prompts (for background contexts)."""
    return _oauth_interactive_enabled.set(False)


def restore_interactive_oauth(token: contextvars.Token[bool]) -> None:
    """Restore the prior interactivity context."""
    _oauth_interactive_enabled.reset(token)


def force_interactive_oauth() -> contextvars.Token[bool]:
    """Force interactive mode past the TTY check (for GUI-driven flows)."""
    return _oauth_interactive_forced.set(True)


def restore_forced_interactive(token: contextvars.Token[bool]) -> None:
    _oauth_interactive_forced.reset(token)


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


def _configure_callback_port(cfg: dict) -> int:
    """Resolve the callback port (0 = auto-pick)."""
    port = int(cfg.get("redirect_port", 0) or 0)
    if port == 0:
        port = _find_free_port()
    cfg["_resolved_port"] = port
    return port


def _build_client_metadata(cfg: dict) -> Any:
    """Build OAuthClientMetadata from the config dict."""
    if OAuthClientMetadata is None or AnyUrl is None:
        return None
    port = cfg.get("_resolved_port", 0)
    redirect_uri = AnyUrl(f"http://127.0.0.1:{port}/callback")
    kwargs: dict[str, Any] = {
        "client_name": cfg.get("client_name", "NIA Agent"),
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    if cfg.get("client_secret"):
        kwargs["token_endpoint_auth_method"] = "client_secret_post"
    if cfg.get("scope"):
        kwargs["scope"] = cfg["scope"]
    return OAuthClientMetadata(**kwargs)


def _maybe_preregister_client(
    storage: NiaTokenStorage,
    cfg: dict,
    client_metadata: Any,
) -> None:
    """If client_id is set in config, persist it to skip dynamic registration."""
    if not cfg.get("client_id") or OAuthClientInformationFull is None:
        return
    try:
        client_info = OAuthClientInformationFull(
            client_id=cfg["client_id"],
            client_secret=cfg.get("client_secret"),
            client_name=cfg.get("client_name", "NIA Agent"),
            redirect_uris=client_metadata.redirect_uris,
            grant_types=client_metadata.grant_types,
            response_types=client_metadata.response_types,
            token_endpoint_auth_method=client_metadata.token_endpoint_auth_method,
            scope=cfg.get("scope"),
        )
        # Write synchronously (the storage's async setter isn't available here).
        _write_json(
            storage._client_info_path(),
            client_info.model_dump(mode="json", exclude_none=True),
        )
    except Exception as exc:
        logger.debug("Could not pre-register client: %s", exc)


def build_oauth_auth(
    server_name: str,
    server_url: str,
    oauth_config: Optional[dict] = None,
) -> Any:
    """Build an ``httpx.Auth``-compatible OAuth provider for an MCP server.

    Returns ``None`` if the MCP SDK OAuth types are not available (install
    with ``pip install 'mcp>=1.26.0'``). Raises
    :class:`OAuthNonInteractiveError` if non-interactive AND no cached
    tokens.

    Args:
        server_name: Logical server name (used for token file naming).
        server_url: The MCP server URL (used for OAuth metadata discovery).
        oauth_config: Optional config dict with ``client_id``,
            ``client_secret``, ``client_name``, ``scope``,
            ``redirect_port``, ``timeout``.

    Returns:
        An :class:`OAuthClientProvider` instance (an ``httpx.Auth``
        subclass) suitable for passing to the MCP SDK client as ``auth=``.
    """
    if not _OAUTH_AVAILABLE:
        logger.warning(
            "MCP OAuth requested for '%s' but SDK auth types are not available. "
            "Install with: pip install 'mcp>=1.26.0'",
            server_name,
        )
        return None

    cfg = dict(oauth_config or {})
    storage = NiaTokenStorage(server_name)

    if not _is_interactive() and not storage.has_cached_tokens():
        raise OAuthNonInteractiveError(
            f"MCP OAuth for '{server_name}': non-interactive environment and no "
            "cached tokens found. Run `nia mcp login <server>` interactively first."
        )

    port = _configure_callback_port(cfg)
    client_metadata = _build_client_metadata(cfg)
    _maybe_preregister_client(storage, cfg, client_metadata)

    # Build a per-flow callback handler bound to the resolved port.
    async def _callback() -> tuple[str, Optional[str]]:
        return await _wait_for_callback(port)

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=_redirect_handler,
        callback_handler=_callback,
        timeout=float(cfg.get("timeout", 300)),
    )


def remove_oauth_tokens(server_name: str) -> None:
    """Delete all stored OAuth state for a server."""
    NiaTokenStorage(server_name).remove()


__all__ = [
    "NiaTokenStorage",
    "OAuthNonInteractiveError",
    "build_oauth_auth",
    "force_interactive_oauth",
    "remove_oauth_tokens",
    "restore_forced_interactive",
    "restore_interactive_oauth",
    "suppress_interactive_oauth",
    # Internal (for the manager + tests):
    "_build_client_metadata",
    "_configure_callback_port",
    "_get_token_dir",
    "_is_interactive",
    "_maybe_preregister_client",
    "_oauth_interactive_enabled",
    "_oauth_interactive_forced",
    "_redirect_handler",
    "_safe_filename",
    "_wait_for_callback",
    "_OAUTH_AVAILABLE",
]
