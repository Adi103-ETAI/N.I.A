"""WebSocket transport for the NIA TUI gateway JSON-RPC server.

Ported from Hermes Agent's ``tui_gateway/ws.py`` (466 LOC).

Reuses :func:`tui_gateway.server.dispatch` verbatim so every RPC method,
every slash command, every approval/clarify/sudo flow, and every agent
event flows through the same handlers whether the client is Ink over
stdio or a web client over WebSocket.

Wire protocol: identical to stdio — newline-delimited JSON-RPC in both
directions. The server emits a ``gateway.ready`` event immediately after
connection accept, then echoes responses/events for inbound requests.

Token coalescing: per-token streaming frames (``message.delta``,
``reasoning.delta``, ``thinking.delta``) are buffered and flushed as a
batch on a short timer (~30fps) instead of waking the event loop once
per token. This cuts GIL churn during model streaming.

Mounting::

    from fastapi import WebSocket
    from niaharness.tui_gateway.ws import handle_ws

    @app.websocket("/api/ws")
    async def ws(ws: WebSocket):
        await handle_ws(ws)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
from typing import Any

from niaharness.tui_gateway import server
from niaharness.tui_gateway.loop_noise import install_loop_noise_filter

_log = logging.getLogger(__name__)

_WS_WRITE_TIMEOUT_S = 10.0
_WS_LOG_PAYLOAD_PREVIEW = 240

# Per-token streaming frames are coalesced: buffered and flushed as a batch.
_STREAMING_EVENT_TYPES = frozenset({
    "message.delta",
    "reasoning.delta",
    "thinking.delta",
})
_TOKEN_COALESCE_S = 0.033  # ~30 fps

try:
    from starlette.websockets import WebSocketDisconnect as _WebSocketDisconnect
except ImportError:
    _WebSocketDisconnect = Exception  # type: ignore[assignment]


class WSTransport:
    """Per-connection WebSocket transport.

    ``write`` is safe to call from any thread. Pool workers run in their own
    threads, so marshalling onto the loop via ``run_coroutine_threadsafe`` +
    ``future.result()`` is correct and deadlock-free there.

    When called from the loop thread itself, the same call would deadlock: we'd
    schedule work onto the loop we're currently blocking. We detect that case
    and fire-and-forget instead.
    """

    def __init__(
        self,
        ws: Any,
        loop: asyncio.AbstractEventLoop,
        *,
        peer: str = "unknown",
    ) -> None:
        self._ws = ws
        self._loop = loop
        self._peer = peer
        self._closed = False
        self._token_lock = threading.Lock()
        self._pending_tokens: list[str] = []
        self._token_flush_handle: asyncio.TimerHandle | None = None
        self._token_flush_armed = False

    @staticmethod
    def _is_streaming_frame(obj: dict) -> bool:
        """True for high-frequency per-token frames eligible for coalescing."""
        params = obj.get("params") if isinstance(obj, dict) else None
        if not isinstance(params, dict):
            return False
        return params.get("type") in _STREAMING_EVENT_TYPES

    def write(self, obj: dict) -> bool:
        if self._closed:
            return False

        line = json.dumps(obj, ensure_ascii=False)

        try:
            on_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_loop = False

        # Coalesce streamed token frames.
        if self._is_streaming_frame(obj):
            with self._token_lock:
                self._pending_tokens.append(line)
                if not self._token_flush_armed:
                    self._token_flush_armed = True
                    self._loop.call_soon_threadsafe(self._arm_token_flush)
            return not self._closed

        # Non-streaming frame: append behind buffered tokens and flush NOW.
        with self._token_lock:
            self._pending_tokens.append(line)
            batch = self._pending_tokens
            self._pending_tokens = []
            if on_loop:
                self._loop.create_task(self._safe_send_many(batch))
                return True

        # Schedule from worker thread.
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._safe_send_many(batch), self._loop
            )
        except RuntimeError:
            self._closed = True
            return False

        try:
            fut.result(timeout=_WS_WRITE_TIMEOUT_S)
            return not self._closed
        except concurrent.futures.TimeoutError:
            _log.warning(
                "ws write slow (loop stalled >%ss) peer=%s — frame left in flight",
                _WS_WRITE_TIMEOUT_S, self._peer,
            )
            return not self._closed
        except Exception as exc:
            self._closed = True
            _log.warning(
                "ws write failed peer=%s error_type=%s error=%s",
                self._peer, type(exc).__name__, exc,
            )
            return False

    def _arm_token_flush(self) -> None:
        """Arm the coalesce timer. Runs on the loop thread."""
        if self._closed:
            return
        self._token_flush_handle = self._loop.call_later(
            _TOKEN_COALESCE_S, self._flush_tokens
        )

    def _flush_tokens(self) -> None:
        """Send buffered tokens as one batch. Runs on the loop thread."""
        with self._token_lock:
            self._token_flush_handle = None
            self._token_flush_armed = False
            batch = self._pending_tokens
            self._pending_tokens = []
        if batch:
            self._loop.create_task(self._safe_send_many(batch))

    async def _safe_send_many(self, lines: list[str]) -> None:
        """Send multiple lines as one Text message (or individually)."""
        if not lines or self._closed:
            return
        try:
            if len(lines) == 1:
                await self._ws.send_text(lines[0])
            else:
                # Batch as newline-delimited.
                await self._ws.send_text("\n".join(lines))
        except Exception as exc:
            self._closed = True
            _log.debug("ws send failed peer=%s: %s", self._peer, exc)

    async def write_async(self, obj: dict) -> bool:
        """Write from the loop thread (async)."""
        if self._closed:
            return False
        line = json.dumps(obj, ensure_ascii=False)
        try:
            await self._ws.send_text(line)
            return True
        except Exception:
            self._closed = True
            return False

    def close(self) -> None:
        self._closed = True
        if self._token_flush_handle is not None:
            self._token_flush_handle.cancel()
            self._token_flush_handle = None


async def handle_ws(ws: Any) -> None:
    """Handle a WebSocket connection.

    Reads JSON-RPC requests, dispatches them via :func:`server.dispatch`,
    and writes responses/events back via :class:`WSTransport`.

    The ``gateway.ready`` event is emitted immediately after connection
    accept so the client knows the gateway is live.
    """
    loop = asyncio.get_running_loop()
    install_loop_noise_filter(loop)

    await ws.accept()
    peer = f"{ws.client.host}:{ws.client.port}" if ws.client else "unknown"
    transport = WSTransport(ws, loop, peer=peer)

    _log.info("ws connected peer=%s", peer)

    # Emit gateway.ready.
    transport.write({
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"type": "gateway.ready", "session_id": ""},
    })

    try:
        while not transport._closed:
            try:
                raw = await ws.receive_text()
            except _WebSocketDisconnect:
                break
            except Exception as exc:
                _log.debug("ws receive error peer=%s: %s", peer, exc)
                break

            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError:
                    transport.write({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "parse error"},
                    })
                    continue

                resp = server.dispatch(req, transport=transport)
                if resp is not None:
                    transport.write(resp)

    finally:
        transport.close()
        _log.info("ws disconnected peer=%s", peer)
        # Close any sessions bound to this transport.
        for sid, session in list(server._sessions.items()):
            if session.get("transport") is transport:
                server._close_session_by_id(sid, end_reason="ws_disconnect")


__all__ = ["WSTransport", "handle_ws"]
