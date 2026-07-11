"""Entry point for the NIA TUI gateway subprocess.

Ported from Hermes Agent's ``tui_gateway/entry.py`` (381 LOC).

The gateway is spawned by ``nia --tui`` as a child process. It reads
JSON-RPC requests from stdin, dispatches them via :func:`server.dispatch`,
and writes responses/events to stdout.

Signal handling: SIGPIPE is ignored (background threads writing to a
closed pipe must not kill the process). SIGTERM/SIGHUP/SIGINT are logged
to the crash log and trigger a graceful shutdown with a configurable
grace period before a hard ``os._exit(0)``.

Sidecar publisher: when ``NIA_TUI_SIDECAR_URL`` is set, every dispatcher
emit is mirrored to a dashboard WebSocket via :class:`WsPublisherTransport`.

Usage::

    python -m niaharness.tui_gateway.entry
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
import traceback

from niaharness.tui_gateway import server
from niaharness.tui_gateway.transport import TeeTransport

logger = logging.getLogger(__name__)

# Handle for the background MCP tool-discovery thread.
_mcp_discovery_thread = None


def _install_sidecar_publisher() -> None:
    """Mirror every dispatcher emit to the dashboard sidebar via WS.

    Activated by ``NIA_TUI_SIDECAR_URL``, set by the dashboard's
    ``/api/pty`` endpoint. Best-effort: connect failure or runtime drop
    falls back to stdio-only.
    """
    url = os.environ.get("NIA_TUI_SIDECAR_URL")
    if not url:
        return

    from niaharness.tui_gateway.event_publisher import WsPublisherTransport

    server._stdio_transport = TeeTransport(
        server._stdio_transport, WsPublisherTransport(url)
    )


_DEFAULT_SHUTDOWN_GRACE_S = 1.0


def _shutdown_grace_seconds() -> float:
    raw = (os.environ.get("NIA_TUI_GATEWAY_SHUTDOWN_GRACE_S") or "").strip()
    if not raw:
        return _DEFAULT_SHUTDOWN_GRACE_S
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_SHUTDOWN_GRACE_S
    return value if value > 0 else _DEFAULT_SHUTDOWN_GRACE_S


def _log_signal(signum: int, frame) -> None:
    """Capture which thread and where a termination signal hit us."""
    _signal_names: dict[int, str] = {}
    for _attr in ("SIGPIPE", "SIGTERM", "SIGHUP", "SIGINT", "SIGBREAK"):
        _sig = getattr(signal, _attr, None)
        if _sig is not None:
            _signal_names[int(_sig)] = _attr
    name = _signal_names.get(signum, f"signal {signum}")
    try:
        os.makedirs(os.path.dirname(server._CRASH_LOG), exist_ok=True)
        with open(server._CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(
                f"\n=== {name} received · {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            )
            if frame is not None:
                f.write("main-thread stack at signal delivery:\n")
                traceback.print_stack(frame, file=f)
            import threading as _threading
            for tid, th in _threading._active.items():
                f.write(f"\n--- thread {th.name} (id={tid}) ---\n")
                f.write("".join(traceback.format_stack(sys._current_frames().get(tid))))
    except Exception:
        pass
    print(f"[gateway-signal] {name}", file=sys.stderr, flush=True)

    def _hard_exit() -> None:
        os._exit(0)

    timer = threading.Timer(_shutdown_grace_seconds(), _hard_exit)
    timer.daemon = True
    timer.start()

    try:
        server._shutdown_sessions()
    except Exception:
        pass

    try:
        sys.exit(0)
    except SystemExit:
        raise


# Signal installation.
if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _log_signal)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, _log_signal)
elif hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, _log_signal)
if hasattr(signal, "SIGINT"):
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _log_exit(reason: str) -> None:
    """Record why the gateway subprocess is shutting down."""
    try:
        os.makedirs(os.path.dirname(server._CRASH_LOG), exist_ok=True)
        with open(server._CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(
                f"\n=== gateway exit · {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"· reason={reason} ===\n"
            )
    except Exception:
        pass


def main() -> int:
    """Main entry point for the gateway subprocess.

    Reads JSON-RPC requests from stdin, dispatches them, writes responses
    to stdout. Exits on stdin EOF or broken pipe.
    """
    _install_sidecar_publisher()

    # Emit gateway.ready.
    server.write_json({
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"type": "gateway.ready", "session_id": ""},
    })

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "parse error"},
            }
            if not server.write_json(resp):
                _log_exit("broken pipe on parse-error response")
                return 0
            continue

        resp = server.dispatch(req)
        if resp is not None:
            if not server.write_json(resp):
                _log_exit("broken pipe on dispatch response")
                return 0

    _log_exit("stdin EOF")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
