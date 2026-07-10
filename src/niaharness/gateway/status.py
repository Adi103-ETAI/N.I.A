"""P1 Gateway runtime status — PID-file lock, runtime status, --replace takeover.

Ported from Hermes Agent's ``gateway/status.py`` (1441 LOC), scoped to NIA's
architecture. Provides:

  - PID-file based detection of whether the gateway daemon is running.
  - Runtime status file (gateway_state.json) with active agent count,
    gateway state (running/draining/stopped), and health metadata.
  - ``--replace`` takeover: terminate an existing gateway instance and
    acquire the runtime lock atomically.
  - Signal handlers for clean shutdown (SIGINT/SIGTERM).
  - PID-reuse guard via process start-time fingerprinting.

The PID file lives at ``{NIA_HOME}/gateway.pid``. NIA_HOME defaults to
``~/.nia`` but can be overridden via the environment variable. Separate
NIA_HOME directories get separate PID files — useful for named profiles.

Usage::

    from niaharness.gateway.status import (
        write_pid_file,
        remove_pid_file,
        get_running_pid,
        is_gateway_running,
        acquire_gateway_runtime_lock,
        release_gateway_runtime_lock,
    )

    # On startup:
    if not acquire_gateway_runtime_lock():
        existing = get_running_pid()
        print(f"Gateway already running (PID {existing})")
        sys.exit(1)
    write_pid_file()

    # On shutdown:
    remove_pid_file()
    release_gateway_runtime_lock()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"
_GATEWAY_LOCK_FILENAME = "gateway.lock"
_RUNTIME_STATUS_FILE = "gateway_state.json"

# Module-level lock handle (held for the gateway's lifetime).
_gateway_lock_handle: Optional[Any] = None


def _get_nia_home() -> Path:
    """Return NIA_HOME (default ~/.nia)."""
    try:
        from niaharness.config.paths import get_nia_home
        return Path(get_nia_home())
    except Exception:
        return Path(os.environ.get("NIA_HOME", os.path.expanduser("~/.nia")))


def _get_pid_path() -> Path:
    """Return the path to the gateway PID file."""
    return _get_nia_home() / "gateway.pid"


def _get_gateway_lock_path() -> Path:
    """Return the path to the runtime gateway lock file."""
    return _get_nia_home() / _GATEWAY_LOCK_FILENAME


def _get_runtime_status_path() -> Path:
    """Return the persisted runtime health/status file path."""
    return _get_nia_home() / _RUNTIME_STATUS_FILE


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Process inspection helpers
# ---------------------------------------------------------------------------


def _pid_exists(pid: int) -> bool:
    """Return True if a process with the given PID exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        # ProcessLookupError = no such process.
        # PermissionError = process exists but we can't signal it.
        return not isinstance(Exception() if False else None, ProcessLookupError)
    except OSError:
        return False


def _get_process_start_time(pid: int) -> Optional[int]:
    """Return a stable per-process start-time fingerprint, or None.

    Used as a PID-reuse guard: a (pid, start_time) pair uniquely identifies
    a process. On Linux, reads /proc/<pid>/stat field 22. On macOS/Windows,
    falls back to psutil.
    """
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        return int(stat_path.read_text(encoding="utf-8").split()[21])
    except (FileNotFoundError, IndexError, PermissionError, ValueError, OSError):
        pass
    try:
        import psutil  # type: ignore
        return int(round(psutil.Process(pid).create_time() * 100))
    except Exception:
        return None


def _read_process_cmdline(pid: int) -> Optional[str]:
    """Return the process command line as a space-separated string."""
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = cmdline_path.read_bytes()
        if raw:
            return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
    except (FileNotFoundError, PermissionError, OSError):
        pass
    if not _IS_WINDOWS:
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


def looks_like_gateway_process(pid: int) -> bool:
    """Heuristic: does this PID look like a NIA gateway process?

    Checks the command line for 'nia' + 'gateway' substrings.
    """
    cmdline = _read_process_cmdline(pid)
    if not cmdline:
        return False
    cmdline_lower = cmdline.lower()
    return "nia" in cmdline_lower and "gateway" in cmdline_lower


# ---------------------------------------------------------------------------
# PID file management
# ---------------------------------------------------------------------------


def _build_pid_record() -> dict[str, Any]:
    """Build the PID record dict."""
    pid = os.getpid()
    return {
        "pid": pid,
        "start_time": _get_process_start_time(pid),
        "cmdline": _read_process_cmdline(pid),
        "started_at": _utc_now_iso(),
        "kind": "nia-gateway",
    }


def write_pid_file() -> None:
    """Write the PID file for this gateway process."""
    path = _get_pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _build_pid_record()
    try:
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.debug("Wrote PID file: %s (pid=%d)", path, record["pid"])
    except OSError as exc:
        logger.warning("Failed to write PID file %s: %s", path, exc)


def remove_pid_file() -> None:
    """Remove the PID file (only if it belongs to this process)."""
    path = _get_pid_path()
    try:
        raw = path.read_text(encoding="utf-8")
        record = json.loads(raw)
        if record.get("pid") == os.getpid():
            path.unlink()
            logger.debug("Removed PID file: %s", path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass


def _read_pid_record() -> Optional[dict[str, Any]]:
    """Read the PID record from the PID file, or None."""
    path = _get_pid_path()
    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def get_running_pid() -> Optional[int]:
    """Return the PID of the running gateway, or None.

    Validates that the PID is alive + looks like a gateway process.
    Stale PID files are cleaned up.
    """
    record = _read_pid_record()
    if record is None:
        return None
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    if not _pid_exists(pid):
        # Stale PID file — clean it up.
        _cleanup_stale_pid_file()
        return None
    # PID-reuse guard: check start_time if we have it.
    recorded_start = record.get("start_time")
    if recorded_start is not None:
        live_start = _get_process_start_time(pid)
        if live_start is not None and live_start != recorded_start:
            # PID was recycled — stale file.
            _cleanup_stale_pid_file()
            return None
    return pid


def _cleanup_stale_pid_file() -> None:
    """Remove a stale PID file (best-effort)."""
    path = _get_pid_path()
    try:
        path.unlink()
    except (FileNotFoundError, OSError):
        pass


def is_gateway_running() -> bool:
    """Return True if the gateway daemon is running."""
    return get_running_pid() is not None


# ---------------------------------------------------------------------------
# Runtime lock (file-based mutex)
# ---------------------------------------------------------------------------


def acquire_gateway_runtime_lock() -> bool:
    """Acquire the runtime lock. Returns True if acquired, False if held by another.

    On POSIX, uses fcntl.flock (exclusive). On Windows, uses msvcrt.locking.
    The lock is held for the gateway's lifetime and released on process exit.
    """
    global _gateway_lock_handle
    if _gateway_lock_handle is not None:
        return True  # Already held by this process.

    lock_path = _get_gateway_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _gateway_lock_handle = open(lock_path, "w")
    except OSError as exc:
        logger.warning("Failed to open gateway lock file %s: %s", lock_path, exc)
        return False

    if _IS_WINDOWS:
        try:
            import msvcrt
            msvcrt.locking(_gateway_lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except (OSError, IOError):
            _gateway_lock_handle.close()
            _gateway_lock_handle = None
            return False
    else:
        import fcntl
        try:
            fcntl.flock(_gateway_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, IOError):
            _gateway_lock_handle.close()
            _gateway_lock_handle = None
            return False


def release_gateway_runtime_lock() -> None:
    """Release the runtime lock."""
    global _gateway_lock_handle
    if _gateway_lock_handle is None:
        return
    try:
        if _IS_WINDOWS:
            import msvcrt
            msvcrt.locking(_gateway_lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(_gateway_lock_handle.fileno(), fcntl.LOCK_UN)
    except (OSError, IOError):
        pass
    try:
        _gateway_lock_handle.close()
    except OSError:
        pass
    _gateway_lock_handle = None


def is_gateway_runtime_lock_active() -> bool:
    """Return True if the runtime lock is held (by any process)."""
    lock_path = _get_gateway_lock_path()
    if not lock_path.exists():
        return False
    # Try to acquire non-exclusively — if it fails, someone holds it.
    try:
        fd = os.open(lock_path, os.O_RDONLY)
    except OSError:
        return False
    try:
        if _IS_WINDOWS:
            return False  # Can't easily test on Windows.
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False  # Lock was free → not active.
        except (OSError, IOError):
            return True  # Lock is held by another process.
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Runtime status (gateway_state.json)
# ---------------------------------------------------------------------------


def write_runtime_status(
    *,
    state: str = "running",
    active_agents: int = 0,
    uptime_seconds: float = 0.0,
    adapters: Optional[list[str]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Write the runtime status file.

    Args:
        state: Gateway state — "running", "draining", "stopped".
        active_agents: Number of in-flight agent turns.
        uptime_seconds: Seconds since the gateway started.
        adapters: List of connected adapter platform names.
        extra: Additional metadata.
    """
    path = _get_runtime_status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "pid": os.getpid(),
        "state": state,
        "active_agents": active_agents,
        "uptime_seconds": uptime_seconds,
        "adapters": adapters or [],
        "updated_at": _utc_now_iso(),
    }
    if extra:
        payload["extra"] = extra
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("Failed to write runtime status: %s", exc)


def read_runtime_status() -> Optional[dict[str, Any]]:
    """Read the runtime status file, or None."""
    path = _get_runtime_status_path()
    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def derive_gateway_busy(status: Optional[dict[str, Any]]) -> bool:
    """Return True if the gateway has in-flight agent work."""
    if not status:
        return False
    return int(status.get("active_agents", 0)) > 0


def derive_gateway_drainable(
    *, gateway_running: bool, gateway_state: Any
) -> bool:
    """Return True if the gateway can be drained (running + not already draining)."""
    if not gateway_running:
        return False
    state = str(gateway_state or "running").lower()
    return state not in ("draining", "stopped")


# ---------------------------------------------------------------------------
# --replace takeover
# ---------------------------------------------------------------------------


def terminate_pid(pid: int, *, force: bool = False) -> None:
    """Terminate a PID with platform-appropriate force semantics."""
    if force and _IS_WINDOWS:
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                raise OSError(
                    (result.stderr or result.stdout or "").strip()
                    or f"taskkill failed for PID {pid}"
                )
        except FileNotFoundError:
            os.kill(pid, signal.SIGTERM)
        return
    sig = signal.SIGTERM if not force else getattr(signal, "SIGKILL", signal.SIGTERM)
    os.kill(pid, sig)


def replace_existing_gateway(timeout: float = 10.0) -> bool:
    """Terminate an existing gateway instance and take over.

    Returns True if the existing instance was terminated (or none was
    running). Returns False if the existing instance didn't terminate
    within the timeout.
    """
    existing_pid = get_running_pid()
    if existing_pid is None:
        return True  # Nothing to replace.

    logger.info("Replacing existing gateway instance (PID %d)", existing_pid)
    try:
        terminate_pid(existing_pid, force=False)
    except (ProcessLookupError, PermissionError) as exc:
        logger.warning("Could not terminate PID %d: %s", existing_pid, exc)
        return False

    # Wait for the process to exit.
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_exists(existing_pid):
            break
        time.sleep(0.2)

    if _pid_exists(existing_pid):
        logger.warning(
            "Existing gateway (PID %d) did not terminate within %.1fs",
            existing_pid, timeout,
        )
        return False

    # Clean up stale PID file + lock.
    _cleanup_stale_pid_file()
    return True


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


class GatewaySignalHandler:
    """Context manager that installs SIGINT/SIGTERM handlers for clean shutdown.

    On signal, sets a threading.Event so the main loop can drain and exit.
    Usage::

        with GatewaySignalHandler() as sig_handler:
            while not sig_handler.shutdown_requested:
                await run_gateway_loop()
    """

    def __init__(self) -> None:
        self._shutdown_event: Optional[Any] = None
        self._old_handlers: dict[int, Any] = {}

    @property
    def shutdown_requested(self) -> bool:
        if self._shutdown_event is None:
            return False
        return self._shutdown_event.is_set()

    def __enter__(self) -> "GatewaySignalHandler":
        import threading
        self._shutdown_event = threading.Event()

        def _handler(signum: int, frame: Any) -> None:
            logger.info("Received signal %d — requesting shutdown", signum)
            if self._shutdown_event is not None:
                self._shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._old_handlers[sig] = signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass  # Not in main thread.
        return self

    def __exit__(self, *args: Any) -> None:
        for sig, old_handler in self._old_handlers.items():
            try:
                signal.signal(sig, old_handler)
            except (ValueError, OSError):
                pass
        self._old_handlers.clear()
        self._shutdown_event = None


__all__ = [
    "GatewaySignalHandler",
    "acquire_gateway_runtime_lock",
    "derive_gateway_busy",
    "derive_gateway_drainable",
    "get_running_pid",
    "is_gateway_running",
    "is_gateway_runtime_lock_active",
    "looks_like_gateway_process",
    "read_runtime_status",
    "release_gateway_runtime_lock",
    "remove_pid_file",
    "replace_existing_gateway",
    "terminate_pid",
    "write_pid_file",
    "write_runtime_status",
]
