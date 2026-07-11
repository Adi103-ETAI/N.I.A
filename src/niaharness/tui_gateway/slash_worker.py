"""Persistent slash-command worker — one NIA CLI per TUI session.

Ported from Hermes Agent's ``tui_gateway/slash_worker.py`` (157 LOC).

Protocol: reads JSON lines from stdin ``{id, command}``, writes
``{id, ok, output|error}`` to stdout.

The worker is a persistent subprocess spawned by the gateway. It keeps a
NIA command registry loaded so slash commands execute without the ~200ms
startup cost of building a fresh runtime each time. A parent-death
watchdog thread detects when the spawning gateway process is gone and
exits cleanly so orphaned workers don't linger.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import threading
import time

logger = logging.getLogger(__name__)

# Env-overridable so the integration test can drive sub-second timing.
def _env_float(name: str, default: float) -> float:
    """Parse a float env knob, falling back to ``default`` on absent/malformed."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


_WATCHDOG_POLL_S = max(0.05, _env_float("NIA_SLASH_WATCHDOG_POLL_S", 2.0))
_ORPHAN_GRACE_S = max(0.0, _env_float("NIA_SLASH_WATCHDOG_GRACE_S", 5.0))
_in_flight = threading.Event()  # set while a command is executing


def _is_orphaned(original_ppid, parent_create_time, getppid=os.getppid) -> bool:
    """True once our spawning gateway is gone.

    Compare to the ORIGINAL ppid (never ==1: Linux reparents to a subreaper)
    and guard PID reuse via create_time.
    """
    if getppid() != original_ppid:
        return True
    try:
        import psutil
        if not psutil.pid_exists(original_ppid):
            return True
        return psutil.Process(original_ppid).create_time() != parent_create_time
    except (ImportError, Exception):
        # psutil not available — fall back to ppid check only.
        return getppid() != original_ppid and getppid() == 1


def _start_parent_death_watchdog(original_ppid, parent_create_time) -> None:
    def _loop():
        while not _is_orphaned(original_ppid, parent_create_time):
            time.sleep(_WATCHDOG_POLL_S)
        deadline = time.monotonic() + _ORPHAN_GRACE_S
        while _in_flight.is_set() and time.monotonic() < deadline:
            time.sleep(0.05)  # let an in-flight command finish/flush
        os._exit(0)

    threading.Thread(target=_loop, daemon=True).start()


def _run(command_registry, command: str) -> str:
    """Execute a single slash command and return its output."""
    cmd = (command or "").strip()
    if not cmd:
        return ""
    if not cmd.startswith("/"):
        cmd = f"/{cmd}"

    buf = io.StringIO()

    try:
        from rich.console import Console
        console = Console(file=buf, force_terminal=True, width=120)
    except ImportError:
        console = None

    try:
        # Try to execute the command via the registry.
        from niaharness.commands import CommandContext, CommandResult
        ctx = CommandContext(
            cwd=os.getcwd(),
            console=console,
            args=cmd[1:].split(),  # strip leading /
        )
        result = command_registry.execute(cmd[1:], ctx)
        if result and result.output:
            buf.write(result.output)
    except Exception as exc:
        buf.write(f"Error: {exc}")

    # Strip ANSI codes for plain-text output.
    try:
        from niaharness.permissions.shell_hardening import strip_ansi
        return strip_ansi(buf.getvalue().rstrip())
    except ImportError:
        import re
        ansi_re = re.compile(r"\x1b\[[0-9;]*m")
        return ansi_re.sub("", buf.getvalue().rstrip())


def main():
    """Entry point for the slash worker subprocess.

    Usage: ``python -m niaharness.tui_gateway.slash_worker --session-key <key>``
    """
    import argparse

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--session-key", required=True)
    p.add_argument("--model", default="")
    args = p.parse_args()

    os.environ["NIA_SESSION_KEY"] = args.session_key
    os.environ["NIA_INTERACTIVE"] = "1"

    # Start the parent-death watchdog before building the command registry
    # — that window is itself an orphan risk if the gateway dies mid-spawn.
    orig_ppid = os.getppid()
    try:
        import psutil
        parent_create_time = psutil.Process(orig_ppid).create_time()
    except (ImportError, Exception):
        parent_create_time = 0.0
    _start_parent_death_watchdog(orig_ppid, parent_create_time)

    # Build the command registry (suppress output during build).
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            from niaharness.commands import create_default_command_registry
            command_registry = create_default_command_registry()
        except Exception:
            command_registry = None

    # Main loop: read JSON lines, execute, write JSON responses.
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        _in_flight.set()
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            if command_registry is not None:
                out = _run(command_registry, req.get("command", ""))
            else:
                out = "Error: command registry not available"
            sys.stdout.write(json.dumps({"id": rid, "ok": True, "output": out}) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"id": rid, "ok": False, "error": str(e)}) + "\n")
            sys.stdout.flush()
        finally:
            _in_flight.clear()


if __name__ == "__main__":
    main()


__all__ = ["main"]
