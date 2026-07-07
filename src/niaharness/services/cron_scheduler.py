"""Background cron scheduler daemon.

The scheduler is a long-running asyncio loop that:
1. Loads jobs from :mod:`niaharness.services.cron`.
2. Determines which jobs are due (``next_run <= now`` and ``enabled=True``).
3. Executes each due job via :func:`execute_job` (subprocess).
4. Appends an entry to the history log.
5. Updates the job's ``last_run``/``last_status``/``next_run`` via
   :func:`mark_job_run`.

The scheduler is normally launched by the CLI ``cron start`` subcommand and
detached from the controlling terminal.  In tests we drive it directly via
:func:`run_scheduler_loop` with ``once=True``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from niaharness.services import cron as cron_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Secret redaction (adapted from Hermes's redact_sensitive_text)
# ---------------------------------------------------------------------------

# Patterns that look like credentials. Redacted to [REDACTED] in job output.
_SECRET_PATTERNS = [
    # API keys (common formats)
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "[REDACTED:api_key]"),
    (re.compile(r"sk-ant-[a-zA-Z0-9]{20,}"), "[REDACTED:anthropic_key]"),
    (re.compile(r"sk-or-[a-zA-Z0-9]{20,}"), "[REDACTED:openrouter_key]"),
    (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "[REDACTED:github_token]"),
    (re.compile(r"gho_[a-zA-Z0-9]{36}"), "[REDACTED:github_token]"),
    # Bearer tokens
    (re.compile(r"[Bb]earer\s+[a-zA-Z0-9._\-]{20,}"), "Bearer [REDACTED:token]"),
    # AWS keys
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED:aws_key]"),
    # Generic password assignments in env-style output
    (re.compile(r"(?i)(password|passwd|pwd|secret|token|api_key)\s*[=:]\s*\S+"), r"\1=[REDACTED]"),
]


def _redact_secrets(text: str) -> str:
    """Redact potential secrets from text before storing in history.

    Adapted from Hermes Agent's redact_sensitive_text. Covers common API
    key formats, bearer tokens, AWS keys, and password assignments.
    """
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_data_dir() -> Path:
    """Return the data dir, honoring the legacy ``OPENHARNESS_DATA_DIR`` env var."""
    legacy = os.environ.get("OPENHARNESS_DATA_DIR")
    if legacy:
        return Path(legacy)
    from niaharness.config.paths import get_data_dir as _impl

    return _impl()


def get_logs_dir() -> Path:
    """Return the logs dir, honoring the legacy ``OPENHARNESS_LOGS_DIR`` env var."""
    legacy = os.environ.get("OPENHARNESS_LOGS_DIR")
    if legacy:
        return Path(legacy)
    from niaharness.config.paths import get_logs_dir as _impl

    return _impl()


def _get_history_path() -> Path:
    return get_logs_dir() / "cron_history.json"


def _get_pidfile_path() -> Path:
    return get_data_dir() / "cron_scheduler.pid"


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def load_history(
    *,
    job_name: str | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Return history entries, optionally filtered by job name.

    Entries are returned newest-last (i.e. in the order they were appended).
    When ``limit`` > 0, returns only the most recent ``limit`` entries.
    """
    path = _get_history_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    if job_name is not None:
        data = [e for e in data if e.get("name") == job_name]
    if limit > 0:
        data = data[-limit:]
    return data


def append_history(entry: dict[str, Any]) -> None:
    """Append a single history entry to the log."""
    path = _get_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except (json.JSONDecodeError, OSError):
            existing = []
    # Always stamp with timestamp if missing.
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    existing.append(entry)
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Due-job detection
# ---------------------------------------------------------------------------


def _jobs_due(
    jobs: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    """Return the subset of ``jobs`` whose ``next_run`` has elapsed."""
    due: list[dict[str, Any]] = []
    for job in jobs:
        if not job.get("enabled", True):
            continue
        schedule = job.get("schedule", "")
        if not cron_service.validate_cron_expression(schedule):
            continue
        next_run = job.get("next_run")
        if not next_run:
            continue
        try:
            next_dt = datetime.fromisoformat(next_run)
        except ValueError:
            continue
        if next_dt.tzinfo is None:
            next_dt = next_dt.replace(tzinfo=timezone.utc)
        if next_dt <= now:
            due.append(job)
    return due


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


# Default per-job timeout (5 minutes).  Override per-job with ``timeout`` field.
_DEFAULT_JOB_TIMEOUT_SECONDS = 300


async def execute_job(job: dict[str, Any]) -> dict[str, Any]:
    """Execute a single cron job and return a history entry dict."""
    name = job.get("name", "<unnamed>")
    command = job.get("command", "")
    cwd = job.get("cwd") or None
    timeout = float(job.get("timeout", _DEFAULT_JOB_TIMEOUT_SECONDS))

    now_iso = datetime.now(timezone.utc).isoformat()
    entry: dict[str, Any] = {
        "name": name,
        "command": command,
        "started_at": now_iso,
    }

    if not command:
        entry.update(
            {
                "status": "failed",
                "returncode": -1,
                "stdout": "",
                "stderr": "empty command",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return entry

    try:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        entry.update(
            {
                "status": "failed",
                "returncode": -1,
                "stdout": "",
                "stderr": f"spawn error: {exc}",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return entry

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        # Kill the process and record timeout.
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await proc.wait()
        except Exception:
            pass
        entry.update(
            {
                "status": "timeout",
                "returncode": -1,
                "stdout": "",
                "stderr": f"timed out after {timeout}s",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return entry

    returncode = proc.returncode if proc.returncode is not None else -1
    stdout_str = stdout_b.decode("utf-8", errors="replace")
    stderr_str = stderr_b.decode("utf-8", errors="replace")

    # Redact potential secrets from output before storing in history
    # (audit fix: shell stdout/stderr was stored verbatim, leaking any
    # secrets the command happened to print).
    stdout_str = _redact_secrets(stdout_str)
    stderr_str = _redact_secrets(stderr_str)

    entry.update(
        {
            "status": "success" if returncode == 0 else "failed",
            "returncode": returncode,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return entry


# ---------------------------------------------------------------------------
# Scheduler loop
# ---------------------------------------------------------------------------


async def run_scheduler_loop(
    *,
    once: bool = False,
    poll_interval_seconds: float = 30.0,
) -> None:
    """Run the scheduler.

    When ``once=True``, performs a single scan + execute pass and returns.
    Otherwise loops forever, sleeping ``poll_interval_seconds`` between scans.
    """
    while True:
        jobs = cron_service.load_cron_jobs()
        now = datetime.now(timezone.utc)
        due = _jobs_due(jobs, now)

        for job in due:
            entry = await execute_job(job)

            # Deliver results to configured channels (email/webhook) BEFORE
            # appending to history, so the history entry includes delivery
            # status. This fixes the double-append bug where the entry was
            # appended both before and after delivery.
            delivery = job.get("delivery")
            if delivery:
                try:
                    from niaharness.services.cron_delivery import deliver_result

                    delivery_statuses = await deliver_result(
                        delivery=delivery,
                        job_name=job.get("name", ""),
                        result=entry,
                    )
                    entry["delivery"] = delivery_statuses

                    # Track delivery errors on the job record (audit fix:
                    # operators couldn't see if delivery succeeded).
                    delivery_errors = [d for d in delivery_statuses if not d.get("success")]
                    if delivery_errors:
                        entry["delivery_error"] = "; ".join(
                            d.get("error", "unknown") for d in delivery_errors
                        )
                except Exception as exc:
                    logger.warning("Delivery failed for job %s: %s", job.get("name"), exc)
                    entry["delivery_error"] = str(exc)

            # Append the complete entry (with delivery status) to history ONCE.
            append_history(entry)
            cron_service.mark_job_run(
                job.get("name", ""),
                success=entry["status"] == "success",
            )

        if once:
            return
        await asyncio.sleep(poll_interval_seconds)


# ---------------------------------------------------------------------------
# Daemon control (used by the CLI)
# ---------------------------------------------------------------------------


def is_scheduler_running() -> bool:
    """Return True if a scheduler daemon appears to be running."""
    pidfile = _get_pidfile_path()
    if not pidfile.exists():
        return False
    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def start_daemon() -> int:
    """Start the scheduler as a detached background process.

    Returns the new daemon's PID.  Raises ``RuntimeError`` if already running.
    """
    if is_scheduler_running():
        raise RuntimeError("cron scheduler is already running")

    pid = os.fork()
    if pid > 0:
        return pid

    # Child: become a daemon.
    os.setsid()
    # Second fork to fully detach.
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)

    # Write our own pid to the pidfile.
    _get_pidfile_path().parent.mkdir(parents=True, exist_ok=True)
    _get_pidfile_path().write_text(str(os.getpid()), encoding="utf-8")

    # Run the loop.  Use asyncio.run for a fresh event loop.
    try:
        asyncio.run(run_scheduler_loop())
    finally:
        try:
            _get_pidfile_path().unlink()
        except OSError:
            pass
    os._exit(0)


def stop_scheduler() -> bool:
    """Stop the running scheduler daemon.  Returns True if a process was signaled."""
    pidfile = _get_pidfile_path()
    if not pidfile.exists():
        return False
    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        pidfile.unlink()
    except OSError:
        pass
    return True


def scheduler_status() -> dict[str, Any]:
    """Return a status dict for the daemon."""
    running = is_scheduler_running()
    pidfile = _get_pidfile_path()
    pid = None
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pid = None
    return {
        "running": running,
        "pid": pid,
        "pidfile": str(pidfile),
    }
