"""Cron job registry — pure storage layer, no scheduling.

Jobs are stored as a JSON list at ``<data_dir>/cron_jobs.json``.  Each job is
a dict with the following shape::

    {
        "name": "nightly-backup",
        "schedule": "0 2 * * *",          # 5-field cron expression
        "command": "echo hi",              # shell command to run
        "cwd": "/path/to/repo",            # optional
        "enabled": true,
        "next_run": "2026-01-01T02:00:00+00:00",  # ISO 8601
        "last_run": null,
        "last_status": null,               # "success" | "failed" | "timeout"
        "created_at": "2026-01-01T00:00:00+00:00",
    }

Tests monkeypatch ``niaharness.services.cron.get_cron_registry_path`` to a
temp path; the public API must respect that override.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_cron_registry_path() -> Path:
    """Return the cron registry file path.

    Honors the legacy ``OPENHARNESS_DATA_DIR`` env var for backward compat
    with tests and scripts that still set it.
    """
    legacy = os.environ.get("OPENHARNESS_DATA_DIR")
    if legacy:
        return Path(legacy) / "cron_jobs.json"
    from niaharness.config.paths import get_cron_registry_path as _impl

    return _impl()


# ---------------------------------------------------------------------------
# Cron expression validation
# ---------------------------------------------------------------------------

# Standard 5-field cron bounds.
_FIELD_BOUNDS = [
    (0, 59),   # minute
    (0, 23),   # hour
    (1, 31),   # day of month
    (1, 12),   # month
    (0, 7),    # day of week (0 and 7 both = Sunday)
]


def _validate_field(field: str, low: int, high: int) -> bool:
    """Return True if a single cron field is syntactically valid."""
    if not field:
        return False
    for part in field.split(","):
        # Handle step values: "a/b"
        if "/" in part:
            base, _, step = part.partition("/")
            if not step.isdigit() or int(step) == 0:
                return False
        else:
            base = part
        # Handle ranges: "a-b"
        if base == "*":
            continue
        if "-" in base:
            a, _, b = base.partition("-")
            if not (a.isdigit() and b.isdigit()):
                return False
            ai, bi = int(a), int(b)
            if not (low <= ai <= high and low <= bi <= high):
                return False
            if ai > bi:
                return False
        elif base.isdigit():
            v = int(base)
            if not (low <= v <= high):
                return False
        else:
            return False
    return True


def validate_cron_expression(expr: str) -> bool:
    """Return True if ``expr`` is a valid 5-field cron expression."""
    if not isinstance(expr, str) or not expr.strip():
        return False
    fields = expr.split()
    if len(fields) != 5:
        return False
    for field, (low, high) in zip(fields, _FIELD_BOUNDS):
        if not _validate_field(field, low, high):
            return False
    return True


# ---------------------------------------------------------------------------
# Schedule computation
# ---------------------------------------------------------------------------


def next_run_time(expr: str, base: datetime) -> datetime | None:
    """Return the next time ``expr`` should fire after ``base``.

    Brute-force minute-by-minute scan up to 366 days.  Returns ``None`` if no
    match is found in that window (e.g. impossible expression like "0 0 30 2 *").
    """
    if not validate_cron_expression(expr):
        return None
    fields = expr.split()
    minute_f, hour_f, dom_f, month_f, dow_f = fields

    # Round base up to the next minute.
    candidate = base.replace(second=0, microsecond=0)
    # Always advance at least one minute so we don't re-fire immediately.
    candidate = candidate.replace(minute=candidate.minute)  # truncate
    # Add one minute to start scanning the next slot
    from datetime import timedelta

    candidate = candidate + timedelta(minutes=1)

    def field_matches(field: str, value: int, low: int, high: int) -> bool:
        if field == "*":
            return True
        for part in field.split(","):
            step = 1
            if "/" in part:
                base_part, _, step_s = part.partition("/")
                step = int(step_s)
            else:
                base_part = part
            if base_part == "*":
                if (value - low) % step == 0:
                    return True
                continue
            if "-" in base_part:
                a, _, b = base_part.partition("-")
                ai, bi = int(a), int(b)
                if ai <= value <= bi and (value - ai) % step == 0:
                    return True
            elif base_part.isdigit():
                v = int(base_part)
                if v == value and step == 1:
                    return True
        return False

    for _ in range(366 * 24 * 60):  # up to one year
        if (
            field_matches(minute_f, candidate.minute, 0, 59)
            and field_matches(hour_f, candidate.hour, 0, 23)
            and field_matches(dom_f, candidate.day, 1, 31)
            and field_matches(month_f, candidate.month, 1, 12)
            and field_matches(dow_f, candidate.weekday(), 0, 6)
        ):
            return candidate
        candidate = candidate + timedelta(minutes=1)
    return None


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def load_cron_jobs() -> list[dict[str, Any]]:
    """Return all cron jobs sorted by name.  Returns ``[]`` on missing/corrupt file."""
    path = get_cron_registry_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return sorted(data, key=lambda j: j.get("name", ""))


def save_cron_jobs(jobs: list[dict[str, Any]]) -> None:
    """Persist the job list to disk."""
    path = get_cron_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jobs, indent=2, default=str), encoding="utf-8")


def upsert_cron_job(job: dict[str, Any]) -> dict[str, Any]:
    """Insert or update a job by name.

    On insert, the job is enriched with ``enabled=True``, ``next_run``,
    ``created_at``, and ``last_run=None``.  On update, mutable fields are
    overwritten but ``enabled``, ``last_run``, ``last_status`` are preserved
    unless explicitly set in ``job``.
    """
    name = job.get("name")
    if not name:
        raise ValueError("cron job requires a 'name' field")
    schedule = job.get("schedule")
    if not validate_cron_expression(schedule or ""):
        raise ValueError(f"invalid cron expression: {schedule!r}")

    jobs = load_cron_jobs()
    existing_idx = next((i for i, j in enumerate(jobs) if j.get("name") == name), None)

    now_iso = datetime.now(timezone.utc).isoformat()
    next_iso = next_run_time(schedule, datetime.now(timezone.utc))
    next_iso_s = next_iso.isoformat() if next_iso else None

    if existing_idx is None:
        new_job = {
            "name": name,
            "schedule": schedule,
            "command": job.get("command", ""),
            "cwd": job.get("cwd"),
            "enabled": job.get("enabled", True),
            "next_run": next_iso_s,
            "last_run": None,
            "last_status": None,
            "created_at": now_iso,
        }
        # Merge any extra fields the caller supplied.
        for k, v in job.items():
            if k not in new_job:
                new_job[k] = v
        jobs.append(new_job)
    else:
        existing = jobs[existing_idx]
        existing.update(job)
        existing["schedule"] = schedule
        # Recompute next_run when schedule changes.
        existing["next_run"] = next_iso_s
        # Ensure required keys exist.
        existing.setdefault("enabled", True)
        existing.setdefault("last_run", None)
        existing.setdefault("last_status", None)
        existing.setdefault("created_at", now_iso)
        jobs[existing_idx] = existing

    save_cron_jobs(jobs)
    return next((j for j in jobs if j.get("name") == name), job)


def delete_cron_job(name: str) -> bool:
    """Delete the job with the given name.  Returns True if a job was deleted."""
    jobs = load_cron_jobs()
    new_jobs = [j for j in jobs if j.get("name") != name]
    if len(new_jobs) == len(jobs):
        return False
    save_cron_jobs(new_jobs)
    return True


def get_cron_job(name: str) -> dict[str, Any] | None:
    """Return the job with the given name, or ``None``."""
    for job in load_cron_jobs():
        if job.get("name") == name:
            return job
    return None


def set_job_enabled(name: str, enabled: bool) -> bool:
    """Toggle a job's ``enabled`` flag.  Returns True if the job exists."""
    jobs = load_cron_jobs()
    for job in jobs:
        if job.get("name") == name:
            job["enabled"] = bool(enabled)
            save_cron_jobs(jobs)
            return True
    return False


def mark_job_run(name: str, *, success: bool) -> None:
    """Update ``last_run`` and ``last_status`` for a job.  No-op if missing."""
    jobs = load_cron_jobs()
    now_iso = datetime.now(timezone.utc).isoformat()
    found = False
    for job in jobs:
        if job.get("name") == name:
            job["last_run"] = now_iso
            job["last_status"] = "success" if success else "failed"
            # Schedule the next run.
            schedule = job.get("schedule", "")
            if validate_cron_expression(schedule):
                nxt = next_run_time(schedule, datetime.now(timezone.utc))
                if nxt is not None:
                    job["next_run"] = nxt.isoformat()
            found = True
            break
    if found:
        save_cron_jobs(jobs)
