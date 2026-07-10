"""P1 Gateway restart-loop breaker.

Ported from Hermes Agent's ``gateway/restart_loop_guard.py`` (151 LOC).

When the gateway auto-resumes a restart-interrupted session on boot,
and that session's next turn re-runs the offending logic (e.g. a
terminal command that SIGTERMs the gateway), the result is a tight
respawn loop: boot → auto-resume → SIGTERM → boot → ...

This module is the last-resort circuit breaker: it records a timestamp
each time the gateway boots with restart-interrupted sessions pending,
keeps a rolling window of recent boots persisted across processes, and
reports the loop as "tripped" once too many such boots happen inside a
short window. When tripped, the caller SKIPS auto-resume for that boot
— the gateway still starts and serves real inbound messages, it just
stops replaying the session that keeps killing it.

State lives in ``<NIA_HOME>/gateway/restart_loop.json`` so it is
profile-scoped and survives process death. Best-effort: any read/write
failure fails OPEN (no false trip) because a broken breaker must never
wedge a healthy gateway.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Defaults: a legitimate operator restart (or two) never trips the
# breaker, but a ~10s respawn loop does within a few cycles.
DEFAULT_MAX_RESTARTS = 3
DEFAULT_WINDOW_SECONDS = 60


def _get_nia_home() -> Path:
    """Return NIA_HOME (default ~/.nia)."""
    try:
        from niaharness.config.paths import get_nia_home
        return Path(get_nia_home())
    except Exception:
        import os
        return Path(os.environ.get("NIA_HOME", os.path.expanduser("~/.nia")))


def _state_path() -> Path:
    """Return the path to the restart-loop state file."""
    return _get_nia_home() / "gateway" / "restart_loop.json"


def _load_boots() -> List[float]:
    """Load the list of recent restart-interrupted boot timestamps."""
    try:
        raw = _state_path().read_text(encoding="utf-8")
        data = json.loads(raw)
        boots = data.get("boots", [])
        return [float(t) for t in boots if isinstance(t, (int, float))]
    except (OSError, ValueError, TypeError):
        return []


def _save_boots(boots: List[float]) -> None:
    """Persist the boot log (best-effort)."""
    try:
        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"boots": boots}), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not save restart-loop state: %s", exc)


def record_restart_interrupted_boot(
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    *,
    now: Optional[float] = None,
) -> List[float]:
    """Record that the gateway just booted with restart-interrupted sessions.

    Prunes boots older than *window_seconds* and appends the current time.
    Returns the pruned+appended list (most recent last). Best-effort — a
    persistence failure returns the in-memory list without raising.
    """
    ts = time.time() if now is None else now
    cutoff = ts - max(1, window_seconds)
    boots = [t for t in _load_boots() if t >= cutoff]
    boots.append(ts)
    _save_boots(boots)
    return boots


def is_restart_loop_tripped(
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    *,
    now: Optional[float] = None,
) -> bool:
    """Return True if the gateway has restarted >= max_restarts times with
    restart-interrupted sessions inside the last window_seconds.

    Fails OPEN (returns False) on any error — a broken breaker must never
    wedge a healthy gateway.
    """
    if max_restarts <= 0:
        return False
    ts = time.time() if now is None else now
    cutoff = ts - max(1, window_seconds)
    try:
        recent = [t for t in _load_boots() if t >= cutoff]
    except Exception:
        return False
    return len(recent) >= max_restarts


def clear() -> None:
    """Remove the persisted boot log (used on clean shutdown / by tests)."""
    try:
        _state_path().unlink(missing_ok=True)
    except OSError:
        pass


def check_and_record(
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    *,
    now: Optional[float] = None,
) -> bool:
    """Record this restart-interrupted boot and report whether the loop is
    now tripped.

    This is the single entry point the gateway calls: it appends the
    current boot, then checks whether the (now-updated) window has
    reached the threshold. Returns True when auto-resume should be
    SKIPPED to break the loop.
    """
    boots = record_restart_interrupted_boot(window_seconds, now=now)
    tripped = len(boots) >= max_restarts if max_restarts > 0 else False
    if tripped:
        logger.warning(
            "Restart-loop breaker TRIPPED: %d restart-interrupted gateway "
            "boots within %ds (threshold %d). Skipping auto-resume to break "
            "a suspected SIGTERM-respawn loop. Restart-interrupted sessions "
            "stay resume-pending and will continue on the next real user "
            "message. If this is a false positive, delete %s.",
            len(boots),
            window_seconds,
            max_restarts,
            _state_path(),
        )
    return tripped


__all__ = [
    "DEFAULT_MAX_RESTARTS",
    "DEFAULT_WINDOW_SECONDS",
    "check_and_record",
    "clear",
    "is_restart_loop_tripped",
    "record_restart_interrupted_boot",
]
