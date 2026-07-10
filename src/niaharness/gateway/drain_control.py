"""P1 Gateway drain control — marker-based drain request/cancel.

Ported from Hermes Agent's ``gateway/drain_control.py`` (274 LOC), scoped
to NIA's architecture.

The dashboard has no way to call into a running gateway — there is no
HTTP control channel. Restart/drain is driven only by the gateway
reacting to its own inputs: slash commands, process signals, and file
markers it writes itself.

So the begin/cancel-drain dashboard endpoint communicates with the
running gateway the same way: it writes (or removes) a marker file, and
a gateway background watcher reacts to it.

Contract (presence-based):
  - begin-drain → write ``{NIA_HOME}/.drain_request.json``
  - cancel-drain → remove the marker
  - The gateway watcher treats presence of a marker stamped with the
    current instantiation epoch as "external drain active": flip state
    to "draining" and stop accepting new turns.

Why the epoch: NIA_HOME is a durable store. A begin-drain marker
written there survives a machine restart. But the disruptive lifecycle
actions a drain protects (auto-update / image migrate / env edit /
profile change) all restart the machine, which is exactly the signal
that the drain is over. Without the epoch, a freshly-restarted gateway
re-reads the orphaned marker on boot and parks itself in "draining"
forever. Stamping the marker with an identity of *this* container/VM
instantiation, and ignoring a marker whose epoch doesn't match, makes
"a deliberate restart clears the drain" true by construction.
"""

from __future__ import annotations

import functools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DRAIN_REQUEST_FILENAME = ".drain_request.json"


def _get_nia_home() -> Path:
    """Return NIA_HOME (default ~/.nia)."""
    try:
        from niaharness.config.paths import get_nia_home
        return Path(get_nia_home())
    except Exception:
        import os
        return Path(os.environ.get("NIA_HOME", os.path.expanduser("~/.nia")))


@functools.lru_cache(maxsize=1)
def current_instantiation_epoch() -> str:
    """Identity of THIS container / VM instantiation.

    Stable for the life of the PID-1 init process — so an s6 respawn of
    just the gateway keeps the same epoch and an in-flight drain is
    honoured — but changes when the machine/container is recreated.

    Composed from:
      - the kernel boot id (``/proc/sys/kernel/random/boot_id``) —
        changes on a VM/microVM reboot.
      - PID 1's start time (field 22 of /proc/1/stat) — changes on a
        plain docker restart.

    Returns "" when neither is readable (non-Linux, no /proc). An empty
    epoch disables the staleness check downstream, degrading to the
    released presence-only behaviour — never fail-closed.
    """
    boot_id = ""
    try:
        boot_id = (
            Path("/proc/sys/kernel/random/boot_id")
            .read_text(encoding="utf-8")
            .strip()
        )
    except OSError:
        pass

    pid1_start = ""
    try:
        stat = Path("/proc/1/stat").read_text(encoding="utf-8")
        tail = stat.rsplit(")", 1)[1].split()
        pid1_start = tail[19]
    except (OSError, IndexError):
        pass

    if not boot_id and not pid1_start:
        return ""
    return f"{boot_id}:{pid1_start}"


def drain_request_path(home: Optional[Path] = None) -> Path:
    """Absolute path to the drain-request marker."""
    base = home if home is not None else _get_nia_home()
    return Path(base) / _DRAIN_REQUEST_FILENAME


def _atomic_write(path: Path, data: str) -> None:
    """Atomic write: write to .tmp then rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def write_drain_request(
    *,
    principal: str = "drain-control",
    suppress_notification: bool = False,
    home: Optional[Path] = None,
) -> dict[str, Any]:
    """Write the begin-drain marker. Returns the payload written.

    Atomic write so the gateway watcher never reads a half-written file.
    Idempotent: re-writing while a drain is already in progress just
    refreshes requested_at.
    """
    payload = {
        "action": "drain",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "principal": principal,
        "epoch": current_instantiation_epoch(),
        "suppress_notification": bool(suppress_notification),
    }
    path = drain_request_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _atomic_write(path, json.dumps(payload, indent=2))
    except OSError as exc:
        logger.warning("drain-control: failed to write %s: %s", path, exc)
    return payload


def clear_drain_request(*, home: Optional[Path] = None) -> bool:
    """Remove the drain marker (cancel-drain). Returns True if one existed.

    Best-effort: a missing file is not an error (cancel is idempotent).
    """
    path = drain_request_path(home)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("drain-control: failed to remove %s: %s", path, exc)
        return False


def read_drain_request(*, home: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Return the marker payload, or None if absent.

    A present-but-unparseable marker returns ``{}`` (truthy-presence
    preserved). Never raises.
    """
    path = drain_request_path(home)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("drain-control: failed to read %s: %s", path, exc)
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _marker_epoch_is_stale(body: dict[str, Any]) -> bool:
    """True iff body's epoch is a definite mismatch with this process.

    Lenient by design — returns False (i.e. "not stale, honour it")
    whenever it can't be sure:
      - the current epoch can't be computed ("" fallback, no /proc), OR
      - the marker carries no epoch (legacy/corrupt marker).
    Only a marker whose epoch is present AND differs from the current
    instantiation epoch is considered stale.
    """
    current = current_instantiation_epoch()
    if not current:
        return False
    marker_epoch = body.get("epoch")
    if not marker_epoch:
        return False
    return marker_epoch != current


def drain_requested(*, home: Optional[Path] = None) -> bool:
    """True iff a begin-drain marker for THIS instantiation is present.

    A marker whose epoch does not match the current instantiation epoch
    is treated as absent: it survived a container/VM restart and the
    lifecycle action that triggered the drain has already completed.
    """
    body = read_drain_request(home=home)
    if body is None:
        return False
    if _marker_epoch_is_stale(body):
        return False
    return True


def drain_notification_suppressed(*, home: Optional[Path] = None) -> bool:
    """True iff an ACTIVE drain marker asks to suppress the shutdown broadcast."""
    body = read_drain_request(home=home)
    if body is None:
        return False
    if _marker_epoch_is_stale(body):
        return False
    return bool(body.get("suppress_notification"))


__all__ = [
    "clear_drain_request",
    "current_instantiation_epoch",
    "drain_notification_suppressed",
    "drain_request_path",
    "drain_requested",
    "read_drain_request",
    "write_drain_request",
]
