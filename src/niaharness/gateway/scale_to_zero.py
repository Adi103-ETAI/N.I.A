"""P1 Gateway scale-to-zero idle detection.

Ported from Hermes Agent's ``gateway/scale_to_zero.py`` (125 LOC).

This is the gateway-side BEHAVIOUR layer for scale-to-zero: it owns the
*decision* to go idle. It does NOT itself suspend the machine — on
cloud hosts, the now-traffic-idle machine is suspended by the host's
autostop mechanism and woken by a wakeUrl poke.

Design constraints:
  - Per-instance enable is gated by the ``NIA_SCALE_TO_ZERO`` env flag.
  - Arm only when messaging is relay-only or absent AND a wakeUrl is
    registered AND the flag is set.
  - Idle = no in-flight agent turn AND no inbound for N min AND no live
    background work.
  - The pure helpers take plain inputs so they unit-test without a live
    gateway.

Usage::

    from niaharness.gateway.scale_to_zero import (
        scale_to_zero_enabled,
        should_arm,
        is_idle,
    )

    if should_arm(
        enabled=scale_to_zero_enabled(),
        relay_only_or_absent=messaging_is_relay_only_or_absent(platforms),
        wake_url=wake_url,
    ):
        # Start the idle watcher.
        if is_idle(
            running_agent_count=0,
            seconds_since_last_inbound=600,
            idle_timeout_seconds=300,
            has_live_background_work=False,
        ):
            await go_dormant()
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

# Env flag stamped when the scale-to-zero feature is enabled.
SCALE_TO_ZERO_ENV = "NIA_SCALE_TO_ZERO"

# Default idle timeout (config.yaml).
DEFAULT_IDLE_TIMEOUT_MINUTES = 5

_TRUTHY = {"1", "true", "yes", "on"}


def scale_to_zero_enabled(environ: Optional[dict] = None) -> bool:
    """Whether the per-instance scale-to-zero flag is on.

    Absent/blank/falsey → disabled (fail-safe default off).
    """
    env = environ if environ is not None else os.environ
    return str(env.get(SCALE_TO_ZERO_ENV, "")).strip().lower() in _TRUTHY


def parse_idle_timeout_seconds(
    cfg_value: Any,
    default_minutes: int = DEFAULT_IDLE_TIMEOUT_MINUTES,
) -> float:
    """Coerce ``scale_to_zero.idle_timeout_minutes`` to seconds.

    Degrades to the default on any non-numeric / non-positive value
    (never raises, never returns <= 0 — a zero/negative timeout would
    make the gateway go dormant instantly).
    """
    try:
        minutes = float(cfg_value)
    except (TypeError, ValueError):
        minutes = float(default_minutes)
    if minutes <= 0:
        minutes = float(default_minutes)
    return minutes * 60.0


def messaging_is_relay_only_or_absent(platforms: Iterable[Any]) -> bool:
    """True iff the only connected messaging platform is RELAY, or there is none.

    A directly-connected platform (Discord/Telegram/Slack/...) holds a
    live socket and cannot scale to zero, so its presence disarms the
    feature.
    """
    names = {_platform_name(p) for p in platforms}
    names.discard("relay")
    return len(names) == 0


def _platform_name(platform: Any) -> str:
    """Extract the platform name from an enum/string/adapter."""
    value = getattr(platform, "value", platform)
    # Also handle PlatformAdapter instances.
    name = getattr(value, "platform_name", value)
    return str(name).strip().lower()


def should_arm(
    *,
    enabled: bool,
    relay_only_or_absent: bool,
    wake_url: Optional[str],
) -> bool:
    """Whether to start the idle watcher at all.

    ALL must hold: the flag is on, messaging is relay-only/absent, and a
    wakeUrl is registered (a suspended instance with no reachable wake
    target is a black hole). Any unmet → the watcher never starts.
    """
    return bool(enabled) and bool(relay_only_or_absent) and bool(wake_url)


def is_idle(
    *,
    running_agent_count: int,
    seconds_since_last_inbound: float,
    idle_timeout_seconds: float,
    has_live_background_work: bool,
) -> bool:
    """The idle predicate. Pure — composes the three conjuncts.

    Idle iff: no in-flight agent turn, no inbound within the timeout
    window, and no live background work. Any active work keeps the
    gateway awake — suspending mid-flight would lose it.
    """
    if running_agent_count > 0:
        return False
    if has_live_background_work:
        return False
    return seconds_since_last_inbound >= idle_timeout_seconds


__all__ = [
    "DEFAULT_IDLE_TIMEOUT_MINUTES",
    "SCALE_TO_ZERO_ENV",
    "is_idle",
    "messaging_is_relay_only_or_absent",
    "parse_idle_timeout_seconds",
    "scale_to_zero_enabled",
    "should_arm",
]
