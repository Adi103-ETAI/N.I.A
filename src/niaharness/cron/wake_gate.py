"""P1 Cron wake gate — parse ``{"wakeAgent": false}`` from script stdout.

Ported from Hermes Agent's ``cron/scheduler.py`` _parse_wake_gate
function (lines 2041-2064).

The convention (ported from nanoclaw): if the last stdout line of a
cron job's pre-check script is JSON like ``{"wakeAgent": false}``, the
agent is skipped entirely — no LLM run, no delivery. Any other output
(non-JSON, missing flag, gate absent, or ``wakeAgent: true``) means
wake the agent normally.

This lets a script act as a conditional trigger: the script checks
some condition (e.g. "are there new important emails?") and only wakes
the agent when the condition is met. Without this, every cron tick
would run the full agent — wasting tokens on no-op runs.

Usage::

    from niaharness.cron.wake_gate import parse_wake_gate

    success, stdout = run_script(job["script"])
    if not parse_wake_gate(stdout):
        # Skip the agent run — script said don't wake.
        return
    # Proceed with the agent run.
    await run_agent(job)
"""

from __future__ import annotations

import json
from typing import Any


def parse_wake_gate(script_output: str) -> bool:
    """Parse the last non-empty stdout line of a cron job's pre-check script
    as a wake gate.

    Returns True if the agent should wake, False to skip.

    Convention:
      - Last stdout line is ``{"wakeAgent": false}`` → return False (skip agent).
      - Any other output (non-JSON, missing flag, ``wakeAgent: true``) → return True.
      - Empty output → return True (wake the agent normally).
    """
    if not script_output:
        return True
    # Get all non-empty lines.
    stripped_lines = [line for line in script_output.splitlines() if line.strip()]
    if not stripped_lines:
        return True
    last_line = stripped_lines[-1].strip()
    try:
        gate = json.loads(last_line)
    except (json.JSONDecodeError, ValueError):
        # Not JSON → wake the agent normally.
        return True
    if not isinstance(gate, dict):
        return True
    # Only ``wakeAgent: false`` (exactly) skips the agent.
    return gate.get("wakeAgent", True) is not False


def build_wake_gate_output(
    *,
    wake_agent: bool = True,
    extra: dict[str, Any] | None = None,
) -> str:
    """Build a wake-gate JSON line for use in a pre-check script.

    The script can print this as its last stdout line to control whether
    the agent runs. Useful for testing + for scripts that want to include
    additional context.

    Args:
        wake_agent: True to wake the agent, False to skip.
        extra: Additional keys to include in the JSON object.

    Returns:
        A JSON string suitable for printing as the last stdout line.
    """
    payload: dict[str, Any] = {"wakeAgent": bool(wake_agent)}
    if extra:
        payload.update(extra)
    return json.dumps(payload)


__all__ = [
    "build_wake_gate_output",
    "parse_wake_gate",
]
