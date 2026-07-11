"""P1 Cron toolset resolver — disabled/enabled toolsets + MCP merging.

Ported from Hermes Agent's ``cron/scheduler.py`` toolset functions
(lines 116-200).

Cron-spawned agents get a restricted toolset:
  - ``cronjob`` is always disabled (would let a cron agent schedule
    more cron jobs — recursion foot-gun).
  - ``messaging`` is always disabled (interactive, needs a live gateway
    session).
  - ``clarify`` is always disabled (interactive, blocks waiting for
    user input).

Per-job ``enabled_toolsets`` overrides the default, with MCP servers
layered on top so a native-toolset allowlist doesn't silently strip
MCP tools.

Usage::

    from niaharness.cron.toolset_resolver import resolve_cron_toolsets

    toolsets = resolve_cron_toolsets(job, config)
    if toolsets is not None:
        agent = AIAgent(enabled_toolsets=toolsets)
    else:
        agent = AIAgent()  # full default toolset
"""

from __future__ import annotations

import logging
from typing import Any, FrozenSet, List, Optional

logger = logging.getLogger(__name__)

# Toolsets a cron-spawned agent must never receive.
# - ``cronjob`` — would let a cron-spawned agent schedule more cron jobs.
# - ``messaging`` — interactive, needs a live gateway session.
# - ``clarify`` — interactive, blocks waiting for user input.
_DEFAULT_DISABLED_TOOLSETS: FrozenSet[str] = frozenset({
    "cronjob", "messaging", "clarify",
})


def resolve_cron_disabled_toolsets(cfg: dict[str, Any]) -> List[str]:
    """Toolsets a cron-spawned agent must never receive.

    Three protected toolsets are always disabled in cron context.
    User-level ``agent.disabled_toolsets`` from config.yaml is layered
    on top so per-job ``enabled_toolsets`` cannot bypass policy that
    applies to ordinary agent runs.
    """
    disabled = list(_DEFAULT_DISABLED_TOOLSETS)
    agent_cfg = (cfg or {}).get("agent") or {}
    user_disabled = agent_cfg.get("disabled_toolsets") or []
    for name in user_disabled:
        name = str(name).strip()
        if name and name not in disabled:
            disabled.append(name)
    return disabled


def merge_mcp_into_per_job_toolsets(
    per_job: List[str],
    cfg: dict[str, Any],
) -> List[str]:
    """Layer enabled MCP servers onto a per-job ``enabled_toolsets`` allowlist.

    A per-job list scopes the *native* toolsets, but on its own it
    silently drops every MCP server. This restores parity with the
    gateway/CLI MCP semantics:

      * ``no_mcp`` sentinel present → no MCP servers (sentinel stripped).
      * One or more MCP server names already listed → treat as an
        allowlist, add nothing further (the user named exactly the
        servers they want).
      * Otherwise → union in every globally-enabled MCP server.
    """
    result = [t for t in per_job if t != "no_mcp"]
    if "no_mcp" in per_job:
        return result

    # Get enabled MCP server names from config.
    enabled_mcp = _enabled_mcp_server_names(cfg)
    if not enabled_mcp:
        return result

    # If the per-job list already names MCP servers, treat it as an
    # allowlist — don't add more.
    if set(result) & enabled_mcp:
        return result

    # Union in every globally-enabled MCP server.
    for name in sorted(enabled_mcp):
        if name not in result:
            result.append(name)
    return result


def _enabled_mcp_server_names(cfg: dict[str, Any]) -> FrozenSet[str]:
    """Get the set of enabled MCP server names from config.

    Best-effort: returns an empty set if MCP config can't be loaded.
    """
    try:
        mcp_cfg = (cfg or {}).get("mcp") or {}
        if not isinstance(mcp_cfg, dict):
            return frozenset()
        servers = mcp_cfg.get("servers") or {}
        if not isinstance(servers, dict):
            return frozenset()
        enabled = set()
        for name, server_cfg in servers.items():
            if isinstance(server_cfg, dict):
                if server_cfg.get("enabled", True):
                    enabled.add(str(name).strip().lower())
            else:
                enabled.add(str(name).strip().lower())
        return frozenset(enabled)
    except Exception:
        return frozenset()


def resolve_cron_enabled_toolsets(
    job: dict[str, Any],
    cfg: dict[str, Any],
) -> Optional[List[str]]:
    """Resolve the toolset list for a cron job.

    Precedence:
      1. Per-job ``enabled_toolsets`` (set via cronjob tool on create/update).
         Enabled MCP servers are layered on per
         ``merge_mcp_into_per_job_toolsets``.
      2. ``None`` — the agent loads the full default toolset (legacy
         behavior, preserved as the safety net).

    Args:
        job: The cron job dict.
        cfg: The gateway/agent config dict.

    Returns:
        A list of toolset names, or None to use the agent's default.
    """
    per_job = job.get("enabled_toolsets")
    if per_job:
        if isinstance(per_job, str):
            per_job = [per_job]
        return merge_mcp_into_per_job_toolsets(list(per_job), cfg or {})

    # No per-job override → use the default (None = full toolset).
    # The disabled toolsets are enforced separately by the agent.
    return None


def resolve_cron_toolsets(
    job: dict[str, Any],
    cfg: Optional[dict[str, Any]] = None,
) -> tuple[Optional[List[str]], List[str]]:
    """Resolve both enabled + disabled toolsets for a cron job.

    Returns:
        (enabled_toolsets, disabled_toolsets). enabled_toolsets may be
        None (use agent default). disabled_toolsets is always a list
        (includes the default cron-disabled set + user config).
    """
    if cfg is None:
        cfg = {}
    enabled = resolve_cron_enabled_toolsets(job, cfg)
    disabled = resolve_cron_disabled_toolsets(cfg)
    return enabled, disabled


__all__ = [
    "merge_mcp_into_per_job_toolsets",
    "resolve_cron_disabled_toolsets",
    "resolve_cron_enabled_toolsets",
    "resolve_cron_toolsets",
]
