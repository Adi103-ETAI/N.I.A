"""Pre-Flight Approval Surface — Sprint 2.

This module presents the MissionManifest to the user for a single,
structured approval before the swarm begins execution.

The approval surface is pluggable (CLI now, Web UI / API later).
"""
from __future__ import annotations

import logging

from src.core.schema.mission import MissionManifest
from src.core.policy.scopes import CapabilityScope
from src.core.policy.engine import audit_plan, CapabilityAudit

logger = logging.getLogger("NIA.PreFlight")


# ─────────────────────────────────────────────────────────────────────────────
# Scope Emoji & Labels
# ─────────────────────────────────────────────────────────────────────────────

_SCOPE_LABELS = {
    CapabilityScope.READ_ONLY:   ("📖", "read_only",   "auto"),
    CapabilityScope.WRITE:       ("✏️ ", "write",       "needs approval"),
    CapabilityScope.EXECUTE:     ("⚙️ ", "execute",     "needs approval"),
    CapabilityScope.NETWORK:     ("🌐", "network",     "needs approval"),
    CapabilityScope.AGENT_SPAWN: ("🤖", "agent_spawn", "needs approval"),
    CapabilityScope.DESTRUCTIVE: ("💣", "destructive", "needs approval ⚠️"),
}

_MODE_LABELS = {
    "fast":     "⚡ fast      (1–2 steps, simple)",
    "standard": "🔄 standard  (3–5 steps, typical)",
    "deep":     "🔭 deep      (6+ steps, multi-agent)",
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI Approval Surface
# ─────────────────────────────────────────────────────────────────────────────

def render_preflight(manifest: MissionManifest, audit: CapabilityAudit) -> str:
    """Render the pre-flight prompt as a formatted string for display."""

    lines = ["", "─" * 58, "  N.I.A. — Mission Pre-Flight Approval", "─" * 58]
    lines.append(f"  Mission : {manifest.mission_id}")
    lines.append(f"  Intent  : {manifest.intent}")
    lines.append(f"  Mode    : {_MODE_LABELS.get(manifest.execution_mode, manifest.execution_mode)}")
    lines.append(f"  Agents  : {manifest.estimated_agents}  │  Depth: {manifest.estimated_depth}")
    lines.append("")
    lines.append("  Plan:")

    for i, step in enumerate(manifest.steps, 1):
        scopes_str = ", ".join(s.value for s in step.required_scopes)
        lines.append(f"    {i}. [{step.assigned_role:^10}] {step.description}")
        lines.append(f"           scopes: {scopes_str}")

    lines.append("")
    lines.append("  Scope Summary:")
    for scope in audit.auto_approved:
        emoji, label, status = _SCOPE_LABELS[scope]
        lines.append(f"    {emoji} {label:<14} — {status}")
    for scope in audit.needs_approval:
        emoji, label, status = _SCOPE_LABELS[scope]
        lines.append(f"    {emoji} {label:<14} — {status}")

    lines.append("")
    lines.append("─" * 58)
    lines.append("  Proceed? [y] yes  [n] cancel  [?] show full plan")
    lines.append("─" * 58)
    return "\n".join(lines)


async def run_preflight_approval(manifest: MissionManifest) -> MissionManifest:
    """
    Show the Mission Plan to the user and wait for approval via CLI.

    This is the SINGLE pre-flight gate described in the Master Plan.
    After this:  manifest.approved = True, manifest.approved_scopes populated.
    No further interruptions until the swarm hits a genuine blocker.

    Returns:
        Updated manifest with approved=True if user confirmed, or
        the original manifest with approved=False if cancelled.
    """
    try:
        import aioconsole
    except ImportError:
        logger.error("aioconsole not installed — cannot run pre-flight approval interactively.")
        return manifest

    audit = audit_plan(manifest)

    # Auto-approve missions with no dangerous scopes (read-only fallback plans)
    if not audit.needs_approval:
        logger.info(f"Pre-flight: All scopes auto-approved for '{manifest.mission_id}'")
        manifest.approved = True
        manifest.approved_scopes = audit.auto_approved[:]
        return manifest

    # Interactive approval for missions needing write/execute/etc.
    prompt = render_preflight(manifest, audit)
    print(prompt)

    while True:
        try:
            answer = (await aioconsole.ainput("  > ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Mission cancelled.")
            return manifest

        if answer in ("y", "yes"):
            manifest.approved = True
            manifest.approved_scopes = (
                audit.auto_approved + audit.needs_approval
            )
            logger.info(f"✅ Mission '{manifest.mission_id}' approved — scopes: "
                        f"{[s.value for s in manifest.approved_scopes]}")
            print("  ✅ Approved. Starting autonomous execution…\n")
            return manifest

        elif answer in ("n", "no", "cancel"):
            logger.info(f"❌ Mission '{manifest.mission_id}' cancelled by user.")
            print("  ❌ Mission cancelled.\n")
            return manifest

        elif answer in ("?", "plan"):
            # Show verbose step list
            for i, step in enumerate(manifest.steps, 1):
                print(f"    Step {i}: {step.description}")
                print(f"             Role : {step.assigned_role}")
                print(f"             Scopes: {[s.value for s in step.required_scopes]}")
            print()

        else:
            print("  Please enter y (yes), n (cancel), or ? (show plan).")
