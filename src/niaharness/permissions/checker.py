"""Permission checking for tool execution.

Enhanced with shell hardening (deobfuscation + hardline blocklist +
dangerous patterns) from ``niaharness.permissions.shell_hardening``.
Shell commands are normalized before pattern matching so that
obfuscation techniques (backslash-escapes, empty-string splits, $IFS
expansions, line continuations, Unicode fullwidth) cannot bypass
detection.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from niaharness.config.settings import PermissionSettings
from niaharness.permissions.modes import PermissionMode
from niaharness.permissions.shell_hardening import (
    ShellHardeningDecision,
    append_permission_audit_log,
    check_command,
)


@dataclass(frozen=True)
class PermissionDecision:
    """Result of checking whether a tool invocation may run."""

    allowed: bool
    requires_confirmation: bool = False
    reason: str = ""
    category: str = "ok"  # ok | hardline | sudo_stdin | user_deny | dangerous | denied_tool | path_rule


@dataclass(frozen=True)
class PathRule:
    """A glob-based path permission rule."""

    pattern: str
    allow: bool  # True = allow, False = deny


class PermissionChecker:
    """Evaluate tool usage against the configured permission mode and rules."""

    def __init__(self, settings: PermissionSettings) -> None:
        self._settings = settings
        # Parse path rules from settings
        self._path_rules: list[PathRule] = []
        for rule in getattr(settings, "path_rules", []):
            pattern = getattr(rule, "pattern", None) or (rule.get("pattern") if isinstance(rule, dict) else None)
            allow = getattr(rule, "allow", True) if not isinstance(rule, dict) else rule.get("allow", True)
            if pattern:
                self._path_rules.append(PathRule(pattern=pattern, allow=allow))

    def evaluate(
        self,
        tool_name: str,
        *,
        is_read_only: bool,
        file_path: str | None = None,
        command: str | None = None,
    ) -> PermissionDecision:
        """Return whether the tool may run immediately."""
        # Explicit tool deny list
        if tool_name in self._settings.denied_tools:
            return PermissionDecision(allowed=False, reason=f"{tool_name} is explicitly denied")

        # Explicit tool allow list
        if tool_name in self._settings.allowed_tools:
            return PermissionDecision(allowed=True, reason=f"{tool_name} is explicitly allowed")

        # Check path-level rules
        if file_path and self._path_rules:
            for rule in self._path_rules:
                if fnmatch.fnmatch(file_path, rule.pattern):
                    if not rule.allow:
                        return PermissionDecision(
                            allowed=False,
                            reason=f"Path {file_path} matches deny rule: {rule.pattern}",
                        )

        # Check command deny patterns (e.g. deny "rm -rf /")
        if command:
            # First, run the shell hardening check (deobfuscation + hardline
            # blocklist + dangerous patterns + sudo stdin guard). This is the
            # primary security gate — it catches obfuscated commands that
            # fnmatch on the raw string would miss (r\m, r''m, rm${IFS}, etc.).
            user_deny_patterns = list(getattr(self._settings, "denied_commands", []))
            full_auto = self._settings.mode == PermissionMode.FULL_AUTO
            hardening_decision = check_command(
                command,
                user_deny_patterns=user_deny_patterns,
                full_auto=full_auto,
            )
            if not hardening_decision.allowed:
                # Log to the permission audit log for forensic review.
                append_permission_audit_log(
                    command=command,
                    decision=hardening_decision,
                    tool_name=tool_name,
                )
                return PermissionDecision(
                    allowed=False,
                    requires_confirmation=hardening_decision.requires_confirmation,
                    reason=hardening_decision.reason,
                    category=hardening_decision.category,
                )
            if hardening_decision.requires_confirmation:
                # Dangerous command in non-auto mode — require confirmation.
                append_permission_audit_log(
                    command=command,
                    decision=hardening_decision,
                    tool_name=tool_name,
                )
                return PermissionDecision(
                    allowed=False,
                    requires_confirmation=True,
                    reason=hardening_decision.reason,
                    category=hardening_decision.category,
                )

            # Also check the legacy fnmatch patterns (for backward compat).
            for pattern in getattr(self._settings, "denied_commands", []):
                if isinstance(pattern, str) and fnmatch.fnmatch(command, pattern):
                    return PermissionDecision(
                        allowed=False,
                        reason=f"Command matches deny pattern: {pattern}",
                        category="user_deny",
                    )

        # Full auto: allow everything (hardline + dangerous already checked above)
        if self._settings.mode == PermissionMode.FULL_AUTO:
            return PermissionDecision(allowed=True, reason="Auto mode allows all tools", category="ok")

        # Read-only tools always allowed
        if is_read_only:
            return PermissionDecision(allowed=True, reason="read-only tools are allowed")

        # Plan mode: block mutating tools
        if self._settings.mode == PermissionMode.PLAN:
            return PermissionDecision(
                allowed=False,
                reason="Plan mode blocks mutating tools until the user exits plan mode",
            )

        # Default mode: require confirmation for mutating tools
        return PermissionDecision(
            allowed=False,
            requires_confirmation=True,
            reason="Mutating tools require user confirmation in default mode",
        )
