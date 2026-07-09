"""Permission checking for tool execution.

Enhanced with shell hardening (deobfuscation + hardline blocklist +
dangerous patterns) from ``niaharness.permissions.shell_hardening`` and
the per-session approval layer from ``niaharness.permissions.approval``.

Flow for shell commands:

  1. Tool deny/allow lists (settings.permission.denied_tools / allowed_tools)
  2. Path-level rules (settings.permission.path_rules)
  3. Shell hardening gate (``shell_hardening.check_command``):
     - Hardline floor (rm -rf /, mkfs, dd to block device, etc.) — unconditional block
     - Sudo stdin guard (sudo -S without SUDO_PASSWORD) — unconditional block
     - User deny patterns — unconditional block
     - Dangerous patterns — ``requires_confirmation``
  4. Approval layer (``approval.ApprovalChecker.check``):
     - YOLO / session-yolo / mode=off bypass
     - Permanent allowlist (exact match or fnmatch glob)
     - Session-scoped approval
     - Smart-approve (LLM auto-approve for low-risk commands)
     - Gateway async approval (blocking wait for chat-based /approve /deny)
     - CLI interactive prompt (fallback)
  5. Mode-based decision (FULL_AUTO / PLAN / DEFAULT)

The approval layer runs BEFORE the mode-based decision so that even in
DEFAULT mode, a command on the permanent allowlist or a smart-approved
low-risk command proceeds without a manual prompt.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from niaharness.config.settings import PermissionSettings
from niaharness.permissions.approval import (
    ApprovalChecker,
    ApprovalConfig,
    ApprovalDecision,
    get_current_session_key,
)
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
    category: str = "ok"  # ok | hardline | sudo_stdin | user_deny | dangerous | denied_tool | path_rule | approval
    pattern_key: str | None = None
    description: str | None = None
    approval_choice: str | None = None  # once | session | always | deny (from approval layer)


@dataclass(frozen=True)
class PathRule:
    """A glob-based path permission rule."""

    pattern: str
    allow: bool  # True = allow, False = deny


class PermissionChecker:
    """Evaluate tool usage against the configured permission mode and rules.

    The checker owns a lazily-initialized :class:`ApprovalChecker` that
    handles the per-session approval flow for dangerous commands. The
    approval checker is constructed on first use so it picks up config
    changes (e.g. ``~/.nia/approvals.json`` written by the gateway).
    """

    def __init__(
        self,
        settings: PermissionSettings,
        *,
        approval_checker: ApprovalChecker | None = None,
        approval_callback=None,
    ) -> None:
        self._settings = settings
        # Parse path rules from settings.
        self._path_rules: list[PathRule] = []
        for rule in getattr(settings, "path_rules", []):
            pattern = getattr(rule, "pattern", None) or (
                rule.get("pattern") if isinstance(rule, dict) else None
            )
            allow = (
                getattr(rule, "allow", True)
                if not isinstance(rule, dict)
                else rule.get("allow", True)
            )
            if pattern:
                self._path_rules.append(PathRule(pattern=pattern, allow=allow))

        # Approval checker — lazily constructed if not provided.
        self._approval_checker = approval_checker
        self._approval_callback = approval_callback

    @property
    def approval_checker(self) -> ApprovalChecker:
        """Lazily construct the ApprovalChecker on first use.

        Construction reads ``~/.nia/approvals.json`` so config changes
        (e.g. mode switch from manual → smart) take effect on the next
        dangerous command without restarting NIA.
        """
        if self._approval_checker is None:
            self._approval_checker = ApprovalChecker(
                approval_callback=self._approval_callback,
            )
        return self._approval_checker

    def evaluate(
        self,
        tool_name: str,
        *,
        is_read_only: bool,
        file_path: str | None = None,
        command: str | None = None,
    ) -> PermissionDecision:
        """Return whether the tool may run immediately.

        See module docstring for the full flow. The approval layer is
        consulted ONLY for commands that the shell-hardening gate flagged
        as ``requires_confirmation`` — hardline / sudo-stdin / user-deny
        blocks are returned immediately without consulting the approval
        layer (those have no recovery path).
        """
        # 1. Explicit tool deny list.
        if tool_name in self._settings.denied_tools:
            return PermissionDecision(
                allowed=False,
                reason=f"{tool_name} is explicitly denied",
                category="denied_tool",
            )

        # 2. Explicit tool allow list.
        if tool_name in self._settings.allowed_tools:
            return PermissionDecision(
                allowed=True,
                reason=f"{tool_name} is explicitly allowed",
            )

        # 3. Path-level rules.
        if file_path and self._path_rules:
            for rule in self._path_rules:
                if fnmatch.fnmatch(file_path, rule.pattern):
                    if not rule.allow:
                        return PermissionDecision(
                            allowed=False,
                            reason=f"Path {file_path} matches deny rule: {rule.pattern}",
                            category="path_rule",
                        )

        # 4. Shell hardening gate (deobfuscation + hardline + dangerous).
        if command:
            user_deny_patterns = list(getattr(self._settings, "denied_commands", []))
            full_auto = self._settings.mode == PermissionMode.FULL_AUTO
            hardening_decision = check_command(
                command,
                user_deny_patterns=user_deny_patterns,
                full_auto=full_auto,
            )

            # 4a. Hard block (hardline / sudo_stdin / user_deny) — no recovery.
            if not hardening_decision.allowed and not hardening_decision.requires_confirmation:
                append_permission_audit_log(
                    command=command,
                    decision=hardening_decision,
                    tool_name=tool_name,
                    session_id=get_current_session_key(default=""),
                )
                return PermissionDecision(
                    allowed=False,
                    reason=hardening_decision.reason,
                    category=hardening_decision.category,
                    description=hardening_decision.description,
                )

            # 4b. Dangerous — requires confirmation. Delegate to the approval layer.
            if hardening_decision.requires_confirmation:
                # In FULL_AUTO mode, the shell-hardening gate already allowed
                # the command (with a warning). Skip the approval layer so
                # FULL_AUTO doesn't prompt.
                if full_auto:
                    append_permission_audit_log(
                        command=command,
                        decision=hardening_decision,
                        tool_name=tool_name,
                        session_id=get_current_session_key(default=""),
                    )
                    return PermissionDecision(
                        allowed=True,
                        reason=f"Dangerous command allowed under FULL_AUTO: {hardening_decision.description}",
                        category="dangerous",
                        description=hardening_decision.description,
                    )

                # Consult the approval layer: permanent allowlist → session →
                # smart-approve → gateway → CLI prompt.
                approval_decision = self.approval_checker.check(
                    command=command,
                    pattern_key=hardening_decision.description or "dangerous_command",
                    description=hardening_decision.description or "potentially dangerous command",
                )

                # Log every approval-layer decision for forensic review.
                append_permission_audit_log(
                    command=command,
                    decision=hardening_decision,
                    tool_name=tool_name,
                    session_id=get_current_session_key(default=""),
                )

                if approval_decision.approved:
                    return PermissionDecision(
                        allowed=True,
                        reason=f"Approved via {approval_decision.category}: {hardening_decision.description}",
                        category="approval",
                        pattern_key=approval_decision.pattern_key,
                        description=approval_decision.description,
                        approval_choice=approval_decision.choice,
                    )

                # Not approved — block.
                # If the approval layer set requires_confirmation (gateway
                # pending without notify_cb), propagate it so the UI can
                # show "waiting for approval".
                return PermissionDecision(
                    allowed=False,
                    requires_confirmation=approval_decision.requires_confirmation,
                    reason=approval_decision.reason,
                    category=approval_decision.category,
                    pattern_key=approval_decision.pattern_key,
                    description=approval_decision.description,
                    approval_choice=approval_decision.choice,
                )

            # 4c. Also check legacy fnmatch patterns (backward compat).
            for pattern in getattr(self._settings, "denied_commands", []):
                if isinstance(pattern, str) and fnmatch.fnmatch(command, pattern):
                    return PermissionDecision(
                        allowed=False,
                        reason=f"Command matches deny pattern: {pattern}",
                        category="user_deny",
                    )

        # 5. Mode-based decision.
        if self._settings.mode == PermissionMode.FULL_AUTO:
            return PermissionDecision(
                allowed=True,
                reason="Auto mode allows all tools",
                category="ok",
            )

        # Read-only tools always allowed.
        if is_read_only:
            return PermissionDecision(
                allowed=True,
                reason="read-only tools are allowed",
            )

        # Plan mode: block mutating tools.
        if self._settings.mode == PermissionMode.PLAN:
            return PermissionDecision(
                allowed=False,
                reason="Plan mode blocks mutating tools until the user exits plan mode",
            )

        # Default mode: require confirmation for mutating tools.
        return PermissionDecision(
            allowed=False,
            requires_confirmation=True,
            reason="Mutating tools require user confirmation in default mode",
        )
