"""P1 Gateway slash command handlers.

Ported from Hermes Agent's ``gateway/slash_commands.py`` (4621 LOC),
scoped to NIA's architecture. Provides in-session slash command handlers
for the GatewayRunner:

  - ``/help`` — list available commands.
  - ``/new`` / ``/reset`` — start a new conversation (clear session).
  - ``/status`` — show gateway + session status.
  - ``/yolo`` — toggle session-scoped yolo mode (skip approvals).
  - ``/update`` — trigger a self-update.
  - ``/whoami`` — show the user's platform + ID.
  - ``/queue`` — show queued messages.
  - ``/cancel`` — cancel the current in-flight turn.

Each handler is an async function that takes a SlashCommandContext and
returns a string (the reply to send to chat) or None (no reply).

Usage::

    from niaharness.gateway.slash_commands import (
        SlashCommandRegistry,
        handle_slash_command,
    )

    registry = SlashCommandRegistry()
    registry.register("help", handle_help)
    registry.register("new", handle_new)

    if text.startswith("/"):
        reply = await handle_slash_command(text, context, registry)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class SlashCommandContext:
    """Context passed to every slash command handler.

    Attributes:
        platform: The platform name (e.g. "telegram").
        chat_id: The chat/channel ID.
        user_id: The user ID of the sender.
        user_name: The user's display name.
        session_id: The current session ID (or None).
        args: The arguments passed to the command (after the command name).
        raw_text: The full raw text including the slash + command name.
        gateway_runner: The GatewayRunner instance (optional).
        metadata: Platform-specific metadata.
    """

    platform: str
    chat_id: str
    user_id: str
    user_name: Optional[str] = None
    session_id: Optional[str] = None
    args: str = ""
    raw_text: str = ""
    gateway_runner: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Type for a slash command handler.
SlashCommandHandler = Callable[[SlashCommandContext], Any]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SlashCommandRegistry:
    """Registry of slash command handlers.

    Commands are case-insensitive. The registry supports aliases (multiple
    names → same handler).
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, SlashCommandHandler] = {}
        self._descriptions: Dict[str, str] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        name: str,
        handler: SlashCommandHandler,
        *,
        description: str = "",
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register a slash command handler.

        Args:
            name: The canonical command name (without the leading /).
            handler: Async callable that takes a SlashCommandContext.
            description: Short description for /help.
            aliases: Alternative names for the same command.
        """
        key = name.lower().lstrip("/")
        self._handlers[key] = handler
        if description:
            self._descriptions[key] = description
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower().lstrip("/")] = key

    def get(self, name: str) -> Optional[SlashCommandHandler]:
        """Look up a handler by name or alias."""
        key = name.lower().lstrip("/")
        if key in self._handlers:
            return self._handlers[key]
        canonical = self._aliases.get(key)
        if canonical:
            return self._handlers.get(canonical)
        return None

    def list_commands(self) -> List[str]:
        """Return all registered command names (canonical only)."""
        return sorted(self._handlers.keys())

    def get_descriptions(self) -> Dict[str, str]:
        """Return {command: description} for all commands."""
        return dict(self._descriptions)

    def has(self, name: str) -> bool:
        """Return True if a command is registered."""
        return self.get(name) is not None


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------


def parse_slash_command(text: str) -> Optional[tuple[str, str]]:
    """Parse a slash command from text.

    Returns (command_name, args) or None if text doesn't start with /.

    Examples:
        "/help" → ("help", "")
        "/new conversation" → ("new", "conversation")
        "hello" → None
    """
    if not text or not text.startswith("/"):
        return None
    stripped = text[1:].strip()
    if not stripped:
        return None
    parts = stripped.split(None, 1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    return (command, args)


async def handle_slash_command(
    text: str,
    context: SlashCommandContext,
    registry: SlashCommandRegistry,
) -> Optional[str]:
    """Handle a slash command. Returns the reply text (or None).

    If the text isn't a slash command, returns None. If the command
    isn't registered, returns an error message.
    """
    parsed = parse_slash_command(text)
    if parsed is None:
        return None
    command, args = parsed
    context.args = args
    context.raw_text = text

    handler = registry.get(command)
    if handler is None:
        return f"Unknown command: /{command}. Try /help for available commands."

    try:
        import asyncio
        if asyncio.iscoroutinefunction(handler):
            result = await handler(context)
        else:
            result = handler(context)
        return result
    except Exception as exc:
        logger.error("Slash command /%s failed: %s", command, exc)
        return f"Command /{command} failed: {exc}"


# ---------------------------------------------------------------------------
# Built-in command handlers
# ---------------------------------------------------------------------------


async def handle_help(context: SlashCommandContext) -> str:
    """Handle /help — list available commands."""
    registry: Optional[SlashCommandRegistry] = context.metadata.get("registry")
    if registry is None:
        return "Help not available."
    commands = registry.list_commands()
    descriptions = registry.get_descriptions()
    lines = ["Available commands:"]
    for cmd in commands:
        desc = descriptions.get(cmd, "")
        if desc:
            lines.append(f"  /{cmd} — {desc}")
        else:
            lines.append(f"  /{cmd}")
    return "\n".join(lines)


async def handle_new(context: SlashCommandContext) -> str:
    """Handle /new or /reset — start a new conversation."""
    runner = context.gateway_runner
    if runner is not None and hasattr(runner, "reset_session"):
        try:
            runner.reset_session(context.platform, context.chat_id)
            return "Started a new conversation. Previous context cleared."
        except Exception as exc:
            logger.error("Session reset failed: %s", exc)
            return f"Could not reset session: {exc}"
    return "Started a new conversation."


async def handle_reset(context: SlashCommandContext) -> str:
    """Alias for /new."""
    return await handle_new(context)


async def handle_status(context: SlashCommandContext) -> str:
    """Handle /status — show gateway + session status."""
    lines = ["NIA Gateway Status"]
    lines.append(f"  Platform: {context.platform}")
    lines.append(f"  Chat ID: {context.chat_id}")
    lines.append(f"  User: {context.user_name or context.user_id}")
    if context.session_id:
        lines.append(f"  Session: {context.session_id}")

    runner = context.gateway_runner
    if runner is not None:
        if hasattr(runner, "get_status"):
            try:
                status = runner.get_status()
                if isinstance(status, dict):
                    for k, v in status.items():
                        lines.append(f"  {k}: {v}")
            except Exception:
                pass
        adapters = getattr(runner, "list_adapters", lambda: [])()
        if adapters:
            lines.append(f"  Adapters: {', '.join(adapters)}")

    from niaharness.gateway.status import is_gateway_running, get_running_pid
    pid = get_running_pid()
    lines.append(f"  Gateway running: {'yes' if pid else 'no'}")
    if pid:
        lines.append(f"  Gateway PID: {pid}")

    return "\n".join(lines)


async def handle_yolo(context: SlashCommandContext) -> str:
    """Handle /yolo — toggle session-scoped yolo mode."""
    runner = context.gateway_runner
    if runner is None:
        return "Yolo mode not available (no gateway runner)."
    if not hasattr(runner, "toggle_yolo"):
        return "Yolo mode not supported."
    try:
        enabled = runner.toggle_yolo(context.platform, context.chat_id)
        if enabled:
            return "Yolo mode ON — approval prompts skipped for this session."
        else:
            return "Yolo mode OFF — approval prompts restored."
    except Exception as exc:
        return f"Could not toggle yolo: {exc}"


async def handle_update(context: SlashCommandContext) -> str:
    """Handle /update — trigger a self-update."""
    try:
        from niaharness.cli.update import perform_update
        result = perform_update()
        return f"Update result: {result}"
    except Exception as exc:
        return f"Update failed: {exc}"


async def handle_whoami(context: SlashCommandContext) -> str:
    """Handle /whoami — show the user's platform + ID."""
    lines = [
        "You are:",
        f"  Platform: {context.platform}",
        f"  User ID: {context.user_id}",
    ]
    if context.user_name:
        lines.append(f"  Username: {context.user_name}")
    return "\n".join(lines)


async def handle_cancel(context: SlashCommandContext) -> str:
    """Handle /cancel — cancel the current in-flight turn."""
    runner = context.gateway_runner
    if runner is None or not hasattr(runner, "cancel_turn"):
        return "Cancel not available."
    try:
        cancelled = runner.cancel_turn(context.platform, context.chat_id)
        if cancelled:
            return "Turn cancelled."
        return "No turn in flight."
    except Exception as exc:
        return f"Cancel failed: {exc}"


async def handle_queue(context: SlashCommandContext) -> str:
    """Handle /queue — show queued messages."""
    runner = context.gateway_runner
    if runner is None or not hasattr(runner, "get_queue"):
        return "Queue not available."
    try:
        queue = runner.get_queue(context.platform, context.chat_id)
        if not queue:
            return "No queued messages."
        lines = [f"Queued messages ({len(queue)}):"]
        for i, msg in enumerate(queue):
            lines.append(f"  [{i}] {str(msg)[:80]}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Queue lookup failed: {exc}"


# ---------------------------------------------------------------------------
# Default registry factory
# ---------------------------------------------------------------------------


def create_default_registry() -> SlashCommandRegistry:
    """Create a SlashCommandRegistry with all built-in commands registered."""
    registry = SlashCommandRegistry()
    registry.register("help", handle_help, description="Show available commands")
    registry.register("new", handle_new, description="Start a new conversation", aliases=["reset"])
    registry.register("reset", handle_reset, description="Alias for /new")
    registry.register("status", handle_status, description="Show gateway + session status")
    registry.register("yolo", handle_yolo, description="Toggle approval-skipping mode")
    registry.register("update", handle_update, description="Trigger a self-update")
    registry.register("whoami", handle_whoami, description="Show your platform + user ID")
    registry.register("cancel", handle_cancel, description="Cancel the current turn")
    registry.register("queue", handle_queue, description="Show queued messages")
    return registry


__all__ = [
    "SlashCommandContext",
    "SlashCommandHandler",
    "SlashCommandRegistry",
    "create_default_registry",
    "handle_cancel",
    "handle_help",
    "handle_new",
    "handle_queue",
    "handle_reset",
    "handle_slash_command",
    "handle_status",
    "handle_update",
    "handle_whoami",
    "handle_yolo",
    "parse_slash_command",
]
