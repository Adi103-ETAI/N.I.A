"""Preview restart agent helpers.

Ported from hermes-agent/tui_gateway/server.py (preview.restart path).

Provides the ephemeral agent kwargs, callback builders, and history
extraction that preview.restart needs to spawn a background agent that
recovers a broken local preview URL.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def preview_restart_history(
    session: dict, max_messages: int = 24, max_tool_chars: int = 1200,
) -> List[Dict[str, Any]]:
    """Extract a compact history for the preview restart agent.

    Ported from hermes-agent/tui_gateway/server.py line 4023.

    The preview restart agent gets the parent session's recent history so it
    can figure out which server was running. Tool results are truncated to
    keep the prompt manageable.
    """
    history = list(session.get("history") or [])
    if not history:
        return []

    # Take the last N messages.
    recent = history[-max_messages:] if len(history) > max_messages else history

    compacted: List[Dict[str, Any]] = []
    for msg in recent:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content") or msg.get("text", "")
        if isinstance(content, list):
            # Multimodal — extract text parts.
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}:
                    parts.append(str(part.get("text") or ""))
                elif isinstance(part, str):
                    parts.append(part)
            content = "\n".join(parts)
        content = str(content)
        if role == "tool":
            # Truncate tool results.
            if len(content) > max_tool_chars:
                content = content[:max_tool_chars] + "... [truncated]"
        compacted.append({"role": role, "content": content})

    return compacted


def preview_restart_callbacks(
    parent_sid: str, task_id: str,
) -> Dict[str, Any]:
    """Build the callback dict for the preview restart agent.

    Ported from hermes-agent/tui_gateway/server.py line 4104.

    Returns callbacks that emit preview.restart.progress / .complete events
    to the parent session.
    """
    # Import here to avoid circular import.
    from niaharness.tui_gateway.server import _emit

    def _on_progress(text: str) -> None:
        _emit("preview.restart.progress", parent_sid, {"task_id": task_id, "text": text})

    def _on_complete(text: str) -> None:
        _emit("preview.restart.complete", parent_sid, {"task_id": task_id, "text": text})

    def _on_error(text: str) -> None:
        _emit("preview.restart.complete", parent_sid, {"task_id": task_id, "text": f"error: {text}"})

    return {
        "on_progress": _on_progress,
        "on_complete": _on_complete,
        "on_error": _on_error,
    }


def ephemeral_preview_agent_kwargs(
    agent: Any, task_id: str,
) -> Dict[str, Any]:
    """Build kwargs for an ephemeral preview-restart agent.

    Ported from hermes-agent/tui_gateway/server.py line 4011.

    The preview agent inherits the parent agent's provider/model/credentials
    so its output matches the model the user is actually using. It runs with
    a reduced toolset (no memory, no skills) to keep it focused.
    """
    kwargs: Dict[str, Any] = {}

    if agent is not None:
        # Inherit provider/model/credentials from the parent agent.
        for field_name in ("provider", "model", "base_url", "api_key", "api_mode"):
            val = getattr(agent, field_name, None)
            if val:
                kwargs[field_name] = val

    # The preview agent gets its own task_id so terminal commands run in
    # the right cwd context.
    kwargs["task_id"] = task_id

    # Reduced toolset — no memory, no skills, no cron.
    kwargs["disable_memory"] = True
    kwargs["disable_skills"] = True
    kwargs["disable_cron"] = True

    return kwargs


def _preview_tool_result_preview(name: str, result: str) -> str:
    """Short preview of a tool result for the restart history.

    Ported from hermes-agent/tui_gateway/server.py line 4082.
    """
    if not result:
        return ""
    # Keep first 200 chars for common tools, 500 for terminal (which often
    # has the server startup output we need).
    limit = 500 if name in {"terminal", "bash", "shell"} else 200
    if len(result) > limit:
        return result[:limit] + "... [truncated]"
    return result


__all__ = [
    "preview_restart_history",
    "preview_restart_callbacks",
    "ephemeral_preview_agent_kwargs",
    "_preview_tool_result_preview",
]
