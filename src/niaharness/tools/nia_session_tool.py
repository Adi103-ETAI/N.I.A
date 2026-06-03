"""Tool for N.I.A session persistence.

Provides session save, restore, and list operations, allowing
conversations to be resumed across restarts.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


SESSION_DIR = Path.home() / ".nia" / "sessions"


class NiaSessionInput(BaseModel):
    """Arguments for session operations."""

    action: str = Field(
        description="Operation: 'save' to save current session, "
        "'restore' to restore a session by ID, "
        "'list' to list saved sessions, "
        "'new' to start a fresh session"
    )
    session_id: str | None = Field(default=None, description="Session ID (for restore)")
    name: str | None = Field(default=None, description="Human-readable name for the session (for save)")


class NiaSessionTool(BaseTool):
    """N.I.A session persistence tool.

    Save and restore conversation sessions across restarts.
    Sessions are stored in ~/.nia/sessions/ as JSON files.
    """

    name = "nia_session"
    description = (
        "Session persistence for N.I.A. Actions: "
        "save (save current session), "
        "restore (restore a session by ID), "
        "list (list saved sessions), "
        "new (start fresh session)"
    )
    input_model = NiaSessionInput

    def __init__(self, engine: object | None = None) -> None:
        self._engine = engine

    def set_engine(self, engine: object) -> None:
        """Set the QueryEngine instance (called during NIA initialization)."""
        self._engine = engine

    async def execute(self, arguments: NiaSessionInput, context: ToolExecutionContext) -> ToolResult:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        action = arguments.action

        if action == "save":
            if self._engine is None:
                return ToolResult(output="Engine not initialized", is_error=True)
            session_id = arguments.session_id or f"nia-{int(__import__('time').time())}"
            messages = [msg.model_dump() for msg in self._engine.messages]
            session_data = {
                "session_id": session_id,
                "name": arguments.name or session_id,
                "messages": messages,
                "cwd": str(context.cwd),
            }
            session_file = SESSION_DIR / f"{session_id}.json"
            session_file.write_text(json.dumps(session_data, indent=2, default=str))
            return ToolResult(output=f"Session saved: {session_id} ({len(messages)} messages)")

        elif action == "restore":
            if not arguments.session_id:
                return ToolResult(output="session_id is required for restore", is_error=True)
            session_file = SESSION_DIR / f"{arguments.session_id}.json"
            if not session_file.exists():
                return ToolResult(output=f"Session not found: {arguments.session_id}", is_error=True)
            if self._engine is None:
                return ToolResult(output="Engine not initialized", is_error=True)
            data = json.loads(session_file.read_text())
            from niaharness.engine.messages import ConversationMessage
            messages = [ConversationMessage.model_validate(m) for m in data.get("messages", [])]
            self._engine.load_messages(messages)
            return ToolResult(output=f"Session restored: {arguments.session_id} ({len(messages)} messages)")

        elif action == "list":
            sessions = []
            for f in sorted(SESSION_DIR.glob("nia-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    data = json.loads(f.read_text())
                    sessions.append({
                        "id": data.get("session_id", f.stem),
                        "name": data.get("name", ""),
                        "messages": len(data.get("messages", [])),
                        "modified": f.stat().st_mtime,
                    })
                except Exception:
                    continue
            if not sessions:
                return ToolResult(output="No saved sessions")
            lines = []
            for s in sessions[:10]:
                lines.append(f"{s['id']}: {s['name']} ({s['messages']} messages)")
            return ToolResult(output="\n".join(lines))

        elif action == "new":
            if self._engine is None:
                return ToolResult(output="Engine not initialized", is_error=True)
            self._engine.clear()
            return ToolResult(output="New session started (conversation cleared)")

        else:
            return ToolResult(
                output=f"Unknown action: {action}. Use: save, restore, list, new",
                is_error=True,
            )

    def is_read_only(self, arguments: NiaSessionInput) -> bool:
        return arguments.action in ("list",)
