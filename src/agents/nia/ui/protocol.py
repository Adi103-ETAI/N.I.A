"""N.I.A Protocol - JSON-lines communication models.

Reuses the same protocol as OpenHarness with N.I.A-specific extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel


class FrontendRequest(BaseModel):
    """Request from React frontend to Python backend."""
    type: str  # submit_line, permission_response, question_response, shutdown
    line: str = ""
    request_id: str = ""
    allowed: bool = False
    answer: str = ""


class TranscriptItem(BaseModel):
    """One row in the conversation transcript."""
    role: str  # system, user, assistant, tool, tool_result, log
    text: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    is_error: bool = False


class TaskSnapshot(BaseModel):
    """Background task status."""
    id: str
    type: str
    status: str
    description: str
    metadata: dict[str, Any] = {}


class BackendEvent(BaseModel):
    """Event from Python backend to React frontend."""
    type: str  # ready, state_snapshot, transcript_item, assistant_delta, etc.

    # State
    state: dict[str, Any] | None = None
    tasks: list[TaskSnapshot] | None = None
    commands: list[str] | None = None

    # Transcript
    item: TranscriptItem | None = None
    message: str | None = None

    # Tool
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    output: str | None = None
    is_error: bool = False

    # Modals
    modal: dict[str, Any] | None = None
    select_options: list[dict[str, str]] | None = None


def build_state_payload(
    provider_id: str = "",
    model: str = "",
    cwd: str = "",
    auth_status: str = "unknown",
    session_id: str = "",
    theme: str = "default",
    permission_mode: str = "default",
    mcp_count: int = 0,
    task_count: int = 0,
) -> dict[str, Any]:
    """Build a state snapshot payload."""
    return {
        "model": model,
        "provider": provider_id,
        "cwd": cwd,
        "auth_status": auth_status,
        "session_id": session_id,
        "theme": theme,
        "permission_mode": permission_mode,
        "mcp_count": mcp_count,
        "task_count": task_count,
        "vim_mode": False,
        "voice_mode": False,
        "fast_mode": False,
        "effort": "medium",
        "passes": 1,
    }
