"""Tool for N.I.A context awareness.

Exposes NIA's context system (time, user state, environment, session)
as a tool, allowing the brain to query situational awareness.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class NiaContextInput(BaseModel):
    """Arguments for NIA context operations."""

    action: str = Field(
        description="Operation: 'full' to get all context, 'environment' to get environment info, "
        "'time' to get time of day, 'session' to get session stats, "
        "'set_user_name' to store the user's name"
    )
    user_name: str | None = Field(default=None, description="User's name (for set_user_name)")


class NiaContextTool(BaseTool):
    """Access N.I.A's context awareness system.

    NIA tracks time of day, user state, environment details (platform, shell,
    project type, git branch), and session progress. Use this tool to query
    situational awareness.
    """

    name = "nia_context"
    description = (
        "Access N.I.A's context awareness. Actions: full (all context), "
        "environment (working dir, platform, project type, git branch), "
        "time (time of day), session (message count, tasks done), "
        "set_user_name (store the user's name for personalization)"
    )
    input_model = NiaContextInput

    def __init__(self, context: object | None = None) -> None:
        self._context = context

    def set_context(self, context: object) -> None:
        """Set the context instance (called during NIA initialization)."""
        self._context = context

    async def execute(self, arguments: NiaContextInput, context: ToolExecutionContext) -> ToolResult:
        if self._context is None:
            return ToolResult(output="Context system not initialized", is_error=True)

        action = arguments.action

        if action == "full":
            data = self._context.get_full_context()
            return ToolResult(output=json.dumps(data, indent=2))

        elif action == "environment":
            env = self._context._environment
            data = {
                "working_directory": env.working_directory,
                "platform": env.platform,
                "shell": env.shell,
                "python_version": env.python_version,
                "git_branch": env.git_branch,
                "project_type": env.project_type,
            }
            return ToolResult(output=json.dumps(data, indent=2))

        elif action == "time":
            return ToolResult(output=f"Time of day: {self._context.time_of_day.value}")

        elif action == "session":
            session = self._context._session
            data = {
                "message_count": session.message_count,
                "tasks_completed": session.tasks_completed,
                "tasks_pending": session.tasks_pending,
                "errors_encountered": session.errors_encountered,
                "user_state": self._context.user_state.value,
            }
            return ToolResult(output=json.dumps(data, indent=2))

        elif action == "set_user_name":
            if not arguments.user_name:
                return ToolResult(output="user_name is required", is_error=True)
            self._context.set_user_name(arguments.user_name)
            return ToolResult(output=f"Stored user name: {arguments.user_name}")

        else:
            return ToolResult(
                output=f"Unknown action: {action}. Use: full, environment, time, session, set_user_name",
                is_error=True,
            )

    def is_read_only(self, arguments: NiaContextInput) -> bool:
        return arguments.action != "set_user_name"
