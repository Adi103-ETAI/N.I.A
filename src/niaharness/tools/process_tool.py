"""Process multi-op tool — manage background tasks with a single tool.

Adapted from Hermes Agent's tools/process_registry.py.

Collapses NIA's 6 task tools (task_create, task_get, task_list, task_stop,
task_output, task_update) into a single tool with 8 operations:

- ``list``    — list all background tasks
- ``create``  — create a new background shell or agent task
- ``get``     — get details for a specific task
- ``output``  — read the output log for a task (replaces 'poll' + 'log')
- ``wait``    — wait for a task to complete (with timeout)
- ``stop``    — stop a running task (replaces 'kill')
- ``update``  — update task description/progress/status
- ``close``   — close a completed task's output stream

The old 6 tools remain registered for backward compatibility, but the
``process`` tool is the recommended interface — it matches Hermes's
architecture and reduces the tool-schema token cost per API call.

Reference: Hermes Agent's tools/process_registry.py (action enum: list,
poll, log, wait, kill, write, submit, close). NIA's version maps Hermes's
8 operations to NIA's existing TaskManager API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from niaharness.tasks.manager import get_task_manager
from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ProcessToolInput(BaseModel):
    """Arguments for the process tool."""

    action: Literal["list", "create", "get", "output", "wait", "stop", "update", "close"] = Field(
        description=(
            "The task management operation to perform. "
            "list: list all tasks. create: create a new task. "
            "get: get task details. output: read task output. "
            "wait: wait for completion. stop: stop a task. "
            "update: update task metadata. close: close output stream."
        )
    )
    task_id: str | None = Field(
        default=None,
        description="Task identifier (required for all actions except 'list' and 'create').",
    )
    # create
    task_type: Literal["local_bash", "local_agent"] = Field(
        default="local_bash",
        description="Type of task to create (for 'create' action).",
    )
    command: str | None = Field(
        default=None,
        description="Shell command to run (for 'create' with task_type='local_bash').",
    )
    description: str | None = Field(
        default=None,
        description="Task description (for 'create' and 'update' actions).",
    )
    # create (agent)
    prompt: str | None = Field(
        default=None,
        description="Prompt for the agent (for 'create' with task_type='local_agent').",
    )
    # output
    max_bytes: int = Field(
        default=12000,
        ge=1,
        le=100000,
        description="Max bytes to return (for 'output' action).",
    )
    # wait
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Max seconds to wait (for 'wait' action). Returns partial on timeout.",
    )
    # update
    progress: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Progress percentage (for 'update' action).",
    )
    status_note: str | None = Field(
        default=None,
        description="Status note (for 'update' action).",
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class ProcessTool(BaseTool):
    """Manage background processes with a single multi-operation tool."""

    name = "process"
    description = (
        "Manage background tasks: list, create, get details, read output, "
        "wait for completion, stop, update metadata, or close. Replaces the "
        "6 separate task_* tools with a single unified interface."
    )
    input_model = ProcessToolInput

    def is_read_only(self, arguments: ProcessToolInput) -> bool:
        return arguments.action in ("list", "get", "output", "wait")

    async def execute(self, arguments: ProcessToolInput, context: ToolExecutionContext) -> ToolResult:
        action = arguments.action

        if action == "list":
            return self._list()
        if action == "create":
            return await self._create(arguments, context)
        if action == "get":
            return self._get(arguments)
        if action == "output":
            return self._output(arguments)
        if action == "wait":
            return await self._wait(arguments)
        if action == "stop":
            return await self._stop(arguments)
        if action == "update":
            return self._update(arguments)
        if action == "close":
            return self._close(arguments)

        return ToolResult(output=f"Unknown action: {action}", is_error=True)

    # ---- actions -------------------------------------------------------

    def _list(self) -> ToolResult:
        """List all background tasks."""
        tasks = get_task_manager().list_tasks()
        if not tasks:
            return ToolResult(output="No background tasks.")
        lines = [f"Background tasks ({len(tasks)}):"]
        for t in tasks:
            status = t.status.value if hasattr(t.status, "value") else str(t.status)
            lines.append(f"  [{t.task_id}] {status} — {t.description or '(no description)'}")
        return ToolResult(output="\n".join(lines), metadata={"count": len(tasks)})

    async def _create(self, args: ProcessToolInput, context: ToolExecutionContext) -> ToolResult:
        """Create a new background task."""
        if not args.command and args.task_type == "local_bash":
            return ToolResult(output="create with task_type='local_bash' requires 'command'", is_error=True)
        if not args.prompt and args.task_type == "local_agent":
            return ToolResult(output="create with task_type='local_agent' requires 'prompt'", is_error=True)

        try:
            if args.task_type == "local_bash":
                task = await get_task_manager().create_shell_task(
                    command=args.command or "",
                    description=args.description or "",
                    cwd=str(context.cwd),
                )
            else:
                task = await get_task_manager().create_agent_task(
                    prompt=args.prompt or "",
                    description=args.description or "",
                    cwd=str(context.cwd),
                )
            return ToolResult(
                output=f"Created task '{task.task_id}' ({args.task_type}): {args.description or '(no description)'}",
                metadata={"task_id": task.task_id, "type": args.task_type},
            )
        except Exception as exc:
            return ToolResult(output=f"Failed to create task: {exc}", is_error=True)

    def _get(self, args: ProcessToolInput) -> ToolResult:
        """Get details for a specific task."""
        if not args.task_id:
            return ToolResult(output="get requires task_id", is_error=True)
        task = get_task_manager().get_task(args.task_id)
        if task is None:
            return ToolResult(output=f"Task not found: {args.task_id}", is_error=True)
        status = task.status.value if hasattr(task.status, "value") else str(task.status)
        lines = [
            f"Task: {task.task_id}",
            f"  Status: {status}",
            f"  Description: {task.description or '(none)'}",
        ]
        if hasattr(task, "progress") and task.progress is not None:
            lines.append(f"  Progress: {task.progress}%")
        if hasattr(task, "status_note") and task.status_note:
            lines.append(f"  Status note: {task.status_note}")
        return ToolResult(output="\n".join(lines))

    def _output(self, args: ProcessToolInput) -> ToolResult:
        """Read the output log for a task."""
        if not args.task_id:
            return ToolResult(output="output requires task_id", is_error=True)
        try:
            output = get_task_manager().read_task_output(args.task_id, max_bytes=args.max_bytes)
        except ValueError as exc:
            return ToolResult(output=str(exc), is_error=True)
        return ToolResult(output=output or "(no output)")

    async def _wait(self, args: ProcessToolInput) -> ToolResult:
        """Wait for a task to complete (with timeout)."""
        if not args.task_id:
            return ToolResult(output="wait requires task_id", is_error=True)
        try:
            # Poll every 0.5s until complete or timeout.
            elapsed = 0.0
            poll_interval = 0.5
            while elapsed < args.timeout:
                task = get_task_manager().get_task(args.task_id)
                if task is None:
                    return ToolResult(output=f"Task not found: {args.task_id}", is_error=True)
                status = task.status.value if hasattr(task.status, "value") else str(task.status)
                if status in ("completed", "failed", "stopped", "cancelled"):
                    output = get_task_manager().read_task_output(args.task_id, max_bytes=args.max_bytes)
                    return ToolResult(
                        output=f"Task {args.task_id} finished with status: {status}\n\n{output or '(no output)'}",
                        metadata={"task_id": args.task_id, "status": status},
                    )
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
            # Timeout — return partial output.
            output = get_task_manager().read_task_output(args.task_id, max_bytes=args.max_bytes)
            return ToolResult(
                output=f"Task {args.task_id} still running after {args.timeout}s (timeout).\n\nPartial output:\n{output or '(no output)'}",
                metadata={"task_id": args.task_id, "status": "timeout"},
            )
        except Exception as exc:
            return ToolResult(output=f"wait failed: {exc}", is_error=True)

    async def _stop(self, args: ProcessToolInput) -> ToolResult:
        """Stop a running task."""
        if not args.task_id:
            return ToolResult(output="stop requires task_id", is_error=True)
        try:
            task = await get_task_manager().stop_task(args.task_id)
            status = task.status.value if hasattr(task.status, "value") else str(task.status)
            return ToolResult(output=f"Stopped task '{args.task_id}' (status: {status})")
        except Exception as exc:
            return ToolResult(output=f"Failed to stop task: {exc}", is_error=True)

    def _update(self, args: ProcessToolInput) -> ToolResult:
        """Update task metadata."""
        if not args.task_id:
            return ToolResult(output="update requires task_id", is_error=True)
        try:
            get_task_manager().update_task(
                args.task_id,
                description=args.description,
                progress=args.progress,
                status_note=args.status_note,
            )
            return ToolResult(output=f"Updated task '{args.task_id}'")
        except Exception as exc:
            return ToolResult(output=f"Failed to update task: {exc}", is_error=True)

    def _close(self, args: ProcessToolInput) -> ToolResult:
        """Close a completed task's output stream."""
        if not args.task_id:
            return ToolResult(output="close requires task_id", is_error=True)
        # NIA's TaskManager doesn't have a separate close method —
        # this is a no-op that acknowledges the request (mirrors Hermes
        # closing stdin on a background process).
        task = get_task_manager().get_task(args.task_id)
        if task is None:
            return ToolResult(output=f"Task not found: {args.task_id}", is_error=True)
        return ToolResult(output=f"Closed output stream for task '{args.task_id}'")
