"""N.I.A Dispatcher - Routes tasks to OpenHarness (the hands).

This is the bridge between N.I.A's brain and OpenHarness's execution.
N.I.A decides WHAT to do, OpenHarness DOES it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a dispatched task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A task to be executed by OpenHarness."""
    id: str
    description: str
    tool_name: str  # Which OpenHarness tool to use
    arguments: dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    dependencies: list[str] = field(default_factory=list)


@dataclass
class DispatchResult:
    """Result of dispatching tasks."""
    tasks_dispatched: int
    tasks_succeeded: int
    tasks_failed: int
    results: list[Any]
    errors: list[str]


class Dispatcher:
    """Dispatches tasks from N.I.A to OpenHarness.

    N.I.A's brain decides WHAT needs to happen.
    The dispatcher translates that into OpenHarness tool calls.
    OpenHarness executes the actual work.

    This is the "nervous system" connecting head to hands.
    """

    def __init__(self) -> None:
        self._task_queue: list[Task] = []
        self._active_tasks: dict[str, Task] = {}
        self._completed_tasks: list[Task] = []
        self._tool_executor: Callable | None = None
        self._task_counter: int = 0

    def set_tool_executor(self, executor: Callable) -> None:
        """Set the function that actually executes OpenHarness tools."""
        self._tool_executor = executor

    def dispatch(self, description: str, tool_name: str, arguments: dict[str, Any]) -> Task:
        """Create and queue a single task."""
        self._task_counter += 1
        task = Task(
            id=f"nia-task-{self._task_counter}",
            description=description,
            tool_name=tool_name,
            arguments=arguments,
        )
        self._task_queue.append(task)
        logger.info(f"Dispatched task: {task.id} - {description}")
        return task

    def dispatch_batch(self, tasks: list[tuple[str, str, dict[str, Any]]]) -> list[Task]:
        """Dispatch multiple tasks at once."""
        dispatched = []
        for description, tool_name, arguments in tasks:
            task = self.dispatch(description, tool_name, arguments)
            dispatched.append(task)
        return dispatched

    async def execute_pending(self) -> DispatchResult:
        """Execute all pending tasks."""
        results = []
        errors = []
        succeeded = 0
        failed = 0

        while self._task_queue:
            task = self._task_queue.pop(0)
            task.status = TaskStatus.RUNNING
            self._active_tasks[task.id] = task

            try:
                result = await self._execute_task(task)
                task.status = TaskStatus.COMPLETED
                task.result = result
                results.append(result)
                succeeded += 1
                logger.info(f"Task {task.id} completed successfully")
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                errors.append(f"Task {task.id} failed: {e}")
                failed += 1
                logger.error(f"Task {task.id} failed: {e}")
            finally:
                del self._active_tasks[task.id]
                self._completed_tasks.append(task)

        return DispatchResult(
            tasks_dispatched=succeeded + failed,
            tasks_succeeded=succeeded,
            tasks_failed=failed,
            results=results,
            errors=errors,
        )

    async def _execute_task(self, task: Task) -> Any:
        """Execute a single task using OpenHarness tools."""
        if self._tool_executor is None:
            raise RuntimeError("No tool executor configured")

        # Build the tool call for OpenHarness
        tool_call = {
            "tool": task.tool_name,
            "arguments": task.arguments,
        }

        logger.info(f"Executing {task.tool_name} with args: {task.arguments}")

        # Call the OpenHarness tool executor
        result = await self._tool_executor(tool_call)
        return result

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        for i, task in enumerate(self._task_queue):
            if task.id == task_id:
                task.status = TaskStatus.CANCELLED
                self._task_queue.pop(i)
                return True
        return False

    def get_status(self) -> dict[str, Any]:
        """Get dispatcher status."""
        return {
            "queued": len(self._task_queue),
            "active": len(self._active_tasks),
            "completed": len(self._completed_tasks),
            "total_dispatched": self._task_counter,
        }


# Tool name mappings from N.I.A intents to OpenHarness tools
INTENT_TO_TOOL: dict[str, str] = {
    "create": "write_file",
    "modify": "file_edit",
    "delete": "file_edit",  # With empty new_string
    "debug": "bash",
    "test": "bash",
    "run": "bash",
    "explain": "file_read",
    "search": "grep",
    "analyze": "file_read",
}


def intent_to_tool_call(intent: str, entities: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Convert an N.I.A intent to an OpenHarness tool call."""
    tool_name = INTENT_TO_TOOL.get(intent, "bash")
    arguments: dict[str, Any] = {}

    if intent in ("create", "modify", "delete"):
        if "files" in entities:
            arguments["file_path"] = entities["files"][0]
        if intent == "create":
            arguments["content"] = ""  # Will be filled by brain
        elif intent == "modify":
            arguments["old_string"] = ""  # Will be filled by brain
            arguments["new_string"] = ""  # Will be filled by brain
    elif intent in ("debug", "test", "run"):
        arguments["command"] = entities.get("command", "echo 'No command specified'")
    elif intent == "search":
        arguments["pattern"] = entities.get("pattern", ".*")
    elif intent == "explain":
        if "files" in entities:
            arguments["file_path"] = entities["files"][0]

    return tool_name, arguments
