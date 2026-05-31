"""BashTool - Enhanced with streaming, background tasks, and timeout handling."""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import AsyncGenerator, NamedTuple

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

from .constants import (
    BASH_TOOL_NAME,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_OUTPUT_LENGTH,
    MAX_TIMEOUT_SECONDS,
    PROGRESS_THRESHOLD_MS,
)
from .prompt import get_bash_description
from .schema import BashToolInput
from .utils import (
    is_autobackgrounding_allowed,
    is_image_output,
    is_search_or_read_command,
    is_silent_command,
    split_command_with_operators,
    strip_empty_lines,
    truncate_output,
)


class BashProgress(NamedTuple):
    """Progress update for a running bash command."""

    output: str
    full_output: str
    elapsed_seconds: int
    total_lines: int
    total_bytes: int
    background_task_id: str | None = None
    timeout_ms: int | None = None


class BashOutput(NamedTuple):
    """Result of a bash command execution."""

    stdout: str
    stderr: str
    return_code: int
    interrupted: bool
    is_image: bool = False
    background_task_id: str | None = None


class BackgroundTask:
    """Represents a background bash task."""

    def __init__(
        self,
        task_id: str,
        command: str,
        description: str,
        process: asyncio.subprocess.Process,
        output_path: Path,
    ):
        self.task_id = task_id
        self.command = command
        self.description = description
        self.process = process
        self.output_path = output_path
        self.start_time = time.monotonic()
        self.output_file = open(output_path, "w", encoding="utf-8")

    def write_output(self, text: str) -> None:
        """Write output to the background task's output file."""
        self.output_file.write(text)
        self.output_file.flush()

    async def close(self) -> None:
        """Close the output file."""
        self.output_file.close()


class BashTool(BaseTool):
    """Enhanced bash tool with streaming, background tasks, and timeout handling."""

    name = BASH_TOOL_NAME
    description = get_bash_description()
    input_model = BashToolInput

    def __init__(self) -> None:
        self._background_tasks: dict[str, BackgroundTask] = {}
        self._background_counter = 0

    def is_read_only(self, arguments: BashToolInput) -> bool:
        """Check if the command is read-only (search/read/list)."""
        result = is_search_or_read_command(arguments.command)
        return result.is_search or result.is_read or result.is_list

    def get_search_read_info(self, arguments: BashToolInput) -> dict[str, bool]:
        """Get detailed search/read/list info for a command."""
        result = is_search_or_read_command(arguments.command)
        return {
            "is_search": result.is_search,
            "is_read": result.is_read,
            "is_list": result.is_list,
        }

    def get_user_facing_name(self, arguments: BashToolInput | None = None) -> str:
        """Get the user-facing name for this tool invocation."""
        return "Bash"

    def get_tool_use_summary(self, arguments: BashToolInput | None = None) -> str | None:
        """Get a summary for the tool use display."""
        if not arguments or not arguments.command:
            return None
        if arguments.description:
            return arguments.description
        return truncate_output(arguments.command, 200)

    def get_activity_description(self, arguments: BashToolInput | None = None) -> str:
        """Get a description of what the tool is doing."""
        if not arguments or not arguments.command:
            return "Running command"
        desc = arguments.description or truncate_output(arguments.command, 200)
        return f"Running {desc}"

    async def execute(
        self,
        arguments: BashToolInput,
        context: ToolExecutionContext,
        on_progress: callable | None = None,
    ) -> ToolResult:
        """Execute a bash command with streaming output and background task support."""
        timeout = arguments.timeout or DEFAULT_TIMEOUT_SECONDS
        timeout = min(timeout, MAX_TIMEOUT_SECONDS)
        timeout_ms = timeout * 1000

        # Handle background execution
        if arguments.run_in_background:
            return await self._execute_background(arguments, context)

        # Check if command should be auto-backgrounded
        if not is_autobackgrounding_allowed(arguments.command):
            pass  # Run in foreground

        # Run with streaming
        output = await self._execute_streaming(
            arguments, context, timeout_ms, on_progress
        )

        # Format and return result
        formatted = self._format_output(output, arguments)
        return formatted

    async def _execute_streaming(
        self,
        arguments: BashToolInput,
        context: ToolExecutionContext,
        timeout_ms: int,
        on_progress: callable | None = None,
    ) -> BashOutput:
        """Execute a command with streaming output."""
        start_time = time.monotonic()
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        interrupted = False

        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-c",
            arguments.command,
            cwd=str(context.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def read_stream(
            stream: asyncio.StreamReader | None, chunks: list[str], label: str
        ) -> None:
            """Read from a stream and accumulate chunks."""
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                chunks.append(text)
                if on_progress:
                    elapsed = int(time.monotonic() - start_time)
                    on_progress(
                        BashProgress(
                            output=text,
                            full_output="".join(chunks),
                            elapsed_seconds=elapsed,
                            total_lines=len(chunks),
                            total_bytes=sum(len(c) for c in chunks),
                        )
                    )

        try:
            # Read stdout and stderr concurrently
            await asyncio.wait_for(
                asyncio.gather(
                    read_stream(process.stdout, stdout_chunks, "stdout"),
                    read_stream(process.stderr, stderr_chunks, "stderr"),
                ),
                timeout=timeout_ms / 1000,
            )
            await process.wait()
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            interrupted = True
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            interrupted = True

        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)
        return_code = process.returncode or 0

        return BashOutput(
            stdout=stdout_text,
            stderr=stderr_text,
            return_code=return_code,
            interrupted=interrupted,
            is_image=is_image_output(stdout_text),
        )

    async def _execute_background(
        self,
        arguments: BashToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Execute a command in the background."""
        self._background_counter += 1
        task_id = f"bg-{self._background_counter}"

        output_dir = Path(tempfile.mkdtemp(prefix="bash_bg_"))
        output_path = output_dir / f"{task_id}.log"

        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-c",
            arguments.command,
            cwd=str(context.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        task = BackgroundTask(
            task_id=task_id,
            command=arguments.command,
            description=arguments.description or arguments.command,
            process=process,
            output_path=output_path,
        )
        self._background_tasks[task_id] = task

        # Start background output writer
        async def write_background_output() -> None:
            if process.stdout is None:
                return
            try:
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace")
                    task.write_output(text)
            except Exception:
                pass
            finally:
                await task.close()
                process.returncode = await process.wait()

        asyncio.create_task(write_background_output())

        return ToolResult(
            output=(
                f"Command running in background.\n"
                f"Task ID: {task_id}\n"
                f"Output path: {output_path}\n"
                f"Command: {arguments.command}"
            ),
            metadata={
                "background_task_id": task_id,
                "output_path": str(output_path),
            },
        )

    def get_background_task_status(self, task_id: str) -> dict | None:
        """Get the status of a background task."""
        task = self._background_tasks.get(task_id)
        if task is None:
            return None

        returncode = task.process.returncode
        is_running = returncode is None
        elapsed = time.monotonic() - task.start_time

        return {
            "task_id": task.task_id,
            "command": task.command,
            "description": task.description,
            "output_path": str(task.output_path),
            "is_running": is_running,
            "return_code": returncode,
            "elapsed_seconds": int(elapsed),
        }

    async def read_background_output(self, task_id: str) -> str | None:
        """Read the output of a background task."""
        task = self._background_tasks.get(task_id)
        if task is None:
            return None

        try:
            return task.output_path.read_text(encoding="utf-8")
        except Exception:
            return None

    async def cancel_background_task(self, task_id: str) -> bool:
        """Cancel a background task."""
        task = self._background_tasks.get(task_id)
        if task is None:
            return False

        if task.process.returncode is None:
            task.process.kill()
            await task.process.wait()
        await task.close()
        del self._background_tasks[task_id]
        return True

    def _format_output(
        self, output: BashOutput, arguments: BashToolInput
    ) -> ToolResult:
        """Format bash output into a ToolResult."""
        stdout = strip_empty_lines(output.stdout)
        stdout = truncate_output(stdout, MAX_OUTPUT_LENGTH)

        stderr_text = output.stderr.strip() if output.stderr else ""

        if output.interrupted:
            stderr_text += "\nCommand was aborted before completion" if stderr_text else "Command was aborted before completion"

        # Build combined output
        parts = []
        if stdout:
            parts.append(stdout)
        if stderr_text:
            parts.append(f"STDERR:\n{stderr_text}")

        if is_silent_command(arguments.command) and not stdout and not stderr_text:
            final_output = "Done (no output expected)"
        elif not stdout and not stderr_text:
            final_output = "(No output)"
        else:
            final_output = "\n\n".join(parts)

        is_error = output.return_code != 0 or output.interrupted

        return ToolResult(
            output=final_output,
            is_error=is_error,
            metadata={
                "return_code": output.return_code,
                "interrupted": output.interrupted,
                "is_image": output.is_image,
                "stdout_length": len(output.stdout),
                "stderr_length": len(output.stderr) if output.stderr else 0,
            },
        )
