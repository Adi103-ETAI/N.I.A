"""BashTool."""
import asyncio
from pathlib import Path
from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult
from .constants import BASH_TOOL_NAME, DEFAULT_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS, MAX_OUTPUT_LENGTH
from .prompt import get_bash_description
from .schema import BashToolInput
from .ui import format_command_output, format_error_message
from .utils import strip_empty_lines, truncate_output


class BashTool(BaseTool):
    name = BASH_TOOL_NAME
    description = get_bash_description()
    input_model = BashToolInput

    def is_read_only(self, arguments: BashToolInput) -> bool:
        return False

    async def execute(self, arguments: BashToolInput, context: ToolExecutionContext) -> ToolResult:
        timeout = arguments.timeout or DEFAULT_TIMEOUT_SECONDS
        timeout = min(timeout, MAX_TIMEOUT_SECONDS)
        
        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-c",
            arguments.command,
            cwd=str(context.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=float(timeout),
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ToolResult(
                output=format_error_message("timeout", f"after {timeout} seconds"),
                is_error=True,
            )

        stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
        
        output = format_command_output(stdout_text, stderr_text, process.returncode or 0)
        output = strip_empty_lines(output)
        output = truncate_output(output, MAX_OUTPUT_LENGTH)

        return ToolResult(
            output=output,
            is_error=(process.returncode != 0),
            metadata={"returncode": process.returncode},
        )
