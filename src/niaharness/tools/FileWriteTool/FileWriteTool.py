"""FileWriteTool."""
import os
from pathlib import Path
from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult
from niaharness.tools.shared_state import get_read_state_tracker
from .constants import FILE_WRITE_TOOL_NAME
from .prompt import get_write_tool_description
from .schema import FileWriteToolInput
from .ui import format_success_message
from .utils import count_lines, get_operation_type


class FileWriteTool(BaseTool):
    name = FILE_WRITE_TOOL_NAME
    description = get_write_tool_description()
    input_model = FileWriteToolInput

    def is_read_only(self, arguments: FileWriteToolInput) -> bool:
        return False

    async def execute(self, arguments: FileWriteToolInput, context: ToolExecutionContext) -> ToolResult:
        file_path = self._resolve_path(context.cwd, arguments.file_path)
        file_existed = file_path.exists()

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(arguments.content, encoding="utf-8")
        except Exception as e:
            return ToolResult(output=f"Failed: {e}", is_error=True)

        # Mark file as "read" so subsequent edit_file calls don't require a
        # separate read first (the agent just wrote the content, so it knows
        # the state).
        try:
            get_read_state_tracker().update_after_read(
                str(file_path),
                arguments.content,
            )
        except Exception:
            pass

        operation = get_operation_type(file_existed)
        num_lines = count_lines(arguments.content)

        return ToolResult(
            output=format_success_message(operation, str(file_path), num_lines),
            metadata={"file_path": str(file_path), "operation": operation},
        )

    @staticmethod
    def _resolve_path(base: Path, candidate: str) -> Path:
        path = Path(candidate).expanduser()
        return (base / path).resolve() if not path.is_absolute() else path.resolve()
