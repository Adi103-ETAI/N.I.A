"""FileReadTool - Main implementation."""

from __future__ import annotations

from pathlib import Path

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

from .constants import FILE_NOT_FOUND_CWD_NOTE, FILE_READ_TOOL_NAME, MAX_LINES_TO_READ
from .prompt import get_file_read_description
from .schema import FileReadToolInput
from .ui import format_error_message
from .utils import add_line_numbers, detect_binary, format_file_info


class FileReadTool(BaseTool):
    """Read a file from the local filesystem with line numbers."""

    name = FILE_READ_TOOL_NAME
    description = get_file_read_description()
    input_model = FileReadToolInput

    def is_read_only(self, arguments: FileReadToolInput) -> bool:
        """FileReadTool is always read-only."""
        return True

    async def execute(
        self,
        arguments: FileReadToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """
        Execute the file read operation.
        
        Args:
            arguments: The tool input arguments
            context: The execution context
            
        Returns:
            ToolResult with the file content
        """
        # Resolve and validate the file path
        file_path = self._resolve_path(context.cwd, arguments.file_path)
        
        # Check if file exists
        if not file_path.exists():
            return ToolResult(
                output=format_error_message(
                    "not_found",
                    f"{FILE_NOT_FOUND_CWD_NOTE} {context.cwd}",
                ),
                is_error=True,
            )
        
        # Check if path is a directory
        if file_path.is_dir():
            return ToolResult(
                output=format_error_message("is_directory", str(file_path)),
                is_error=True,
            )
        
        # Read file as bytes first to detect binary
        try:
            raw_content = file_path.read_bytes()
        except Exception as e:
            return ToolResult(
                output=f"Failed to read file: {e}",
                is_error=True,
            )
        
        # Check for binary content
        if detect_binary(raw_content):
            return ToolResult(
                output=format_error_message("binary_file", str(file_path)),
                is_error=True,
            )
        
        # Decode content
        try:
            text_content = raw_content.decode("utf-8", errors="replace")
        except Exception as e:
            return ToolResult(
                output=format_error_message("decode_error", str(e)),
                is_error=True,
            )
        
        # Split into lines
        lines = text_content.splitlines()
        total_lines = len(lines)
        
        # Handle empty file
        if total_lines == 0:
            return ToolResult(
                output=f"{file_path} is empty",
                metadata={"file_path": str(file_path), "total_lines": 0},
            )
        
        # Determine offset and limit
        offset = arguments.offset if arguments.offset is not None else 1
        limit = arguments.limit if arguments.limit is not None else MAX_LINES_TO_READ
        
        # Convert 1-indexed offset to 0-indexed
        start_index = max(0, offset - 1)
        end_index = min(start_index + limit, total_lines)
        
        # Validate range
        if start_index >= total_lines:
            return ToolResult(
                output=format_error_message(
                    "invalid_range",
                    f"offset {offset} exceeds file length {total_lines}",
                ),
                is_error=True,
            )
        
        # Extract the requested lines
        selected_lines = lines[start_index:end_index]
        num_lines = len(selected_lines)
        
        # Add line numbers (1-indexed)
        numbered_content = add_line_numbers(
            "\n".join(selected_lines),
            start_line=offset,
        )
        
        # Build output with file info header
        file_info = format_file_info(str(file_path), total_lines, offset, num_lines)
        output = f"{file_info}\n\n{numbered_content}"
        
        return ToolResult(
            output=output,
            metadata={
                "file_path": str(file_path),
                "total_lines": total_lines,
                "start_line": offset,
                "num_lines": num_lines,
            },
        )
    
    @staticmethod
    def _resolve_path(base: Path, candidate: str) -> Path:
        """
        Resolve a file path relative to the base directory.
        
        Args:
            base: The base directory (usually cwd)
            candidate: The file path to resolve
            
        Returns:
            Resolved absolute path
        """
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = base / path
        return path.resolve()
