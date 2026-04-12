"""FileEditTool - Enhanced with sophisticated validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult
from openharness.tools.shared_state import get_read_state_tracker

from .constants import (
    FILE_EDIT_TOOL_NAME,
    FILE_NOT_FOUND_CWD_NOTE,
    FILE_UNEXPECTEDLY_MODIFIED_ERROR,
    MAX_EDIT_FILE_SIZE,
    ERROR_JUPYTER_NOTEBOOK,
    ERROR_NOT_READ_FIRST,
    ERROR_MODIFIED_SINCE_READ,
)
from .prompt import get_edit_tool_description
from .schema import FileEditToolInput
from .ui import format_error_message, format_success_message
from .utils import (
    apply_edit_to_file,
    count_matches,
    find_actual_string,
    get_patch_for_edit,
    preserve_quote_style,
    get_file_modification_time,
    read_file_for_edit,
    suggest_similar_file,
)


class FileEditTool(BaseTool):
    """Replace text in an existing file using exact string matching with advanced validation."""

    name = FILE_EDIT_TOOL_NAME
    description = get_edit_tool_description()
    input_model = FileEditToolInput

    def __init__(self):
        """Initialize with shared read state tracker."""
        self.read_state_tracker = get_read_state_tracker()

    async def execute(
        self,
        arguments: FileEditToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Execute the file edit operation with full validation."""
        
        file_path = self._resolve_path(context.cwd, arguments.file_path)
        
        # Comprehensive validation
        validation_error = await self._validate_edit(file_path, arguments, context)
        if validation_error:
            return validation_error
        
        # Read current file state
        original_content, file_exists, encoding = read_file_for_edit(str(file_path))
        
        if file_exists:
            # Race condition protection: re-check staleness
            last_write_time = get_file_modification_time(str(file_path))
            read_state = self.read_state_tracker.get_state(str(file_path))
            
            if read_state and last_write_time > read_state.timestamp:
                if original_content != read_state.content:
                    return ToolResult(
                        output=FILE_UNEXPECTEDLY_MODIFIED_ERROR,
                        is_error=True,
                    )
        
        # Find actual string with quote normalization
        actual_old_string = find_actual_string(original_content, arguments.old_string) or arguments.old_string
        
        # Preserve quote style from file
        actual_new_string = preserve_quote_style(
            arguments.old_string,
            actual_old_string,
            arguments.new_string,
        )
        
        # Generate patch and apply edit
        try:
            patch, updated_content = get_patch_for_edit(
                str(file_path),
                original_content,
                actual_old_string,
                actual_new_string,
                arguments.replace_all,
            )
        except ValueError as e:
            return ToolResult(output=str(e), is_error=True)
        
        # Write file
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if encoding == "utf-16-le":
                file_path.write_bytes(updated_content.encode(encoding))
            else:
                file_path.write_text(updated_content, encoding=encoding)
        except Exception as e:
            return ToolResult(output=f"Failed to write file: {e}", is_error=True)
        
        # Update read state
        new_timestamp = get_file_modification_time(str(file_path))
        self.read_state_tracker.update_after_edit(str(file_path), updated_content, new_timestamp)
        
        # Build output
        match_count = count_matches(original_content, actual_old_string)
        replacements = match_count if arguments.replace_all else 1
        success_message = format_success_message(str(file_path), replacements)
        
        return ToolResult(
            output=f"{success_message}\n\nChanges:\n{patch}",
            metadata={"file_path": str(file_path), "replacements": replacements, "patch": patch},
        )
    
    async def _validate_edit(
        self,
        file_path: Path,
        arguments: FileEditToolInput,
        context: ToolExecutionContext,
    ) -> Optional[ToolResult]:
        """Comprehensive validation. Returns error or None."""
        
        # 1. Check old_string != new_string
        if arguments.old_string == arguments.new_string:
            return ToolResult(
                output=format_error_message("no_changes", "old_string and new_string are the same"),
                is_error=True,
            )
        
        # 2. Check for Jupyter notebooks
        if str(file_path).endswith(".ipynb"):
            return ToolResult(output=ERROR_JUPYTER_NOTEBOOK, is_error=True)
        
        # 3. Check file size
        if file_path.exists():
            file_size = file_path.stat().st_size
            if file_size > MAX_EDIT_FILE_SIZE:
                return ToolResult(
                    output=format_error_message(
                        "too_large",
                        f"File is {file_size/(1024*1024):.1f} MB, max is {MAX_EDIT_FILE_SIZE/(1024*1024):.1f} MB",
                    ),
                    is_error=True,
                )
        
        # 4. Handle file creation
        if arguments.old_string == "":
            if not file_path.exists():
                return None
            content = file_path.read_text(encoding="utf-8")
            if content.strip():
                return ToolResult(output="Cannot create file - already exists with content", is_error=True)
            return None
        
        # 5. File must exist
        if not file_path.exists():
            similar = suggest_similar_file(str(file_path))
            msg = f"{FILE_NOT_FOUND_CWD_NOTE} {context.cwd}"
            if similar:
                msg += f"\nDid you mean: {similar}?"
            return ToolResult(output=format_error_message("not_found", msg), is_error=True)
        
        # 6. Must read before edit
        read_state = self.read_state_tracker.get_state(str(file_path))
        if not read_state:
            return ToolResult(output=ERROR_NOT_READ_FIRST, is_error=True)
        
        # 7. No partial reads
        if read_state.is_partial_view:
            return ToolResult(
                output="File was partially read. Read full file before editing.",
                is_error=True,
            )
        
        # 8. Staleness check
        last_write_time = get_file_modification_time(str(file_path))
        if last_write_time > read_state.timestamp:
            current_content, _, _ = read_file_for_edit(str(file_path))
            if current_content != read_state.content:
                return ToolResult(output=ERROR_MODIFIED_SINCE_READ, is_error=True)
        
        # 9. String must exist
        current_content, _, _ = read_file_for_edit(str(file_path))
        actual_old_string = find_actual_string(current_content, arguments.old_string)
        if actual_old_string is None:
            return ToolResult(
                output=format_error_message("string_not_found", f"Not found:\n{arguments.old_string[:200]}"),
                is_error=True,
            )
        
        # 10. Multiple matches check
        match_count = count_matches(current_content, actual_old_string)
        if match_count > 1 and not arguments.replace_all:
            return ToolResult(
                output=format_error_message(
                    "multiple_matches",
                    f"Found {match_count} matches. Use replace_all=true or add context to make unique",
                ),
                is_error=True,
            )
        
        return None
    
    @staticmethod
    def _resolve_path(base: Path, candidate: str) -> Path:
        path = Path(candidate).expanduser()
        return (base / path).resolve() if not path.is_absolute() else path.resolve()
    
    def is_read_only(self, arguments: BaseModel) -> bool:
        return False
