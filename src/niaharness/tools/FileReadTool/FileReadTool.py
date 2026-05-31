"""FileReadTool - Enhanced with image/PDF/notebook support and encoding detection."""

from __future__ import annotations

import json
from pathlib import Path

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

from .constants import FILE_NOT_FOUND_CWD_NOTE, FILE_READ_TOOL_NAME, MAX_LINES_TO_READ
from .prompt import get_file_read_description
from .schema import FileReadToolInput
from .ui import format_error_message
from .schema import ImageOutput, NotebookOutput, PDFOutput
from .utils import (
    add_line_numbers,
    detect_binary,
    detect_encoding,
    format_file_info,
    format_file_size,
    get_media_type,
    is_binary_file,
    is_image_file,
    is_notebook_file,
    is_pdf_file,
    parse_pdf_page_range,
    read_image_as_base64,
    read_notebook,
    read_pdf_as_base64,
)


class FileReadTool(BaseTool):
    """Read a file from the local filesystem with line numbers.

    Supports text files, images (PNG, JPG, GIF, WebP), PDFs,
    and Jupyter notebooks (.ipynb).
    """

    name = FILE_READ_TOOL_NAME
    description = get_file_read_description()
    input_model = FileReadToolInput

    def is_read_only(self, arguments: FileReadToolInput) -> bool:
        """FileReadTool is always read-only."""
        return True

    def get_user_facing_name(self, arguments: FileReadToolInput | None = None) -> str:
        """Get the user-facing name."""
        return "Read"

    def get_tool_use_summary(self, arguments: FileReadToolInput | None = None) -> str | None:
        """Get a summary for display."""
        if not arguments:
            return None
        return arguments.file_path

    async def execute(
        self,
        arguments: FileReadToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Execute the file read operation."""
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

        # Route to appropriate handler based on file type
        if is_image_file(str(file_path)):
            return await self._read_image(file_path)
        elif is_pdf_file(str(file_path)):
            return await self._read_pdf(file_path, arguments.pages)
        elif is_notebook_file(str(file_path)):
            return await self._read_notebook(file_path)
        elif is_binary_file(str(file_path)):
            return ToolResult(
                output=format_error_message(
                    "binary_file",
                    f"Cannot read binary file: {file_path}",
                ),
                is_error=True,
            )
        else:
            return await self._read_text(file_path, arguments, context)

    async def _read_image(self, file_path: Path) -> ToolResult:
        """Read an image file and return base64-encoded data."""
        try:
            b64_data, media_type, file_size = read_image_as_base64(str(file_path))
            output = (
                f"Image file: {file_path.name}\n"
                f"Type: {media_type}\n"
                f"Size: {format_file_size(file_size)}\n"
                f"Base64 length: {len(b64_data)} characters"
            )
            return ToolResult(
                output=output,
                metadata={
                    "type": "image",
                    "file_path": str(file_path),
                    "base64": b64_data,
                    "media_type": media_type,
                    "original_size": file_size,
                },
            )
        except Exception as e:
            return ToolResult(
                output=f"Failed to read image: {e}",
                is_error=True,
            )

    async def _read_pdf(
        self, file_path: Path, pages: str | None = None
    ) -> ToolResult:
        """Read a PDF file and return base64-encoded data."""
        try:
            b64_data, file_size = read_pdf_as_base64(str(file_path))

            # Parse page range if provided
            page_info = ""
            if pages:
                parsed = parse_pdf_page_range(pages)
                if parsed is None:
                    return ToolResult(
                        output=format_error_message(
                            "invalid_pages",
                            f'Invalid pages parameter: "{pages}". Use formats like "1-5", "3", or "10-20".',
                        ),
                        is_error=True,
                    )
                first, last = parsed
                if last - first + 1 > 50:
                    return ToolResult(
                        output=format_error_message(
                            "too_many_pages",
                            "Maximum 50 pages per request.",
                        ),
                        is_error=True,
                    )
                page_info = f"\nPage range: {first}-{last}"

            output = (
                f"PDF file: {file_path.name}\n"
                f"Size: {format_file_size(file_size)}\n"
                f"Base64 length: {len(b64_data)} characters"
                f"{page_info}"
            )
            return ToolResult(
                output=output,
                metadata={
                    "type": "pdf",
                    "file_path": str(file_path),
                    "base64": b64_data,
                    "original_size": file_size,
                },
            )
        except Exception as e:
            return ToolResult(
                output=f"Failed to read PDF: {e}",
                is_error=True,
            )

    async def _read_notebook(self, file_path: Path) -> ToolResult:
        """Read a Jupyter notebook and return cells."""
        try:
            cells = read_notebook(str(file_path))

            # Format cells for display
            output_parts = [f"Jupyter Notebook: {file_path.name}"]
            output_parts.append(f"Total cells: {len(cells)}")
            output_parts.append("")

            for i, cell in enumerate(cells[:100], start=1):  # Limit display
                cell_type = cell.get("cell_type", "unknown")
                source = cell.get("source", "")
                output_parts.append(f"--- Cell {i} ({cell_type}) ---")
                if source:
                    output_parts.append(source)
                output_parts.append("")

            if len(cells) > 100:
                output_parts.append(f"... ({len(cells) - 100} more cells)")

            return ToolResult(
                output="\n".join(output_parts),
                metadata={
                    "type": "notebook",
                    "file_path": str(file_path),
                    "cells": cells,
                },
            )
        except Exception as e:
            return ToolResult(
                output=f"Failed to read notebook: {e}",
                is_error=True,
            )

    async def _read_text(
        self,
        file_path: Path,
        arguments: FileReadToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Read a text file with line numbers."""
        try:
            raw_content = file_path.read_bytes()
        except Exception as e:
            return ToolResult(
                output=f"Failed to read file: {e}",
                is_error=True,
            )

        # Detect binary content
        if detect_binary(raw_content):
            return ToolResult(
                output=format_error_message("binary_file", str(file_path)),
                is_error=True,
            )

        # Detect encoding
        encoding = detect_encoding(raw_content)

        # Decode content
        try:
            text_content = raw_content.decode(encoding, errors="replace")
        except Exception as e:
            return ToolResult(
                output=format_error_message("decode_error", str(e)),
                is_error=True,
            )

        # Normalize line endings
        text_content = text_content.replace("\r\n", "\n").replace("\r", "\n")

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
                "encoding": encoding,
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
