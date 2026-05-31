"""Type definitions for FileReadTool."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FileReadToolInput(BaseModel):
    """Input schema for FileReadTool."""

    file_path: str = Field(description="The absolute path to the file to read")
    offset: int | None = Field(
        default=None,
        description="The line number to start reading from (1-indexed). Only provide if file is too large to read at once",
    )
    limit: int | None = Field(
        default=None,
        description="The number of lines to read. Only provide if file is too large to read at once",
    )


class FileReadToolOutput(BaseModel):
    """Output schema for FileReadTool."""

    file_path: str = Field(description="The path to the file that was read")
    content: str = Field(description="The content of the file with line numbers")
    num_lines: int = Field(description="Number of lines in the returned content")
    start_line: int = Field(description="The starting line number")
    total_lines: int = Field(description="Total number of lines in the file")
