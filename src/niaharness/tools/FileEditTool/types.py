"""FileEditTool type definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FileEditToolInput(BaseModel):
    """Input schema for FileEditTool."""

    file_path: str = Field(description="The absolute path to the file to modify")
    old_string: str = Field(description="The text to replace")
    new_string: str = Field(description="The text to replace it with (must be different)")
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences of old_string (default: false)",
    )


class HunkInfo(BaseModel):
    """Information about a diff hunk."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


class FileEditToolOutput(BaseModel):
    """Output schema for FileEditTool."""

    file_path: str
    old_string: str
    new_string: str
    original_file: str
    structured_patch: str
    user_modified: bool = False
    replace_all: bool = False
