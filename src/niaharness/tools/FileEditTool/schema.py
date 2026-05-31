"""Type definitions for FileEditTool."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FileEditToolInput(BaseModel):
    """Input schema for FileEditTool."""

    file_path: str = Field(description="The absolute path to the file to modify")
    old_string: str = Field(description="The text to replace")
    new_string: str = Field(
        description="The text to replace it with (must be different from old_string)"
    )
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences of old_string (default false)",
    )


class HunkInfo(BaseModel):
    """Information about a diff hunk."""

    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    lines: list[str]


class GitDiffInfo(BaseModel):
    """Git diff information."""

    filename: str
    status: str  # 'modified' or 'added'
    additions: int
    deletions: int
    changes: int
    patch: str
    repository: str | None = None


class FileEditToolOutput(BaseModel):
    """Output schema for FileEditTool."""

    file_path: str = Field(description="The file path that was edited")
    old_string: str = Field(description="The original string that was replaced")
    new_string: str = Field(description="The new string that replaced it")
    original_file: str = Field(description="The original file contents before editing")
    structured_patch: list[HunkInfo] = Field(
        description="Diff patch showing the changes"
    )
    user_modified: bool = Field(
        description="Whether the user modified the proposed changes"
    )
    replace_all: bool = Field(description="Whether all occurrences were replaced")
    git_diff: GitDiffInfo | None = Field(default=None)
