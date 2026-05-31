"""Filesystem globbing tool with enhanced pattern matching."""

from __future__ import annotations

import fnmatch
import os
import time
from pathlib import Path

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class GlobToolInput(BaseModel):
    """Arguments for the glob tool."""

    pattern: str = Field(description="The glob pattern to match files against")
    path: str | None = Field(
        default=None,
        description=(
            "The directory to search in. If not specified, the current working "
            "directory will be used."
        ),
    )


class GlobResult:
    """Result of a glob operation."""

    def __init__(
        self,
        filenames: list[str],
        duration_ms: int,
        num_files: int,
        truncated: bool,
    ):
        self.filenames = filenames
        self.duration_ms = duration_ms
        self.num_files = num_files
        self.truncated = truncated


class GlobTool(BaseTool):
    """List files matching a glob pattern with enhanced matching."""

    name = "glob"
    description = "List files matching a glob pattern."
    input_model = GlobToolInput

    def is_read_only(self, arguments: GlobToolInput) -> bool:
        return True

    def get_user_facing_name(self, arguments: GlobToolInput | None = None) -> str:
        return "Glob"

    async def execute(
        self, arguments: GlobToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        start = time.monotonic()
        root = self._resolve_path(context.cwd, arguments.path)

        if not root.exists():
            return ToolResult(
                output=f"Directory does not exist: {arguments.path or '.'}",
                is_error=True,
            )

        if not root.is_dir():
            return ToolResult(
                output=f"Path is not a directory: {arguments.path or '.'}",
                is_error=True,
            )

        # Use Python's pathlib glob with proper pattern handling
        limit = 100
        matches: list[str] = []
        truncated = False

        try:
            # Handle brace expansion patterns like *.{ts,tsx}
            expanded_patterns = self._expand_braces(arguments.pattern)

            for expanded_pattern in expanded_patterns:
                for path in root.glob(expanded_pattern):
                    if len(matches) >= limit:
                        truncated = True
                        break
                    if path.is_file():
                        rel_path = str(path.relative_to(root))
                        if rel_path not in matches:
                            matches.append(rel_path)
                if truncated:
                    break

            matches.sort()

        except Exception as e:
            return ToolResult(
                output=f"Glob error: {e}",
                is_error=True,
            )

        duration_ms = int((time.monotonic() - start) * 1000)

        if not matches:
            return ToolResult(
                output="No files found",
                metadata={
                    "duration_ms": duration_ms,
                    "num_files": 0,
                    "truncated": False,
                },
            )

        output = "\n".join(matches)
        if truncated:
            output += "\n\n(Results are truncated. Consider using a more specific path or pattern.)"

        return ToolResult(
            output=output,
            metadata={
                "duration_ms": duration_ms,
                "num_files": len(matches),
                "truncated": truncated,
            },
        )

    @staticmethod
    def _expand_braces(pattern: str) -> list[str]:
        """
        Expand brace patterns like *.{ts,tsx} into separate patterns.

        Args:
            pattern: The glob pattern with optional braces

        Returns:
            List of expanded patterns
        """
        # Find brace sections
        brace_start = pattern.find("{")
        if brace_start == -1:
            return [pattern]

        brace_end = pattern.find("}", brace_start)
        if brace_end == -1:
            return [pattern]

        prefix = pattern[:brace_start]
        suffix = pattern[brace_end + 1 :]
        options = pattern[brace_start + 1 : brace_end].split(",")

        expanded = []
        for option in options:
            option = option.strip()
            if option:
                expanded.append(f"{prefix}{option}{suffix}")

        return expanded if expanded else [pattern]

    @staticmethod
    def _resolve_path(base: Path, candidate: str | None) -> Path:
        if not candidate:
            return base
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = base / path
        return path.resolve()
