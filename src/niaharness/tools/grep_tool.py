"""Content search tool with ripgrep integration and pagination."""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class GrepToolInput(BaseModel):
    """Arguments for the grep tool."""

    pattern: str = Field(description="Regular expression to search for")
    path: str | None = Field(
        default=None,
        description="File or directory to search in. Defaults to current working directory.",
    )
    glob: str | None = Field(
        default=None,
        description='Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}")',
    )
    output_mode: str = Field(
        default="files_with_matches",
        description='Output mode: "content", "files_with_matches", or "count"',
    )
    context_before: int | None = Field(
        default=None,
        alias="-B",
        description="Number of lines to show before each match",
    )
    context_after: int | None = Field(
        default=None,
        alias="-A",
        description="Number of lines to show after each match",
    )
    context: int | None = Field(
        default=None,
        description="Number of lines to show before and after each match",
    )
    show_line_numbers: bool = Field(
        default=True,
        alias="-n",
        description="Show line numbers in output",
    )
    case_insensitive: bool = Field(
        default=False,
        alias="-i",
        description="Case insensitive search",
    )
    type: str | None = Field(
        default=None,
        description="File type to search (e.g., js, py, rust, go, java)",
    )
    head_limit: int | None = Field(
        default=250,
        description="Limit output to first N lines/entries. Pass 0 for unlimited.",
    )
    offset: int = Field(
        default=0,
        description="Skip first N lines/entries before applying head_limit",
    )
    multiline: bool = Field(
        default=False,
        description="Enable multiline mode where patterns can span lines",
    )


class GrepResult(NamedTuple):
    """Result of a grep operation."""

    num_files: int
    filenames: list[str]
    content: str | None = None
    num_lines: int | None = None
    num_matches: int | None = None
    applied_limit: int | None = None
    applied_offset: int | None = None


class GrepTool(BaseTool):
    """Search file contents using ripgrep when available, with Python fallback."""

    name = "grep"
    description = "Search file contents with a regular expression (ripgrep when available)."
    input_model = GrepToolInput

    def is_read_only(self, arguments: GrepToolInput) -> bool:
        return True

    def get_user_facing_name(self, arguments: GrepToolInput | None = None) -> str:
        return "Search"

    async def execute(
        self, arguments: GrepToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        root = self._resolve_path(context.cwd, arguments.path)

        # Try ripgrep first
        if shutil.which("rg"):
            return await self._ripgrep_search(arguments, root)

        # Fall back to Python
        return await self._python_search(arguments, root)

    async def _ripgrep_search(
        self, arguments: GrepToolInput, root: Path
    ) -> ToolResult:
        """Search using ripgrep subprocess."""
        args = ["rg", "--hidden"]

        # Exclude VCS directories
        for vcs_dir in [".git", ".svn", ".hg", ".bzr", ".jj", ".sl"]:
            args.extend(["--glob", f"!{vcs_dir}"])

        # Limit line length
        args.extend(["--max-columns", "500"])

        # Multiline mode
        if arguments.multiline:
            args.extend(["-U", "--multiline-dotall"])

        # Case insensitive
        if arguments.case_insensitive:
            args.append("-i")

        # Output mode
        if arguments.output_mode == "files_with_matches":
            args.append("-l")
        elif arguments.output_mode == "count":
            args.append("-c")

        # Line numbers
        if arguments.show_line_numbers and arguments.output_mode == "content":
            args.append("-n")

        # Context flags
        if arguments.output_mode == "content":
            if arguments.context is not None:
                args.extend(["-C", str(arguments.context)])
            else:
                if arguments.context_before is not None:
                    args.extend(["-B", str(arguments.context_before)])
                if arguments.context_after is not None:
                    args.extend(["-A", str(arguments.context_after)])

        # Pattern (use -e if starts with dash)
        if arguments.pattern.startswith("-"):
            args.extend(["-e", arguments.pattern])
        else:
            args.append(arguments.pattern)

        # Type filter
        if arguments.type:
            args.extend(["--type", arguments.type])

        # Glob filter
        if arguments.glob:
            for glob_pattern in arguments.glob.split(","):
                glob_pattern = glob_pattern.strip()
                if glob_pattern:
                    args.extend(["--glob", glob_pattern])

        # Execute ripgrep
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                str(root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30.0,
            )

            output = stdout.decode("utf-8", errors="replace")
            lines = [line for line in output.strip().split("\n") if line]

            # Apply pagination
            result = self._apply_pagination(lines, arguments, root)
            return self._format_result(result, arguments)

        except asyncio.TimeoutError:
            return ToolResult(
                output="Search timed out after 30 seconds",
                is_error=True,
            )
        except FileNotFoundError:
            # Ripgrep not found, fall back to Python
            return await self._python_search(arguments, root)
        except Exception as e:
            return ToolResult(
                output=f"ripgrep error: {e}",
                is_error=True,
            )

    async def _python_search(
        self, arguments: GrepToolInput, root: Path
    ) -> ToolResult:
        """Pure Python fallback search."""
        flags = 0 if arguments.case_insensitive else re.IGNORECASE
        try:
            pattern = re.compile(arguments.pattern, flags)
        except re.error as e:
            return ToolResult(
                output=f"Invalid regex pattern: {e}",
                is_error=True,
            )

        matches: list[str] = []
        glob_pattern = arguments.glob or "**/*"

        for path in sorted(root.glob(glob_pattern)):
            if not path.is_file():
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw:
                continue

            text = raw.decode("utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    rel_path = str(path.relative_to(root))
                    if arguments.output_mode == "files_with_matches":
                        if rel_path not in matches:
                            matches.append(rel_path)
                    elif arguments.output_mode == "count":
                        matches.append(f"{rel_path}:1")
                    else:
                        matches.append(f"{rel_path}:{line_no}:{line}")

        # Apply pagination
        result = self._apply_python_pagination(matches, arguments)
        return self._format_result(result, arguments)

    def _apply_pagination(
        self,
        lines: list[str],
        arguments: GrepToolInput,
        root: Path,
    ) -> GrepResult:
        """Apply pagination to ripgrep results."""
        from .utils import to_relative_path

        offset = arguments.offset
        limit = arguments.head_limit if arguments.head_limit is not None else 250

        if limit == 0:
            # Unlimited
            paginated = lines[offset:]
            applied_limit = None
        else:
            paginated = lines[offset : offset + limit]
            was_truncated = len(lines) - offset > limit
            applied_limit = limit if was_truncated else None

        if arguments.output_mode == "content":
            # Convert absolute paths to relative
            final_lines = []
            for line in paginated:
                colon_index = line.find(":")
                if colon_index > 0:
                    file_path = line[:colon_index]
                    rest = line[colon_index:]
                    final_lines.append(f"{to_relative_path(file_path, root)}{rest}")
                else:
                    final_lines.append(line)

            return GrepResult(
                num_files=0,
                filenames=[],
                content="\n".join(final_lines),
                num_lines=len(final_lines),
                applied_limit=applied_limit,
                applied_offset=offset if offset > 0 else None,
            )

        if arguments.output_mode == "count":
            final_lines = []
            total_matches = 0
            file_count = 0
            for line in paginated:
                colon_index = line.rfind(":")
                if colon_index > 0:
                    file_path = line[:colon_index]
                    count_str = line[colon_index + 1 :]
                    try:
                        count = int(count_str)
                        total_matches += count
                        file_count += 1
                    except ValueError:
                        pass
                    final_lines.append(
                        f"{to_relative_path(file_path, root)}:{count_str}"
                    )
                else:
                    final_lines.append(line)

            return GrepResult(
                num_files=file_count,
                filenames=[],
                content="\n".join(final_lines),
                num_matches=total_matches,
                applied_limit=applied_limit,
                applied_offset=offset if offset > 0 else None,
            )

        # files_with_matches mode
        filenames = [to_relative_path(f, root) for f in paginated]
        return GrepResult(
            num_files=len(filenames),
            filenames=filenames,
            applied_limit=applied_limit,
            applied_offset=offset if offset > 0 else None,
        )

    def _apply_python_pagination(
        self,
        matches: list[str],
        arguments: GrepToolInput,
    ) -> GrepResult:
        """Apply pagination to Python fallback results."""
        offset = arguments.offset
        limit = arguments.head_limit if arguments.head_limit is not None else 250

        if limit == 0:
            paginated = matches[offset:]
            applied_limit = None
        else:
            paginated = matches[offset : offset + limit]
            was_truncated = len(matches) - offset > limit
            applied_limit = limit if was_truncated else None

        if arguments.output_mode == "content":
            return GrepResult(
                num_files=0,
                filenames=[],
                content="\n".join(paginated),
                num_lines=len(paginated),
                applied_limit=applied_limit,
                applied_offset=offset if offset > 0 else None,
            )

        if arguments.output_mode == "count":
            total_matches = 0
            file_count = 0
            for line in paginated:
                colon_index = line.rfind(":")
                if colon_index > 0:
                    try:
                        count = int(line[colon_index + 1 :])
                        total_matches += count
                        file_count += 1
                    except ValueError:
                        pass

            return GrepResult(
                num_files=file_count,
                filenames=[],
                content="\n".join(paginated),
                num_matches=total_matches,
                applied_limit=applied_limit,
                applied_offset=offset if offset > 0 else None,
            )

        # files_with_matches
        return GrepResult(
            num_files=len(paginated),
            filenames=paginated,
            applied_limit=applied_limit,
            applied_offset=offset if offset > 0 else None,
        )

    def _format_result(
        self, result: GrepResult, arguments: GrepToolInput
    ) -> ToolResult:
        """Format grep results into a ToolResult."""
        # Build limit info string
        parts = []
        if result.applied_limit is not None:
            parts.append(f"limit: {result.applied_limit}")
        if result.applied_offset:
            parts.append(f"offset: {result.applied_offset}")
        limit_info = ", ".join(parts)
        limit_suffix = f"\n\n[Showing results with pagination = {limit_info}]" if limit_info else ""

        if arguments.output_mode == "content":
            content = result.content or "No matches found"
            return ToolResult(output=content + limit_suffix)

        if arguments.output_mode == "count":
            content = result.content or "No matches found"
            matches = result.num_matches or 0
            files = result.num_files or 0
            summary = f"\n\nFound {matches} total {'occurrence' if matches == 1 else 'occurrences'} across {files} {'file' if files == 1 else 'files'}."
            return ToolResult(output=content + summary + limit_suffix)

        # files_with_matches
        if result.num_files == 0:
            return ToolResult(output="No files found")

        filenames_str = "\n".join(result.filenames)
        return ToolResult(
            output=f"Found {result.num_files} {'file' if result.num_files == 1 else 'files'}{limit_suffix}\n{filenames_str}"
        )

    @staticmethod
    def _resolve_path(base: Path, candidate: str | None) -> Path:
        if not candidate:
            return base
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = base / path
        return path.resolve()


def to_relative_path(file_path: str, root: Path) -> str:
    """Convert an absolute path to a relative path from root."""
    try:
        path = Path(file_path)
        if path.is_absolute():
            return str(path.relative_to(root))
        return file_path
    except ValueError:
        return file_path
