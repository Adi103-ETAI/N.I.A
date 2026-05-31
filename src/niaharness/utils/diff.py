"""Diff computation utilities.

Provides functions for computing and displaying diffs between file contents.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import List, Optional


CONTEXT_LINES = 3
DIFF_TIMEOUT_MS = 5_000


@dataclass(frozen=True)
class DiffHunk:
    """A single hunk in a diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str]


AMPERSAND_TOKEN = "<<:AMPERSAND_TOKEN:>>"
DOLLAR_TOKEN = "<<:DOLLAR_TOKEN:>>"


def _escape_for_diff(s: str) -> str:
    """Escape special characters that confuse the diff library."""
    return s.replace("&", AMPERSAND_TOKEN).replace("$", DOLLAR_TOKEN)


def _unescape_from_diff(s: str) -> str:
    """Restore special characters after diff computation."""
    return s.replace(AMPERSAND_TOKEN, "&").replace(DOLLAR_TOKEN, "$")


def adjust_hunk_line_numbers(hunks: List[DiffHunk], offset: int) -> List[DiffHunk]:
    """Shift hunk line numbers by offset.

    Use when get_patch_for_display received a slice of the file rather than
    the whole file.
    """
    if offset == 0:
        return hunks
    return [
        DiffHunk(
            old_start=h.old_start + offset,
            old_count=h.old_count,
            new_start=h.new_start + offset,
            new_count=h.new_count,
            lines=h.lines,
        )
        for h in hunks
    ]


def get_patch_from_contents(
    file_path: str,
    old_content: str,
    new_content: str,
    ignore_whitespace: bool = False,
    single_hunk: bool = False,
) -> List[DiffHunk]:
    """Compute a diff patch from old and new content.

    Returns a list of DiffHunk objects representing the changes.
    """
    old_lines = _escape_for_diff(old_content).splitlines(keepends=True)
    new_lines = _escape_for_diff(new_content).splitlines(keepends=True)

    context = 100_000 if single_hunk else CONTEXT_LINES

    differ = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=file_path,
        tofile=file_path,
        n=context,
    )

    return _parse_unified_diff(list(differ))


def _parse_unified_diff(diff_lines: List[str]) -> List[DiffHunk]:
    """Parse unified diff output into DiffHunk objects."""
    hunks: list[DiffHunk] = []
    current_hunk: Optional[DiffHunk] = None

    for line in diff_lines:
        # Skip file headers
        if line.startswith("---") or line.startswith("+++"):
            continue

        # Parse hunk header
        hunk_match = re.match(r"^@@ -(\d+),?\d* \+(\d+),?\d* @@", line)
        if hunk_match:
            if current_hunk is not None:
                hunks.append(current_hunk)
            old_start = int(hunk_match.group(1))
            new_start = int(hunk_match.group(2))
            current_hunk = DiffHunk(
                old_start=old_start,
                old_count=0,
                new_start=new_start,
                new_count=0,
                lines=[],
            )
            continue

        if current_hunk is not None:
            unescaped = _unescape_from_diff(line.rstrip("\n"))
            current_hunk = DiffHunk(
                old_start=current_hunk.old_start,
                old_count=current_hunk.old_count + (1 if line.startswith("-") or line.startswith(" ") else 0),
                new_start=current_hunk.new_start,
                new_count=current_hunk.new_count + (1 if line.startswith("+") or line.startswith(" ") else 0),
                lines=current_hunk.lines + [unescaped],
            )

    if current_hunk is not None:
        hunks.append(current_hunk)

    return hunks


def count_lines_changed(
    hunks: List[DiffHunk],
    new_file_content: Optional[str] = None,
) -> tuple[int, int]:
    """Count lines added and removed in a diff patch.

    Returns (additions, removals).
    """
    if not hunks and new_file_content:
        # For new files, count all lines as additions
        return len(new_file_content.splitlines()), 0

    additions = 0
    removals = 0

    for hunk in hunks:
        for line in hunk.lines:
            if line.startswith("+"):
                additions += 1
            elif line.startswith("-"):
                removals += 1

    return additions, removals


@dataclass
class FileEdit:
    """A file edit operation."""

    old_string: str
    new_string: str
    replace_all: bool = False


def _convert_leading_tabs_to_spaces(content: str, tab_size: int = 4) -> str:
    """Convert leading tabs to spaces for display."""
    lines = content.split("\n")
    result = []
    for line in lines:
        stripped = line.lstrip("\t")
        num_tabs = len(line) - len(stripped)
        result.append(" " * (num_tabs * tab_size) + stripped)
    return "\n".join(result)


def get_patch_for_display(
    file_path: str,
    file_contents: str,
    edits: List[FileEdit],
    ignore_whitespace: bool = False,
) -> List[DiffHunk]:
    """Get a diff patch with edits applied for display.

    Converts leading tabs to spaces and computes the diff.
    """
    prepared = _escape_for_diff(_convert_leading_tabs_to_spaces(file_contents))

    # Apply edits to the prepared content
    new_content = prepared
    for edit in edits:
        escaped_old = _escape_for_diff(_convert_leading_tabs_to_spaces(edit.old_string))
        escaped_new = _escape_for_diff(_convert_leading_tabs_to_spaces(edit.new_string))

        if edit.replace_all:
            new_content = new_content.replace(escaped_old, escaped_new)
        else:
            new_content = new_content.replace(escaped_old, escaped_new, 1)

    # Compute diff
    old_lines = prepared.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    differ = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=file_path,
        tofile=file_path,
        n=CONTEXT_LINES,
    )

    return _parse_unified_diff(list(differ))


def format_diff_for_display(hunks: List[DiffHunk]) -> str:
    """Format diff hunks as a human-readable string."""
    lines = []
    for hunk in hunks:
        lines.append(f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@")
        for line in hunk.lines:
            lines.append(line)
    return "\n".join(lines)
