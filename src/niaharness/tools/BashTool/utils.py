"""BashTool utility functions."""

from __future__ import annotations

import re
from typing import NamedTuple

from .constants import (
    BASH_LIST_COMMANDS,
    BASH_READ_COMMANDS,
    BASH_SEARCH_COMMANDS,
    BASH_SEMANTIC_NEUTRAL_COMMANDS,
)


def strip_empty_lines(content: str) -> str:
    """Strip leading/trailing empty lines from content."""
    lines = content.split("\n")

    start_index = 0
    while start_index < len(lines) and not lines[start_index].strip():
        start_index += 1

    end_index = len(lines) - 1
    while end_index >= 0 and not lines[end_index].strip():
        end_index -= 1

    if start_index > end_index:
        return ""

    return "\n".join(lines[start_index : end_index + 1])


def truncate_output(text: str, max_length: int) -> str:
    """Truncate output if too long."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "\n...[truncated]..."


def split_command_with_operators(command: str) -> list[str]:
    """
    Split a shell command into parts, preserving operators.

    Handles pipes (|), logical operators (&&, ||), semicolons, and redirects.
    """
    parts: list[str] = []
    current: list[str] = []
    i = 0
    in_single = False
    in_double = False
    escaped = False

    for ch in command:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            current.append(ch)
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            continue

        if in_single or in_double:
            current.append(ch)
            continue

        # Check for two-char operators first
        if ch == "&" and i + 1 < len(command) and command[i + 1] == "&":
            if current:
                parts.append("".join(current).strip())
                current = []
            parts.append("&&")
            i += 1
            continue
        if ch == "|" and i + 1 < len(command) and command[i + 1] == "|":
            if current:
                parts.append("".join(current).strip())
                current = []
            parts.append("||")
            i += 1
            continue
        if ch == ">" and i + 1 < len(command) and command[i + 1] == ">":
            if current:
                parts.append("".join(current).strip())
                current = []
            parts.append(">>")
            i += 1
            continue

        if ch == "|":
            if current:
                parts.append("".join(current).strip())
                current = []
            parts.append("|")
        elif ch == ";":
            if current:
                parts.append("".join(current).strip())
                current = []
            parts.append(";")
        elif ch == ">":
            if current:
                parts.append("".join(current).strip())
                current = []
            parts.append(">")
        else:
            current.append(ch)

        i += 1

    remainder = "".join(current).strip()
    if remainder:
        parts.append(remainder)

    return parts


class SearchReadResult(NamedTuple):
    """Result of is_search_or_read_command check."""

    is_search: bool
    is_read: bool
    is_list: bool


def is_search_or_read_command(command: str) -> SearchReadResult:
    """
    Check if a bash command is a search or read operation.

    For pipelines, ALL parts must be search/read commands for the whole
    command to be considered collapsible.
    """
    try:
        parts = split_command_with_operators(command)
    except Exception:
        return SearchReadResult(is_search=False, is_read=False, is_list=False)

    if not parts:
        return SearchReadResult(is_search=False, is_read=False, is_list=False)

    has_search = False
    has_read = False
    has_list = False
    has_non_neutral = False
    skip_next = False

    for part in parts:
        if skip_next:
            skip_next = False
            continue
        if part in (">", ">>", ">&"):
            skip_next = True
            continue
        if part in ("||", "&&", "|", ";"):
            continue

        tokens = part.strip().split()
        if not tokens:
            continue
        base_cmd = tokens[0]

        if base_cmd in BASH_SEMANTIC_NEUTRAL_COMMANDS:
            continue

        has_non_neutral = True
        is_part_search = base_cmd in BASH_SEARCH_COMMANDS
        is_part_read = base_cmd in BASH_READ_COMMANDS
        is_part_list = base_cmd in BASH_LIST_COMMANDS

        if not is_part_search and not is_part_read and not is_part_list:
            return SearchReadResult(is_search=False, is_read=False, is_list=False)

        if is_part_search:
            has_search = True
        if is_part_read:
            has_read = True
        if is_part_list:
            has_list = True

    if not has_non_neutral:
        return SearchReadResult(is_search=False, is_read=False, is_list=False)

    return SearchReadResult(is_search=has_search, is_read=has_read, is_list=has_list)


def is_silent_command(command: str) -> bool:
    """Check if a command is expected to produce no stdout on success."""
    try:
        parts = split_command_with_operators(command)
    except Exception:
        return False

    if not parts:
        return False

    has_non_fallback = False
    last_operator: str | None = None
    skip_next = False

    for part in parts:
        if skip_next:
            skip_next = False
            continue
        if part in (">", ">>", ">&"):
            skip_next = True
            continue
        if part in ("||", "&&", "|", ";"):
            last_operator = part
            continue

        tokens = part.strip().split()
        if not tokens:
            continue
        base_cmd = tokens[0]

        if last_operator == "||" and base_cmd in BASH_SEMANTIC_NEUTRAL_COMMANDS:
            continue

        has_non_fallback = True
        if base_cmd not in BASH_SILENT_COMMANDS:
            return False

    return has_non_fallback


def is_image_output(stdout: str) -> bool:
    """Detect if stdout contains image data (base64-encoded)."""
    if not stdout:
        return False
    stripped = stdout.strip()
    if len(stripped) < 100:
        return False
    image_signatures = [
        b"\x89PNG",
        b"\xff\xd8\xff",
        b"GIF87a",
        b"GIF89a",
        b"RIFF",
    ]
    try:
        raw = stripped[:1024].encode("utf-8", errors="replace")
        for sig in image_signatures:
            if raw[: len(sig)] == sig:
                return True
    except Exception:
        pass
    return False


def is_autobackgrounding_allowed(command: str) -> bool:
    """Check if a command is allowed to be auto-backgrounded."""
    from .constants import DISALLOWED_AUTO_BACKGROUND_COMMANDS

    tokens = command.strip().split()
    if not tokens:
        return True
    return tokens[0] not in DISALLOWED_AUTO_BACKGROUND_COMMANDS
