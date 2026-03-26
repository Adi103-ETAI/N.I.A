"""Security policy constants and helpers for Warden enforcement."""
from __future__ import annotations

from typing import Final

AUTO_APPROVED_TOOLS: Final[set[str]] = {
    "sandboxed_shell",
    "start_session",
    "end_session",
}

BLOCKED_HOST_FILE_TOOLS: Final[set[str]] = {
    "write_file",
    "delete_file",
    "git_clone",
}

ALLOWED_HOST_PROCESS_TOOLS: Final[set[str]] = {
    "terminate_process",
    "find_process",
}

RESTRICTED_FILE_TOOLS: Final[set[str]] = {
    "move_file",
}


def is_auto_approved_tool(tool_name: str) -> bool:
    """Return True when a high-risk tool can bypass additional checks."""
    return tool_name in AUTO_APPROVED_TOOLS


def is_blocked_host_file_tool(tool_name: str) -> bool:
    """Return True when host-side file operation must be denied."""
    return tool_name in BLOCKED_HOST_FILE_TOOLS


def is_allowed_host_process_tool(tool_name: str) -> bool:
    """Return True when host process operation is allowed by policy."""
    return tool_name in ALLOWED_HOST_PROCESS_TOOLS


def is_restricted_file_tool(tool_name: str) -> bool:
    """Return True when file operation must be validated against safe zones."""
    return tool_name in RESTRICTED_FILE_TOOLS


__all__ = [
    "AUTO_APPROVED_TOOLS",
    "BLOCKED_HOST_FILE_TOOLS",
    "ALLOWED_HOST_PROCESS_TOOLS",
    "RESTRICTED_FILE_TOOLS",
    "is_auto_approved_tool",
    "is_blocked_host_file_tool",
    "is_allowed_host_process_tool",
    "is_restricted_file_tool",
]
