"""BashTool UI formatting."""

from __future__ import annotations


def format_command_output(stdout: str, stderr: str, return_code: int) -> str:
    """Format command output for display."""
    parts = []
    if stdout.strip():
        parts.append(stdout.strip())
    if stderr.strip():
        parts.append(f"STDERR:\n{stderr.strip()}")
    if not parts:
        return "(No output)" if return_code == 0 else f"Exit code: {return_code}"
    return "\n\n".join(parts)


def format_error_message(error_type: str, details: str) -> str:
    """Format an error message for display."""
    return f"Error ({error_type}): {details}"


def format_timeout_message(seconds: int) -> str:
    """Format a timeout message."""
    return f"Command timed out after {seconds} seconds"


def format_background_status(task_id: str, output_path: str, command: str) -> str:
    """Format background task status."""
    return (
        f"Background task running.\n"
        f"Task ID: {task_id}\n"
        f"Output: {output_path}\n"
        f"Command: {command}"
    )
