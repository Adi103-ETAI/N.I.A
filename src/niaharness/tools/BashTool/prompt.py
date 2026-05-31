"""BashTool prompt."""

from __future__ import annotations


def get_bash_description() -> str:
    """Get the bash tool description."""
    return """Execute shell commands with streaming output and background task support.

Features:
- Streaming output with real-time progress updates
- Background task execution with output capture
- Configurable timeout handling
- Automatic detection of read-only commands (search, list, read)
- Image output detection

Usage:
- Provide the command to execute
- Optional timeout in seconds (default: 120, max: 600)
- Optional description of what the command does
- Set run_in_background=true for long-running commands

Example:
  command: ls -la /path/to/directory
  description: List files in directory
"""
