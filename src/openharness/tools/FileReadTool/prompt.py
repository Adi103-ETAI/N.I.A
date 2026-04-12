"""Prompt and description for FileReadTool."""

from .constants import MAX_LINES_TO_READ


def get_file_read_description() -> str:
    """Get the tool description for FileReadTool."""
    return f"""Reads a file from the local filesystem. You can access any file directly by using this tool.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to {MAX_LINES_TO_READ} lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files)
- Results are returned with line numbers starting at 1
- This tool can only read files, not directories
- If you read a file that exists but has empty contents you will receive a warning"""
