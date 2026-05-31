"""Utility functions for FileReadTool."""

from __future__ import annotations


def add_line_numbers(content: str, start_line: int = 1) -> str:
    """
    Add line numbers to content.
    
    Args:
        content: The content to number
        start_line: The starting line number (1-indexed)
        
    Returns:
        Content with line numbers prepended
    """
    lines = content.splitlines()
    numbered = [
        f"{start_line + index:>6}. {line}"
        for index, line in enumerate(lines)
    ]
    return "\n".join(numbered)


def detect_binary(data: bytes) -> bool:
    """
    Detect if data is binary (contains null bytes).
    
    Args:
        data: The byte data to check
        
    Returns:
        True if binary, False if text
    """
    return b"\x00" in data[:8192]  # Check first 8KB


def format_file_info(
    file_path: str,
    total_lines: int,
    start_line: int,
    num_lines: int,
) -> str:
    """
    Format file information header.
    
    Args:
        file_path: Path to the file
        total_lines: Total lines in file
        start_line: Starting line number
        num_lines: Number of lines shown
        
    Returns:
        Formatted info string
    """
    if start_line == 1 and num_lines == total_lines:
        return f"{file_path} ({total_lines} lines)"
    return f"{file_path} (lines {start_line}-{start_line + num_lines - 1} of {total_lines})"
