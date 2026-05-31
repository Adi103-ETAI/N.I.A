"""FileEditTool prompt."""

from __future__ import annotations


def get_edit_tool_description() -> str:
    """Get the edit tool description."""
    return """A tool for editing files.

This tool replaces exact strings in files. It supports:
- Exact string matching with quote normalization (curly quotes ↔ straight quotes)
- Replace all occurrences with replace_all=true
- Diff generation showing changes
- File modification time tracking to detect concurrent edits

Usage:
- Provide the exact old_string to replace
- Provide the new_string replacement
- old_string and new_string must be different
- The file must be read before editing

Example:
  file_path: /path/to/file.py
  old_string: def old_function():
  new_string: def new_function():
"""
