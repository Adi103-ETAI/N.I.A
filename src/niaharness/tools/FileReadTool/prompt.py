"""FileReadTool prompt."""

from __future__ import annotations


def get_file_read_description() -> str:
    """Get the file read tool description."""
    return """Read a file from the local filesystem with line numbers.

Supports:
- Text files with encoding detection (UTF-8, UTF-16, etc.)
- Image files (PNG, JPG, GIF, WebP) - returns base64-encoded data
- PDF files - returns base64-encoded data with optional page range
- Jupyter notebooks (.ipynb) - returns cell contents

Usage:
- Provide the absolute file_path
- Use offset and limit for large files
- Use pages parameter for PDF page ranges (e.g., "1-5", "3", "10-20")

Example:
  file_path: /path/to/file.py
  offset: 1
  limit: 100
"""
