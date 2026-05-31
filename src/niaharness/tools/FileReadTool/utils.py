"""FileReadTool utility functions."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

# Image file extensions
IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"})

# PDF extensions
PDF_EXTENSIONS = frozenset({"pdf"})

# Notebook extensions
NOTEBOOK_EXTENSIONS = frozenset({"ipynb"})

# Binary extensions that should not be read as text
BINARY_EXTENSIONS = frozenset({
    "exe", "dll", "so", "dylib", "bin", "o", "a",
    "zip", "tar", "gz", "bz2", "xz", "7z", "rar",
    "mp3", "mp4", "avi", "mov", "mkv", "flac", "wav",
    "pyc", "pyo", "class", "wasm",
})


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
    Detect if data is binary (contains null bytes or too many non-text bytes).

    Args:
        data: The byte data to check

    Returns:
        True if binary, False if text
    """
    # Check first 8KB for null bytes
    check_size = min(len(data), 8192)
    if b"\x00" in data[:check_size]:
        return True

    # Check for high ratio of non-ASCII bytes
    if check_size > 0:
        non_ascii = sum(1 for b in data[:check_size] if b > 127)
        if non_ascii / check_size > 0.3:
            return True

    return False


def detect_encoding(data: bytes) -> str:
    """
    Detect file encoding from byte data.

    Args:
        data: The byte data to detect encoding from

    Returns:
        Detected encoding name
    """
    # Check for UTF-16 LE BOM
    if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xFE:
        return "utf-16-le"

    # Check for UTF-16 BE BOM
    if len(data) >= 2 and data[0] == 0xFE and data[1] == 0xFF:
        return "utf-16-be"

    # Check for UTF-8 BOM
    if len(data) >= 3 and data[0] == 0xEF and data[1] == 0xBB and data[2] == 0xBF:
        return "utf-8-sig"

    # Try to detect UTF-8 validity
    try:
        data[:8192].decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    # Default to UTF-8 with replacement
    return "utf-8"


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


def is_image_file(file_path: str) -> bool:
    """Check if a file is an image based on extension."""
    ext = Path(file_path).suffix.lower().lstrip(".")
    return ext in IMAGE_EXTENSIONS


def is_pdf_file(file_path: str) -> bool:
    """Check if a file is a PDF based on extension."""
    ext = Path(file_path).suffix.lower().lstrip(".")
    return ext in PDF_EXTENSIONS


def is_notebook_file(file_path: str) -> bool:
    """Check if a file is a Jupyter notebook based on extension."""
    ext = Path(file_path).suffix.lower().lstrip(".")
    return ext in NOTEBOOK_EXTENSIONS


def is_binary_file(file_path: str) -> bool:
    """Check if a file is binary based on extension."""
    ext = Path(file_path).suffix.lower().lstrip(".")
    return ext in BINARY_EXTENSIONS


def get_media_type(file_path: str) -> str:
    """Get the MIME type for a file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type

    ext = Path(file_path).suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
        ".pdf": "application/pdf",
    }
    return media_types.get(ext, "application/octet-stream")


def read_image_as_base64(file_path: str) -> tuple[str, str, int]:
    """
    Read an image file and return base64-encoded data.

    Returns:
        Tuple of (base64_data, media_type, file_size)
    """
    path = Path(file_path)
    data = path.read_bytes()
    media_type = get_media_type(file_path)
    b64_data = base64.b64encode(data).decode("ascii")
    return b64_data, media_type, len(data)


def read_pdf_as_base64(file_path: str) -> tuple[str, int]:
    """
    Read a PDF file and return base64-encoded data.

    Returns:
        Tuple of (base64_data, file_size)
    """
    path = Path(file_path)
    data = path.read_bytes()
    b64_data = base64.b64encode(data).decode("ascii")
    return b64_data, len(data)


def read_notebook(file_path: str) -> list[dict]:
    """
    Read a Jupyter notebook and extract cells.

    Returns:
        List of cell dictionaries with cell_type and source
    """
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    notebook = json.loads(content)

    cells = []
    for cell in notebook.get("cells", []):
        cell_data = {
            "cell_type": cell.get("cell_type", "unknown"),
            "source": "".join(cell.get("source", [])),
        }
        if "execution_count" in cell:
            cell_data["execution_count"] = cell["execution_count"]
        cells.append(cell_data)

    return cells


def parse_pdf_page_range(pages_str: str) -> tuple[int, int] | None:
    """
    Parse a PDF page range string like "1-5" or "3".

    Returns:
        Tuple of (first_page, last_page) or None if invalid
    """
    if not pages_str:
        return None

    pages_str = pages_str.strip()

    if "-" in pages_str:
        parts = pages_str.split("-", 1)
        try:
            first = int(parts[0].strip())
            last = int(parts[1].strip())
            if first < 1 or last < first:
                return None
            return (first, last)
        except ValueError:
            return None
    else:
        try:
            page = int(pages_str)
            if page < 1:
                return None
            return (page, page)
        except ValueError:
            return None


def format_file_size(size_bytes: int) -> str:
    """Format a file size in bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
