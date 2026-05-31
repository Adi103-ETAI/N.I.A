"""FileReadTool UI formatting."""

from __future__ import annotations


def format_error_message(error_type: str, details: str) -> str:
    """Format an error message for display."""
    messages = {
        "not_found": f"File does not exist. Current working directory: {details}",
        "is_directory": f"Path is a directory, not a file: {details}",
        "binary_file": f"Cannot read binary file: {details}",
        "decode_error": f"Failed to decode file: {details}",
        "invalid_range": f"Invalid read range: {details}",
        "invalid_pages": f"Invalid pages parameter: {details}",
        "too_many_pages": f"Too many pages requested: {details}",
    }
    return messages.get(error_type, f"Error ({error_type}): {details}")


def format_image_info(file_path: str, media_type: str, size: int) -> str:
    """Format image file information."""
    return (
        f"Image file: {file_path}\n"
        f"Type: {media_type}\n"
        f"Size: {_format_size(size)}"
    )


def format_pdf_info(file_path: str, size: int, pages: str | None = None) -> str:
    """Format PDF file information."""
    info = (
        f"PDF file: {file_path}\n"
        f"Size: {_format_size(size)}"
    )
    if pages:
        info += f"\nPage range: {pages}"
    return info


def format_notebook_info(file_path: str, num_cells: int) -> str:
    """Format notebook file information."""
    return f"Jupyter Notebook: {file_path}\nTotal cells: {num_cells}"


def _format_size(size_bytes: int) -> str:
    """Format file size to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
