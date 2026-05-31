"""FileEditTool UI formatting."""

from __future__ import annotations


def format_success_message(file_path: str, replacements: int) -> str:
    """Format a success message for a file edit."""
    if replacements == 1:
        return f"The file {file_path} has been updated successfully."
    return f"The file {file_path} has been updated successfully. All {replacements} occurrences were replaced."


def format_error_message(error_type: str, details: str) -> str:
    """Format an error message for display."""
    messages = {
        "no_changes": "No changes to make: old_string and new_string are exactly the same.",
        "not_found": f"File does not exist: {details}",
        "too_large": f"File is too large to edit: {details}",
        "string_not_found": f"String to replace not found in file:\n{details}",
        "multiple_matches": f"Multiple matches found: {details}",
        "binary_file": f"Cannot edit binary file: {details}",
        "decode_error": f"Failed to decode file: {details}",
    }
    return messages.get(error_type, f"Error ({error_type}): {details}")
