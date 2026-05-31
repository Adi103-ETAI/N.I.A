"""User interface messages for FileReadTool."""


def format_error_message(error_type: str, details: str = "") -> str:
    """Format an error message for display."""
    messages = {
        "not_found": "File not found",
        "is_directory": "Cannot read directory",
        "binary_file": "Binary file cannot be read as text",
        "decode_error": "File encoding not supported",
        "empty_file": "File is empty",
        "invalid_range": "Invalid line range",
    }
    
    base_message = messages.get(error_type, "Error reading file")
    if details:
        return f"{base_message}: {details}"
    return base_message
