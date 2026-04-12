"""User interface messages for FileEditTool."""


def format_error_message(error_type: str, details: str = "") -> str:
    """
    Format an error message for display.
    
    Args:
        error_type: The type of error
        details: Additional error details
        
    Returns:
        Formatted error message
    """
    messages = {
        "not_found": "File not found",
        "string_not_found": "String to replace not found in file",
        "multiple_matches": "Found multiple matches but replace_all is false",
        "not_unique": "String is not unique in file",
        "no_changes": "No changes to make: old_string and new_string are the same",
        "file_modified": "File has been modified since read",
        "not_read": "File has not been read yet. Read it first before writing to it",
        "too_large": "File is too large to edit",
    }
    
    base_message = messages.get(error_type, "Error editing file")
    if details:
        return f"{base_message}: {details}"
    return base_message


def format_success_message(file_path: str, changes_made: int = 1) -> str:
    """
    Format a success message.
    
    Args:
        file_path: The path of the edited file
        changes_made: Number of replacements made
        
    Returns:
        Formatted success message
    """
    if changes_made == 1:
        return f"Successfully edited {file_path}"
    return f"Successfully edited {file_path} ({changes_made} replacements)"
