"""UI."""
def format_success_message(operation: str, file_path: str, num_lines: int) -> str:
    verb = "Created" if operation == "create" else "Updated"
    return f"{verb} {file_path} ({num_lines} lines)"
