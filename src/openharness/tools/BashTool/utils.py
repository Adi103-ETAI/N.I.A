"""Utils."""
def strip_empty_lines(content: str) -> str:
    """Strip leading/trailing empty lines."""
    lines = content.split("\n")
    
    start_index = 0
    while start_index < len(lines) and not lines[start_index].strip():
        start_index += 1
    
    end_index = len(lines) - 1
    while end_index >= 0 and not lines[end_index].strip():
        end_index -= 1
    
    if start_index > end_index:
        return ""
    
    return "\n".join(lines[start_index:end_index + 1])


def truncate_output(text: str, max_length: int) -> str:
    """Truncate output if too long."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "\n...[truncated]..."
