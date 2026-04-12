"""UI."""
def format_command_output(stdout: str, stderr: str, returncode: int) -> str:
    """Format command output."""
    parts = []
    if stdout:
        parts.append(stdout.rstrip())
    if stderr:
        parts.append(stderr.rstrip())
    
    text = "\n".join(part for part in parts if part).strip()
    return text if text else "(no output)"


def format_error_message(error_type: str, details: str = "") -> str:
    """Format error message."""
    messages = {
        "timeout": "Command timed out",
        "failed": "Command failed",
    }
    base = messages.get(error_type, "Error executing command")
    return f"{base}: {details}" if details else base
