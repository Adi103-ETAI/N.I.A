"""Utils."""
def count_lines(content: str) -> int:
    if not content:
        return 0
    parts = content.split("\n")
    return len(parts) - 1 if content.endswith("\n") else len(parts)

def get_operation_type(file_existed: bool) -> str:
    return "update" if file_existed else "create"
