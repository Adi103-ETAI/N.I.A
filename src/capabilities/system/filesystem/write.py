import os
from .utils import resolve_to_cwd

async def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates the file if it doesn't exist, 
    overwrites if it does. Automatically creates parent directories.
    
    Args:
        path: Path to the file to write (relative or absolute)
        content: Content to write to the file
    """
    cwd = os.getcwd()
    abs_path = resolve_to_cwd(path, cwd)
    
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"
