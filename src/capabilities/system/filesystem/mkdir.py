import os
from .utils import resolve_to_cwd

async def make_dir(path: str) -> str:
    """Create a new directory. Automatically creates parent directories if needed.
    
    Args:
        path: Path to the directory to create (relative or absolute)
    """
    cwd = os.getcwd()
    abs_path = resolve_to_cwd(path, cwd)
    
    try:
        if os.path.exists(abs_path):
            if os.path.isdir(abs_path):
                return f"Directory already exists: {path}"
            return f"Error: A file already exists at {path}"
            
        os.makedirs(abs_path, exist_ok=True)
        return f"Successfully created directory: {path}"
    except Exception as e:
        return f"Error creating directory: {e}"
