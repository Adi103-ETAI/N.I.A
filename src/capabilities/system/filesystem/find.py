import os
import sys
from typing import List, Optional

from .utils import (
    DEFAULT_MAX_BYTES, 
    format_size, 
    resolve_to_cwd,
    truncate_head
)

DEFAULT_FIND_LIMIT = 500

async def find_files(pattern: str, path: str = ".") -> str:
    """Find files in a directory that match a substring pattern.
    
    Args:
        pattern: The substring to search for in filenames (e.g. '.py', 'app', 'config')
        path: The root directory to start searching from (default: current directory)
    """
    cwd = os.getcwd()
    dir_path = resolve_to_cwd(path, cwd)
    
    if not os.path.exists(dir_path):
        return f"Error: Path not found: {dir_path}"
    if not os.path.isdir(dir_path):
        return f"Error: Not a directory: {dir_path}"

    results: List[str] = []
    limit_reached = False
    
    # Exclude common noisy directories
    excludes = {".git", ".venv", "node_modules", "__pycache__", ".next", "dist", "build"}

    for root, dirs, files in os.walk(dir_path):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in excludes]
        
        for file in files:
            if pattern.lower() in file.lower():
                if len(results) >= DEFAULT_FIND_LIMIT:
                    limit_reached = True
                    break
                
                # Get path relative to the searched directory
                rel_path = os.path.relpath(os.path.join(root, file), dir_path)
                results.append(rel_path)
        
        if limit_reached:
            break

    if not results:
        return f"No files found matching '{pattern}' in {path}"

    results.sort(key=str.lower)
    raw_output = "\\n".join(results)
    
    truncation = truncate_head(raw_output, max_lines=sys.maxsize if hasattr(sys, "maxsize") else 9999999)
    output = truncation.content
    notices = []
    
    if limit_reached:
        notices.append(f"{DEFAULT_FIND_LIMIT} entries limit reached")
        
    if truncation.truncated:
        notices.append(f"{format_size(DEFAULT_MAX_BYTES)} byte limit reached")
        
    if notices:
        output += f"\\n\\n[{'. '.join(notices)}]"
        
    return output
