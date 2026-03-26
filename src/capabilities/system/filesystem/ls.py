import os
import sys

from .utils import resolve_to_cwd, DEFAULT_MAX_BYTES, format_size, truncate_head

DEFAULT_LIMIT = 500

async def ls(path: str = ".") -> str:
    """List directory contents. Returns entries sorted alphabetically. 
    Output is truncated to 500 entries or 50KB to preserve context limits.
    
    Args:
        path: Directory to list (default: current directory)
    """
    cwd = os.getcwd() # Agent functions run from the process local cwd
    dir_path = resolve_to_cwd(path, cwd)
    
    if not os.path.exists(dir_path):
        return f"Error: Path not found: {dir_path}"
    if not os.path.isdir(dir_path):
        return f"Error: Not a directory: {dir_path}"
        
    entries = os.listdir(dir_path)
    entries.sort(key=str.lower)
    
    results = []
    entry_limit_reached = False
    
    for entry in entries:
        if len(results) >= DEFAULT_LIMIT:
            entry_limit_reached = True
            break
            
        full_path = os.path.join(dir_path, entry)
        suffix = "/" if os.path.isdir(full_path) else ""
        results.append(entry + suffix)
        
    if not results:
        return "(empty directory)"
        
    raw_output = "\n".join(results)
    truncation = truncate_head(raw_output, max_lines=sys.maxsize if hasattr(sys, "maxsize") else 9999999)
    
    output = truncation.content
    notices = []
    
    if entry_limit_reached:
        notices.append(f"{DEFAULT_LIMIT} entries limit reached.")
        
    if truncation.truncated:
        notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
        
    if notices:
        output += f"\n\n[{'. '.join(notices)}]"
        
    return output
