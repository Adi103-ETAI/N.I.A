import os
from typing import List

from .utils import (
    DEFAULT_MAX_BYTES, 
    format_size, 
    resolve_to_cwd,
    truncate_head,
    truncate_line
)

DEFAULT_GREP_LIMIT = 200

async def grep_search(pattern: str, path: str = ".") -> str:
    """Search for a specific text pattern inside files within a directory.
    
    Args:
        pattern: The exact text or substring to search for inside file contents
        path: The file or directory to search in (default: current directory)
    """
    cwd = os.getcwd()
    target_path = resolve_to_cwd(path, cwd)
    
    if not os.path.exists(target_path):
        return f"Error: Path not found: {target_path}"

    results: List[str] = []
    match_count = 0
    file_count = 0
    limit_reached = False

    excludes = {".git", ".venv", "node_modules", "__pycache__", ".next", "dist", "build"}
    # Very basic text file extension heuristic to avoid searching binaries
    binary_exts = {".jpg", ".png", ".gif", ".pdf", ".zip", ".tar", ".gz", ".mp4", ".ico", ".sqlite3"}

    def search_file(file_path: str, rel_path: str):
        nonlocal match_count, file_count, limit_reached
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                
            file_matches = []
            for i, line in enumerate(lines):
                if pattern in line:
                    if match_count >= DEFAULT_GREP_LIMIT:
                        limit_reached = True
                        break
                    
                    clean_line = line.strip()
                    truncated_line, _ = truncate_line(clean_line)
                    file_matches.append(f"  {i+1}: {truncated_line}")
                    match_count += 1
                    
            if file_matches:
                file_count += 1
                results.append(rel_path)
                results.extend(file_matches)
                results.append("") # Empty line between files
                
        except Exception:
            pass # Skip unreadable files silently

    if os.path.isfile(target_path):
        search_file(target_path, os.path.basename(target_path))
    else:
        for root, dirs, files in os.walk(target_path):
            dirs[:] = [d for d in dirs if d not in excludes]
            
            for file in files:
                if limit_reached:
                    break
                    
                _, ext = os.path.splitext(file)
                if ext.lower() in binary_exts:
                    continue
                    
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, target_path)
                search_file(full_path, rel_path)
            
            if limit_reached:
                break

    if not results:
        return f"No matches found for '{pattern}' in {path}"

    raw_output = "\\n".join(results)
    truncation = truncate_head(raw_output)
    
    output = truncation.content
    notices = []
    
    if limit_reached:
        notices.append(f"{DEFAULT_GREP_LIMIT} match limit reached")
        
    if truncation.truncated:
        notices.append(f"{format_size(DEFAULT_MAX_BYTES)} byte limit reached")
        
    if notices:
        output += f"\\n\\n[{'. '.join(notices)}]"
        
    return f"Found {match_count} matches in {file_count} files:\\n\\n{output}"
