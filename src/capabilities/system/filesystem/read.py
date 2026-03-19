import os
import mimetypes
import base64
from typing import Optional

from .utils import (
    DEFAULT_MAX_LINES, 
    DEFAULT_MAX_BYTES, 
    format_size, 
    truncate_head,
    resolve_read_path
)

def get_mime_type(file_path: str) -> Optional[str]:
    mime, _ = mimetypes.guess_type(file_path)
    if mime and mime.startswith("image/"):
        return mime
    return None

async def read_file(path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> str:
    """Read the contents of a file. Supports text files. 
    Output is automatically truncated to protect context limits.
    
    Args:
        path: Path to the file to read (relative or absolute)
        offset: Line number to start reading from (1-indexed)
        limit: Maximum number of lines to read
    """
    cwd = os.getcwd()
    abs_path = resolve_read_path(path, cwd)
    if not os.path.exists(abs_path):
        return f"Error: File not found: {abs_path}"
        
    mime_type = get_mime_type(abs_path)
    if mime_type:
        return f"Error: Cannot read binary/image file [{mime_type}] as text. Provide a text file."
        
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.read().splitlines()
    except Exception as e:
        return f"Error reading file: {e}"
        
    total_lines = len(all_lines)
    start_line = max(0, (offset or 1) - 1)
    start_line_display = start_line + 1
    
    if total_lines > 0 and start_line >= total_lines:
        return f"Error: Offset {offset} is beyond end of file ({total_lines} lines total)"
        
    if limit is not None:
        end_line = min(start_line + limit, total_lines)
        selected_content = "\\n".join(all_lines[start_line:end_line])
        user_limited_lines = end_line - start_line
    else:
        selected_content = "\\n".join(all_lines[start_line:])
        user_limited_lines = None
        
    truncation = truncate_head(selected_content)
    
    if truncation.firstLineExceedsLimit:
        first_line_size = format_size(len(all_lines[start_line].encode('utf-8')))
        output_text = f"[Line {start_line_display} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit.]"
    elif truncation.truncated:
        end_line_display = start_line_display + truncation.outputLines - 1
        next_offset = end_line_display + 1
        
        output_text = truncation.content
        if truncation.truncatedBy == "lines":
            output_text += f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_lines}. Use offset={next_offset} to continue.]"
        else:
            output_text += f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_lines} ({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
    elif user_limited_lines is not None and start_line + user_limited_lines < total_lines:
        remaining = total_lines - (start_line + user_limited_lines)
        next_offset = start_line + user_limited_lines + 1
        output_text = truncation.content + f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
    else:
        output_text = truncation.content
        
    return output_text
