"""Utility functions for FileEditTool."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import HunkInfo

# Claude can't output curly quotes, so we define them as constants
LEFT_SINGLE_CURLY_QUOTE = "'"
RIGHT_SINGLE_CURLY_QUOTE = "'"
LEFT_DOUBLE_CURLY_QUOTE = """
RIGHT_DOUBLE_CURLY_QUOTE = """


def normalize_quotes(text: str) -> str:
    """
    Normalize quotes in a string by converting curly quotes to straight quotes.
    
    Args:
        text: The string to normalize
        
    Returns:
        The string with all curly quotes replaced by straight quotes
    """
    return (
        text
        .replace(LEFT_SINGLE_CURLY_QUOTE, "'")
        .replace(RIGHT_SINGLE_CURLY_QUOTE, "'")
        .replace(LEFT_DOUBLE_CURLY_QUOTE, '"')
        .replace(RIGHT_DOUBLE_CURLY_QUOTE, '"')
    )


def strip_trailing_whitespace(text: str) -> str:
    """
    Strip trailing whitespace from each line while preserving line endings.
    
    Args:
        text: The string to process
        
    Returns:
        The string with trailing whitespace removed from each line
    """
    lines = text.splitlines(keepends=True)
    result = []
    for line in lines:
        # Strip trailing whitespace but keep the line ending
        if line.endswith('\r\n'):
            result.append(line.rstrip() + '\r\n' if line.strip() else '\r\n')
        elif line.endswith('\n'):
            result.append(line.rstrip() + '\n' if line.strip() else '\n')
        elif line.endswith('\r'):
            result.append(line.rstrip() + '\r' if line.strip() else '\r')
        else:
            result.append(line.rstrip())
    return ''.join(result)


def find_actual_string(file_content: str, search_string: str) -> str | None:
    """
    Find the actual string in the file content that matches the search string,
    accounting for quote normalization.
    
    Args:
        file_content: The file content to search in
        search_string: The string to search for
        
    Returns:
        The actual string found in the file, or None if not found
    """
    # First try exact match
    if search_string in file_content:
        return search_string
    
    # Try with normalized quotes
    normalized_search = normalize_quotes(search_string)
    normalized_file = normalize_quotes(file_content)
    
    search_index = normalized_file.find(normalized_search)
    if search_index != -1:
        # Find the actual string in the file that matches
        return file_content[search_index:search_index + len(search_string)]
    
    return None


def preserve_quote_style(
    model_old_string: str,
    actual_old_string: str,
    model_new_string: str,
) -> str:
    """
    When old_string matched via quote normalization (curly quotes in file,
    straight quotes from model), apply the same curly quote style to new_string
    so the edit preserves the file's typography.
    
    Args:
        model_old_string: The old_string from the model (normalized)
        actual_old_string: The actual string found in the file (may have curly quotes)
        model_new_string: The new_string from the model (normalized)
        
    Returns:
        new_string with quotes adjusted to match file style
    """
    if model_old_string == actual_old_string:
        return model_new_string
    
    # Build a mapping of straight to curly quotes found in actual_old_string
    result = model_new_string
    
    # Simple replacement: if actual has curly quotes, apply them to result
    if LEFT_SINGLE_CURLY_QUOTE in actual_old_string and "'" in model_new_string:
        result = result.replace("'", LEFT_SINGLE_CURLY_QUOTE)
    if RIGHT_SINGLE_CURLY_QUOTE in actual_old_string and "'" in model_new_string:
        result = result.replace("'", RIGHT_SINGLE_CURLY_QUOTE)
    if LEFT_DOUBLE_CURLY_QUOTE in actual_old_string and '"' in model_new_string:
        result = result.replace('"', LEFT_DOUBLE_CURLY_QUOTE)
    if RIGHT_DOUBLE_CURLY_QUOTE in actual_old_string and '"' in model_new_string:
        result = result.replace('"', RIGHT_DOUBLE_CURLY_QUOTE)
    
    return result


def apply_edit_to_file(
    original_content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """
    Apply an edit to file content.
    
    Args:
        original_content: The original file content
        old_string: The string to replace
        new_string: The replacement string
        replace_all: Whether to replace all occurrences
        
    Returns:
        The updated file content
    """
    if replace_all:
        result = original_content.replace(old_string, new_string)
    else:
        result = original_content.replace(old_string, new_string, 1)
    
    # Special case: if new_string is empty and we're removing a line,
    # also remove the trailing newline if present
    if new_string == '' and not old_string.endswith('\n'):
        if old_string + '\n' in original_content:
            if replace_all:
                result = original_content.replace(old_string + '\n', new_string)
            else:
                result = original_content.replace(old_string + '\n', new_string, 1)
    
    return result


def get_unified_diff(
    file_path: str,
    original_content: str,
    updated_content: str,
) -> str:
    """
    Generate a unified diff between original and updated content.
    
    Args:
        file_path: The file path (for display in diff header)
        original_content: The original content
        updated_content: The updated content
        
    Returns:
        Unified diff as a string
    """
    original_lines = original_content.splitlines(keepends=True)
    updated_lines = updated_content.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        original_lines,
        updated_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm='',
    )
    
    return ''.join(diff)


def get_patch_for_edit(
    file_path: str,
    file_contents: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> tuple[str, str]:
    """
    Apply an edit and generate a patch showing the changes.
    
    Args:
        file_path: The file path
        file_contents: The original file contents
        old_string: The string to replace
        new_string: The replacement string
        replace_all: Whether to replace all occurrences
        
    Returns:
        Tuple of (patch_string, updated_file_content)
    """
    updated_file = apply_edit_to_file(file_contents, old_string, new_string, replace_all)
    
    if updated_file == file_contents:
        raise ValueError("Original and edited file match exactly. Failed to apply edit.")
    
    patch = get_unified_diff(file_path, file_contents, updated_file)
    
    return patch, updated_file


def count_matches(content: str, search_string: str) -> int:
    """
    Count how many times a string appears in content.
    
    Args:
        content: The content to search in
        search_string: The string to count
        
    Returns:
        Number of matches
    """
    return content.count(search_string)


def get_file_modification_time(file_path: str) -> float:
    """Get file modification time as Unix timestamp."""
    from pathlib import Path
    import os
    
    path = Path(file_path)
    if not path.exists():
        return 0.0
    return os.path.getmtime(str(path))


def detect_encoding(data: bytes) -> str:
    """Detect file encoding from byte data."""
    # Check for UTF-16 LE BOM
    if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xFE:
        return "utf-16-le"
    
    # Check for UTF-8 BOM
    if len(data) >= 3 and data[0] == 0xEF and data[1] == 0xBB and data[2] == 0xBF:
        return "utf-8-sig"
    
    # Default to UTF-8
    return "utf-8"


def read_file_for_edit(file_path: str) -> tuple[str, bool, str]:
    """
    Read file for editing, detecting encoding.
    
    Returns:
        Tuple of (content, file_exists, encoding)
    """
    from pathlib import Path
    
    path = Path(file_path)
    if not path.exists():
        return ("", False, "utf-8")
    
    try:
        raw_data = path.read_bytes()
        encoding = detect_encoding(raw_data)
        content = raw_data.decode(encoding, errors="replace")
        # Normalize line endings to \n
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        return (content, True, encoding)
    except Exception:
        return ("", False, "utf-8")


def suggest_similar_file(file_path: str) -> Optional[str]:
    """Suggest a similar file if the requested one doesn't exist."""
    from pathlib import Path
    
    path = Path(file_path)
    if not path.parent.exists():
        return None
    
    # Try files with different extensions
    stem = path.stem
    parent = path.parent
    
    for item in parent.iterdir():
        if item.stem == stem and item.suffix != path.suffix:
            return str(item)
    
    return None
