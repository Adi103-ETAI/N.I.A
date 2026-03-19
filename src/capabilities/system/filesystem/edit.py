import os
from pathlib import Path

from .utils import (
    strip_bom, detect_line_ending, normalize_to_lf, 
    fuzzy_find_text, normalize_for_fuzzy_match, 
    restore_line_endings, generate_diff_string
)

async def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Edit a file by replacing exact text. The old_text must match 
    exactly (including whitespace). Use this for precise, surgical edits.
    
    Args:
        path: Path to the file to edit (relative or absolute)
        old_text: Exact text to find and replace
        new_text: New text to replace the old text with
    """
    cwd = os.getcwd()
    abs_path = Path(cwd) / path
    if not abs_path.is_absolute():
        abs_path = abs_path.resolve()
        
    if not abs_path.exists():
        return f"Error: File not found: {path}"

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
    except Exception as e:
        return f"Error reading file for edit: {e}"

    bom, content = strip_bom(raw_content)

    original_ending = detect_line_ending(content)
    normalized_content = normalize_to_lf(content)
    normalized_old_text = normalize_to_lf(old_text)
    normalized_new_text = normalize_to_lf(new_text)

    match_result = fuzzy_find_text(normalized_content, normalized_old_text)

    if not match_result["found"]:
        return (
            f"Error: Could not find the exact text in {path}. "
            f"The old_text must match exactly including all whitespace and newlines."
        )

    fuzzy_content = normalize_for_fuzzy_match(normalized_content)
    fuzzy_old_text = normalize_for_fuzzy_match(normalized_old_text)
    occurrences = fuzzy_content.count(fuzzy_old_text)

    if occurrences > 1:
        return (
            f"Error: Found {occurrences} occurrences of the text in {path}. "
            f"The text must be unique. Please provide more context to make it unique."
        )

    base_content = match_result["contentForReplacement"]
    index = match_result["index"]
    match_length = match_result["matchLength"]
    
    new_filtered_content = (
        base_content[:index] +
        normalized_new_text +
        base_content[index + match_length:]
    )

    if base_content == new_filtered_content:
        return f"Warning: No changes made to {path}. The replacement produced identical content."

    final_content = bom + restore_line_endings(new_filtered_content, original_ending)
    
    try:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(final_content)
    except Exception as e:
        return f"Error writing edit to file: {e}"

    diff_str, first_changed_line = generate_diff_string(base_content, new_filtered_content)
    
    return f"Successfully replaced text in {path}.\\n\\nDiff:\\n{diff_str}"
