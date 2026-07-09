"""Fuzzy string matching for file-edit tools.

Ported from Hermes Agent's tools/fuzzy_match.py (950 LOC), scoped to
what NIA's FileEditTool needs: when an exact old_string match fails,
find the closest match in the file and suggest it.

Uses a difflib-based approach with word-level tokenization for better
matching of code edits where whitespace or minor tyops differ.

Usage::

    from niaharness.tools.fuzzy_match import find_best_match

    result = find_best_match("def helloword()", "def helloworld():\\n    pass")
    # → Match(score=0.95, start=0, end=19, suggestion="def helloworld():")
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum similarity ratio to consider a match (0.0 to 1.0).
MIN_MATCH_RATIO = 0.6
# Max number of suggestions to return.
MAX_SUGGESTIONS = 3


@dataclass
class FuzzyMatch:
    """A fuzzy match result."""

    score: float  # 0.0 to 1.0
    start: int  # Character offset in the file
    end: int  # Character offset (exclusive)
    text: str  # The matched text
    suggestion: str  # The closest matching text from the file


def find_best_match(
    needle: str,
    haystack: str,
    *,
    min_ratio: float = MIN_MATCH_RATIO,
) -> Optional[FuzzyMatch]:
    """Find the best fuzzy match for needle in haystack.

    Uses SequenceMatcher to find the closest substring. Returns None if
    no match exceeds min_ratio.

    Args:
        needle: The string to search for (typically old_string from edit).
        haystack: The file content to search in.
        min_ratio: Minimum similarity ratio (0.0 to 1.0).

    Returns:
        FuzzyMatch or None.
    """
    if not needle or not haystack:
        return None

    # Strategy: split haystack into lines, try matching needle against
    # sliding windows of consecutive lines.
    lines = haystack.splitlines(keepends=True)
    needle_lines = needle.splitlines(keepends=True)
    needle_count = len(needle_lines)

    if needle_count == 0:
        return None

    best_score = 0.0
    best_start = 0
    best_end = 0
    best_text = ""

    # Try windows of size needle_count ± 2 (to handle slight line-count mismatches).
    for window_size in range(max(1, needle_count - 2), needle_count + 3):
        if window_size > len(lines):
            break
        for i in range(len(lines) - window_size + 1):
            window = "".join(lines[i : i + window_size])
            # Compute similarity ratio.
            ratio = difflib.SequenceMatcher(None, needle, window).ratio()
            if ratio > best_score:
                best_score = ratio
                best_start = sum(len(lines[j]) for j in range(i))
                best_end = best_start + len(window)
                best_text = window

    if best_score < min_ratio:
        return None

    return FuzzyMatch(
        score=best_score,
        start=best_start,
        end=best_end,
        text=best_text,
        suggestion=best_text.strip(),
    )


def find_similar_files(
    target: str,
    candidates: list[str],
    *,
    min_ratio: float = MIN_MATCH_RATIO,
    max_results: int = MAX_SUGGESTIONS,
) -> list[tuple[str, float]]:
    """Find files with names similar to target.

    Used by FileEditTool/FileReadTool when a file doesn't exist — suggests
    files the user might have meant.

    Args:
        target: The file path that wasn't found.
        candidates: List of existing file paths to search.
        min_ratio: Minimum similarity ratio.
        max_results: Max number of suggestions.

    Returns:
        List of (path, score) tuples, sorted by score descending.
    """
    if not target or not candidates:
        return []

    # Use just the filename for matching (not the full path).
    target_name = target.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

    results: list[tuple[str, float]] = []
    for candidate in candidates:
        candidate_name = candidate.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        ratio = difflib.SequenceMatcher(None, target_name.lower(), candidate_name.lower()).ratio()
        if ratio >= min_ratio:
            results.append((candidate, ratio))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:max_results]


def suggest_correction(
    old_string: str,
    file_content: str,
    *,
    min_ratio: float = MIN_MATCH_RATIO,
) -> Optional[str]:
    """Suggest a corrected old_string that exists in the file.

    When FileEditTool's exact match fails, this function finds the closest
    match in the file and returns it as a suggestion for the user/model.

    Args:
        old_string: The string that wasn't found.
        file_content: The file content to search in.
        min_ratio: Minimum similarity ratio.

    Returns:
        The suggested replacement string, or None if no good match.
    """
    match = find_best_match(old_string, file_content, min_ratio=min_ratio)
    if match is None:
        return None
    return match.suggestion


__all__ = [
    "FuzzyMatch",
    "find_best_match",
    "find_similar_files",
    "suggest_correction",
]
