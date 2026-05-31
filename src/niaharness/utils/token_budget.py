"""Token budget utilities.

Provides parsing of token budget expressions like '+500k' or 'use 2M tokens'.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Shorthand (+500k) anchored to start/end to avoid false positives
_SHORTHAND_START_RE = re.compile(r"^\s*\+(\d+(?:\.\d+)?)\s*(k|m|b)\b", re.IGNORECASE)
_SHORTHAND_END_RE = re.compile(r"\s\+(\d+(?:\.\d+)?)\s*(k|m|b)\s*[.!?]?\s*$", re.IGNORECASE)
_VERBOSE_RE = re.compile(
    r"\b(?:use|spend)\s+(\d+(?:\.\d+)?)\s*(k|m|b)\s*tokens?\b", re.IGNORECASE
)

_MULTIPLIERS = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
}


def _parse_budget_match(value: str, suffix: str) -> int:
    """Convert a numeric value with suffix to an integer token count."""
    return int(float(value) * _MULTIPLIERS[suffix.lower()])


def parse_token_budget(text: str) -> Optional[int]:
    """Parse a token budget expression from text.

    Supports formats like:
    - '+500k' (shorthand at start)
    - '+500k.' (shorthand at end)
    - 'use 2M tokens' (verbose)

    Returns the token count or None if no budget expression found.
    """
    start_match = _SHORTHAND_START_RE.search(text)
    if start_match:
        return _parse_budget_match(start_match.group(1), start_match.group(2))

    end_match = _SHORTHAND_END_RE.search(text)
    if end_match:
        return _parse_budget_match(end_match.group(1), end_match.group(2))

    verbose_match = _VERBOSE_RE.search(text)
    if verbose_match:
        return _parse_budget_match(verbose_match.group(1), verbose_match.group(2))

    return None


def find_token_budget_positions(text: str) -> List[Tuple[int, int]]:
    """Find all token budget expression positions in text.

    Returns a list of (start, end) tuples for each found expression.
    """
    positions: list[tuple[int, int]] = []

    start_match = _SHORTHAND_START_RE.search(text)
    if start_match:
        offset = start_match.start() + len(start_match.group(0)) - len(
            start_match.group(0).lstrip()
        )
        positions.append((offset, start_match.end()))

    end_match = _SHORTHAND_END_RE.search(text)
    if end_match:
        # +1 to skip the leading whitespace captured by the regex
        end_start = end_match.start() + 1
        already_covered = any(
            end_start >= p[0] and end_start < p[1] for p in positions
        )
        if not already_covered:
            positions.append((end_start, end_match.end()))

    for match in _VERBOSE_RE.finditer(text):
        positions.append((match.start(), match.end()))

    return positions


def get_budget_continuation_message(
    pct: int, turn_tokens: int, budget: int
) -> str:
    """Generate a continuation message for token budget exhaustion."""
    return (
        f"Stopped at {pct}% of token target ({turn_tokens:,} / {budget:,}). "
        "Keep working — do not summarize."
    )
