"""TARA Graph Nodes — JSON Utility Layer.

Provides robust JSON parsing helpers for processing messy LLM output.
These utilities are used by the reasoner node to parse Llama 3.1
native tool call format when the ChatNVIDIA SDK doesn't handle it.

Functions:
    _sanitize_json_string  — clean up markdown, comments, trailing commas
    _extract_json_objects  — bracket-match JSON objects from raw text
    _parse_llama_tool_calls — parse <|python_tag|> tool call format
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from src.core.logger import setup_logger

logger = setup_logger("TARA.Nodes.Utils")


# =============================================================================
# JSON Sanitization
# =============================================================================

def _sanitize_json_string(raw_json: str) -> str:
    """Sanitize messy LLM JSON output before parsing.

    Handles common LLM output issues:
    - Markdown code blocks (```json ... ```)
    - Trailing commas in objects/arrays
    - JavaScript-style comments (// and /* */)
    - Leading/trailing whitespace
    - Single quotes instead of double quotes

    Args:
        raw_json: Potentially malformed JSON string.

    Returns:
        Sanitized JSON string (best effort).
    """
    text = raw_json.strip()

    # 1. Remove markdown code blocks
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # 2. Remove JavaScript-style single-line comments (// ...)
    text = re.sub(r'//[^\n]*', '', text)

    # 3. Remove JavaScript-style multi-line comments (/* ... */)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # 4. Fix trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # 5. Fix single quotes (best effort — won't work for all cases)
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')

    return text.strip()


def _extract_json_objects(text: str) -> List[str]:
    """Extract all JSON objects from text using bracket matching.

    More robust than regex — handles nested objects, whitespace, and newlines.
    Applies sanitization before extraction for reliable parsing.

    Args:
        text: Raw text containing potential JSON objects.

    Returns:
        List of sanitized JSON string candidates.
    """
    json_objects = []
    depth = 0
    start_idx = None

    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start_idx is not None:
                raw_json = text[start_idx:i + 1]
                sanitized = _sanitize_json_string(raw_json)
                json_objects.append(sanitized)
                start_idx = None

    return json_objects


# =============================================================================
# Llama 3.1 Tool Call Parser
# =============================================================================

def _parse_llama_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse Llama 3.1's native tool call format when ChatNVIDIA doesn't.

    Uses bracket-matching JSON extraction instead of fragile regex.

    Llama 3.1 outputs tool calls in various formats::

        <|python_tag|><function>tool_name</function>{"arg": "value"}
        <|python_tag|>tool_name.call({"arg": "value"})
        <|python_tag|>tool_name {"arg": "value"}

    Args:
        content: Raw LLM response string.

    Returns:
        List of tool call dicts with 'name', 'args', and 'id' keys.
        Returns empty list [] on any parsing failure (safe fallback).
    """
    tool_calls = []

    if not content or "<|python_tag|>" not in content:
        return tool_calls

    logger.debug("[ROBUST PARSER] Detected <|python_tag|> in response, parsing...")

    json_candidates = _extract_json_objects(content)
    if not json_candidates:
        logger.warning("[ROBUST PARSER] No JSON objects found in response")
        return tool_calls

    for i, json_str in enumerate(json_candidates):
        try:
            args = json.loads(json_str)

            json_pos = content.find(json_str)
            prefix = content[:json_pos].strip()
            func_name = None

            # Strategy 1: <function>name</function> pattern
            if "</function>" in prefix:
                func_match = re.search(r'<function>(\w+)</function>\s*$', prefix)
                if func_match:
                    func_name = func_match.group(1)

            # Strategy 2: name.call( or name( pattern
            if not func_name:
                call_match = re.search(r'(\w+)(?:\.call)?\s*\(\s*$', prefix)
                if call_match:
                    func_name = call_match.group(1)

            # Strategy 3: bare word before the JSON
            if not func_name:
                word_match = re.search(r'(\w+)\s*$', prefix)
                if word_match:
                    func_name = word_match.group(1)

            # Strategy 4: 'name' field inside the JSON itself
            if not func_name and isinstance(args, dict) and 'name' in args:
                func_name = args.pop('name')

            if func_name:
                tool_calls.append({
                    "name": func_name,
                    "args": args if isinstance(args, dict) else {"value": args},
                    "id": f"call_{func_name}_{i}",
                })
                logger.info(f"[ROBUST PARSER] Extracted: {func_name}({args})")
            else:
                logger.warning(f"[ROBUST PARSER] Found JSON but no function name: {json_str[:50]}...")

        except json.JSONDecodeError as e:
            logger.debug(f"[ROBUST PARSER] Invalid JSON: {e}")
        except Exception as e:
            logger.debug(f"[ROBUST PARSER] Parse error: {e}")

    return tool_calls


__all__ = [
    "_sanitize_json_string",
    "_extract_json_objects",
    "_parse_llama_tool_calls",
]
