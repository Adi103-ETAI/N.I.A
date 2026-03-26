from __future__ import annotations

import json
import re
from typing import Any, Dict, List, TypedDict

from src.core.logger import setup_logger
from src.core.utils.file_utils import GREP_MAX_LINE_LENGTH

logger = setup_logger("Core.Utils.Text")


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    """Truncate a single line to max characters."""
    if len(line) <= max_chars:
        return line, False
    return f"{line[:max_chars]}... [truncated]", True


def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching."""
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"[\u2018\u2019\u201A\u201B]", "'", text)
    text = re.sub(r"[\u201C\u201D\u201E\u201F]", '"', text)
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", text)
    text = re.sub(r"[\u00A0\u2002-\u200A\u202F\u205F\u3000]", " ", text)
    return text


class FuzzyMatchResult(TypedDict):
    found: bool
    index: int
    matchLength: int
    usedFuzzyMatch: bool
    contentForReplacement: str


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    exact_index = content.find(old_text)
    if exact_index != -1:
        return {
            "found": True,
            "index": exact_index,
            "matchLength": len(old_text),
            "usedFuzzyMatch": False,
            "contentForReplacement": content,
        }
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)
    if fuzzy_index == -1:
        return {
            "found": False,
            "index": -1,
            "matchLength": 0,
            "usedFuzzyMatch": False,
            "contentForReplacement": content,
        }
    return {
        "found": True,
        "index": fuzzy_index,
        "matchLength": len(fuzzy_old_text),
        "usedFuzzyMatch": True,
        "contentForReplacement": fuzzy_content,
    }


def _sanitize_json_string(raw_json: str) -> str:
    """Sanitize messy LLM JSON output before parsing."""
    text = raw_json.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    return text.strip()


def _extract_json_objects(text: str) -> List[str]:
    """Extract all JSON objects from text using bracket matching."""
    json_objects = []
    depth = 0
    start_idx = None

    for i, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start_idx is not None:
                raw_json = text[start_idx : i + 1]
                sanitized = _sanitize_json_string(raw_json)
                json_objects.append(sanitized)
                start_idx = None

    return json_objects


def _parse_llama_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse Llama 3.1's native tool call format when ChatNVIDIA doesn't."""
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

            if "</function>" in prefix:
                func_match = re.search(r"<function>(\w+)</function>\s*$", prefix)
                if func_match:
                    func_name = func_match.group(1)

            if not func_name:
                call_match = re.search(r"(\w+)(?:\.call)?\s*\(\s*$", prefix)
                if call_match:
                    func_name = call_match.group(1)

            if not func_name:
                word_match = re.search(r"(\w+)\s*$", prefix)
                if word_match:
                    func_name = word_match.group(1)

            if not func_name and isinstance(args, dict) and "name" in args:
                func_name = args.pop("name")

            if func_name:
                tool_calls.append(
                    {
                        "name": func_name,
                        "args": args if isinstance(args, dict) else {"value": args},
                        "id": f"call_{func_name}_{i}",
                    }
                )
                logger.info(f"[ROBUST PARSER] Extracted: {func_name}({args})")
            else:
                logger.warning(f"[ROBUST PARSER] Found JSON but no function name: {json_str[:50]}...")

        except json.JSONDecodeError as e:
            logger.debug(f"[ROBUST PARSER] Invalid JSON: {e}")
        except Exception as e:
            logger.debug(f"[ROBUST PARSER] Parse error: {e}")

    return tool_calls


__all__ = [
    "truncate_line",
    "FuzzyMatchResult",
    "normalize_for_fuzzy_match",
    "fuzzy_find_text",
    "_sanitize_json_string",
    "_extract_json_objects",
    "_parse_llama_tool_calls",
]
