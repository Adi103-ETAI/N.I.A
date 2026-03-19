import os
import difflib
import re
from dataclasses import dataclass
from typing import Optional, Tuple, TypedDict, Literal
from pathlib import Path

# =============================================================================
# Truncation and Formatting Limits
# =============================================================================
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB
GREP_MAX_LINE_LENGTH = 500

@dataclass
class TruncationResult:
    content: str
    truncated: bool
    truncatedBy: Optional[Literal["lines", "bytes"]]
    totalLines: int
    totalBytes: int
    outputLines: int
    outputBytes: int
    lastLinePartial: bool
    firstLineExceedsLimit: bool
    maxLines: int
    maxBytes: int

def format_size(bytes_num: int) -> str:
    if bytes_num < 1024:
        return f"{bytes_num}B"
    elif bytes_num < 1024 * 1024:
        return f"{bytes_num / 1024:.1f}KB"
    else:
        return f"{bytes_num / (1024 * 1024):.1f}MB"

def truncate_head(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult:
    """Truncate content from the head (keep first N lines/bytes)."""
    encoded = content.encode('utf-8')
    total_bytes = len(encoded)
    lines = content.split('\n')
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content, truncated=False, truncatedBy=None,
            totalLines=total_lines, totalBytes=total_bytes,
            outputLines=total_lines, outputBytes=total_bytes,
            lastLinePartial=False, firstLineExceedsLimit=False,
            maxLines=max_lines, maxBytes=max_bytes
        )

    first_line_bytes = len(lines[0].encode('utf-8'))
    if first_line_bytes > max_bytes:
        return TruncationResult(
            content="", truncated=True, truncatedBy="bytes",
            totalLines=total_lines, totalBytes=total_bytes,
            outputLines=0, outputBytes=0,
            lastLinePartial=False, firstLineExceedsLimit=True,
            maxLines=max_lines, maxBytes=max_bytes
        )

    output_lines = []
    output_bytes_count = 0
    truncated_by: Literal["lines", "bytes"] = "lines"

    for i in range(min(len(lines), max_lines)):
        line = lines[i]
        line_bytes = len(line.encode('utf-8')) + (1 if i > 0 else 0)

        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break

        output_lines.append(line)
        output_bytes_count += line_bytes

    if len(output_lines) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines)
    final_output_bytes = len(output_content.encode('utf-8'))

    return TruncationResult(
        content=output_content, truncated=True, truncatedBy=truncated_by,
        totalLines=total_lines, totalBytes=total_bytes,
        outputLines=len(output_lines), outputBytes=final_output_bytes,
        lastLinePartial=False, firstLineExceedsLimit=False,
        maxLines=max_lines, maxBytes=max_bytes
    )

def _truncate_string_to_bytes_from_end(s: str, max_bytes: int) -> str:
    buf = s.encode('utf-8')
    if len(buf) <= max_bytes:
        return s
    start = len(buf) - max_bytes
    while start < len(buf) and (buf[start] & 0xc0) == 0x80:
        start += 1
    return buf[start:].decode('utf-8', errors='replace')

def truncate_tail(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult:
    """Truncate content from the tail (keep last N lines/bytes)."""
    encoded = content.encode('utf-8')
    total_bytes = len(encoded)
    lines = content.split('\n')
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content, truncated=False, truncatedBy=None,
            totalLines=total_lines, totalBytes=total_bytes,
            outputLines=total_lines, outputBytes=total_bytes,
            lastLinePartial=False, firstLineExceedsLimit=False,
            maxLines=max_lines, maxBytes=max_bytes
        )

    output_lines = []
    output_bytes_count = 0
    truncated_by: Literal["lines", "bytes"] = "lines"
    last_line_partial = False

    for i in range(len(lines) - 1, -1, -1):
        if len(output_lines) >= max_lines:
            break
            
        line = lines[i]
        line_bytes = len(line.encode('utf-8')) + (1 if len(output_lines) > 0 else 0)

        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            if len(output_lines) == 0:
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines.insert(0, truncated_line)
                output_bytes_count = len(truncated_line.encode('utf-8'))
                last_line_partial = True
            break

        output_lines.insert(0, line)
        output_bytes_count += line_bytes

    if len(output_lines) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines)
    final_output_bytes = len(output_content.encode('utf-8'))

    return TruncationResult(
        content=output_content, truncated=True, truncatedBy=truncated_by,
        totalLines=total_lines, totalBytes=total_bytes,
        outputLines=len(output_lines), outputBytes=final_output_bytes,
        lastLinePartial=last_line_partial, firstLineExceedsLimit=False,
        maxLines=max_lines, maxBytes=max_bytes
    )

def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    """Truncate a single line to max characters."""
    if len(line) <= max_chars:
        return line, False
    return f"{line[:max_chars]}... [truncated]", True

# =============================================================================
# Path Utils
# =============================================================================
def resolve_to_cwd(p: str, cwd: str = ".") -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((Path(cwd) / path).resolve())

def resolve_read_path(p: str, cwd: str = ".") -> str:
    return resolve_to_cwd(p, cwd)

# =============================================================================
# Diff and Edit Utilities
# =============================================================================
def detect_line_ending(content: str) -> str:
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    if crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"

def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")

def restore_line_endings(text: str, ending: str) -> str:
    if ending == "\r\n":
        return text.replace("\n", "\r\n")
    return text

def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching."""
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r'[\u2018\u2019\u201A\u201B]', "'", text)
    text = re.sub(r'[\u201C\u201D\u201E\u201F]', '"', text)
    text = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]', "-", text)
    text = re.sub(r'[\u00A0\u2002-\u200A\u202F\u205F\u3000]', " ", text)
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
            "contentForReplacement": content
        }
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)
    if fuzzy_index == -1:
        return {
            "found": False, "index": -1, "matchLength": 0,
            "usedFuzzyMatch": False, "contentForReplacement": content
        }
    return {
        "found": True, "index": fuzzy_index, "matchLength": len(fuzzy_old_text),
        "usedFuzzyMatch": True, "contentForReplacement": fuzzy_content
    }

def strip_bom(content: str) -> Tuple[str, str]:
    if content.startswith("\uFEFF"):
        return "\uFEFF", content[1:]
    return "", content

def generate_diff_string(old_content: str, new_content: str, context_lines: int = 4) -> Tuple[str, Optional[int]]:
    """Generate a unified diff string with line numbers and context."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    diff_gen = list(difflib.unified_diff(
        old_lines, new_lines, n=context_lines,
        fromfile='old', tofile='new', lineterm='\n'
    ))
    
    output = []
    max_line_num = max(len(old_lines), len(new_lines))
    line_num_width = len(str(max_line_num))
    s = difflib.SequenceMatcher(None, [l.rstrip('\n') for l in old_lines], [l.rstrip('\n') for l in new_lines])
    
    first_changed_line = None
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            lines_chunk = [l.rstrip('\n') for l in old_lines[i1:i2]]
            is_start, is_end = (i1 == 0), (i2 == len(old_lines))
            if is_start and is_end: continue
                
            if len(lines_chunk) > context_lines * 2 and not (is_start or is_end):
                for idx, line in enumerate(lines_chunk[:context_lines]):
                    output.append(f" {str(i1 + idx + 1).rjust(line_num_width)} {line}")
                output.append(f" {''.rjust(line_num_width)} ...")
                for idx, line in enumerate(lines_chunk[-context_lines:]):
                    output.append(f" {str(i2 - context_lines + idx + 1).rjust(line_num_width)} {line}")
            else:
                if is_start and len(lines_chunk) > context_lines:
                    output.append(f" {''.rjust(line_num_width)} ...")
                    for idx, line in enumerate(lines_chunk[-context_lines:]):
                        output.append(f" {str(i2 - context_lines + idx + 1).rjust(line_num_width)} {line}")
                elif is_end and len(lines_chunk) > context_lines:
                    for idx, line in enumerate(lines_chunk[:context_lines]):
                        output.append(f" {str(i1 + idx + 1).rjust(line_num_width)} {line}")
                    output.append(f" {''.rjust(line_num_width)} ...")
                else:
                    for idx, line in enumerate(lines_chunk):
                        output.append(f" {str(i1 + idx + 1).rjust(line_num_width)} {line}")
        elif tag in ('replace', 'insert', 'delete'):
            if first_changed_line is None: first_changed_line = j1 + 1
            if tag in ('replace', 'delete'):
                for idx, line in enumerate([l.rstrip('\n') for l in old_lines[i1:i2]]):
                    output.append(f"-{str(i1 + idx + 1).rjust(line_num_width)} {line}")
            if tag in ('replace', 'insert'):
                for idx, line in enumerate([l.rstrip('\n') for l in new_lines[j1:j2]]):
                    output.append(f"+{str(j1 + idx + 1).rjust(line_num_width)} {line}")
                    
    return "\n".join(output), first_changed_line
