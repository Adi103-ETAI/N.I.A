from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

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
    if bytes_num < 1024 * 1024:
        return f"{bytes_num / 1024:.1f}KB"
    return f"{bytes_num / (1024 * 1024):.1f}MB"


def truncate_head(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult:
    """Truncate content from the head (keep first N lines/bytes)."""
    encoded = content.encode("utf-8")
    total_bytes = len(encoded)
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncatedBy=None,
            totalLines=total_lines,
            totalBytes=total_bytes,
            outputLines=total_lines,
            outputBytes=total_bytes,
            lastLinePartial=False,
            firstLineExceedsLimit=False,
            maxLines=max_lines,
            maxBytes=max_bytes,
        )

    first_line_bytes = len(lines[0].encode("utf-8"))
    if first_line_bytes > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncatedBy="bytes",
            totalLines=total_lines,
            totalBytes=total_bytes,
            outputLines=0,
            outputBytes=0,
            lastLinePartial=False,
            firstLineExceedsLimit=True,
            maxLines=max_lines,
            maxBytes=max_bytes,
        )

    output_lines = []
    output_bytes_count = 0
    truncated_by: Literal["lines", "bytes"] = "lines"

    for i in range(min(len(lines), max_lines)):
        line = lines[i]
        line_bytes = len(line.encode("utf-8")) + (1 if i > 0 else 0)

        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break

        output_lines.append(line)
        output_bytes_count += line_bytes

    if len(output_lines) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines)
    final_output_bytes = len(output_content.encode("utf-8"))

    return TruncationResult(
        content=output_content,
        truncated=True,
        truncatedBy=truncated_by,
        totalLines=total_lines,
        totalBytes=total_bytes,
        outputLines=len(output_lines),
        outputBytes=final_output_bytes,
        lastLinePartial=False,
        firstLineExceedsLimit=False,
        maxLines=max_lines,
        maxBytes=max_bytes,
    )


def _truncate_string_to_bytes_from_end(s: str, max_bytes: int) -> str:
    buf = s.encode("utf-8")
    if len(buf) <= max_bytes:
        return s
    start = len(buf) - max_bytes
    while start < len(buf) and (buf[start] & 0xC0) == 0x80:
        start += 1
    return buf[start:].decode("utf-8", errors="replace")


def truncate_tail(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult:
    """Truncate content from the tail (keep last N lines/bytes)."""
    encoded = content.encode("utf-8")
    total_bytes = len(encoded)
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncatedBy=None,
            totalLines=total_lines,
            totalBytes=total_bytes,
            outputLines=total_lines,
            outputBytes=total_bytes,
            lastLinePartial=False,
            firstLineExceedsLimit=False,
            maxLines=max_lines,
            maxBytes=max_bytes,
        )

    output_lines = []
    output_bytes_count = 0
    truncated_by: Literal["lines", "bytes"] = "lines"
    last_line_partial = False

    for i in range(len(lines) - 1, -1, -1):
        if len(output_lines) >= max_lines:
            break

        line = lines[i]
        line_bytes = len(line.encode("utf-8")) + (1 if len(output_lines) > 0 else 0)

        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            if len(output_lines) == 0:
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines.insert(0, truncated_line)
                output_bytes_count = len(truncated_line.encode("utf-8"))
                last_line_partial = True
            break

        output_lines.insert(0, line)
        output_bytes_count += line_bytes

    if len(output_lines) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines)
    final_output_bytes = len(output_content.encode("utf-8"))

    return TruncationResult(
        content=output_content,
        truncated=True,
        truncatedBy=truncated_by,
        totalLines=total_lines,
        totalBytes=total_bytes,
        outputLines=len(output_lines),
        outputBytes=final_output_bytes,
        lastLinePartial=last_line_partial,
        firstLineExceedsLimit=False,
        maxLines=max_lines,
        maxBytes=max_bytes,
    )


def resolve_to_cwd(p: str, cwd: str = ".") -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((Path(cwd) / path).resolve())


def resolve_read_path(p: str, cwd: str = ".") -> str:
    return resolve_to_cwd(p, cwd)


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


def strip_bom(content: str) -> Tuple[str, str]:
    if content.startswith("\uFEFF"):
        return "\uFEFF", content[1:]
    return "", content


def generate_diff_string(old_content: str, new_content: str, context_lines: int = 4) -> Tuple[str, Optional[int]]:
    """Generate a unified diff string with line numbers and context."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    output = []
    max_line_num = max(len(old_lines), len(new_lines))
    line_num_width = len(str(max_line_num))
    matcher = difflib.SequenceMatcher(None, [l.rstrip("\n") for l in old_lines], [l.rstrip("\n") for l in new_lines])

    first_changed_line = None
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            lines_chunk = [l.rstrip("\n") for l in old_lines[i1:i2]]
            is_start, is_end = (i1 == 0), (i2 == len(old_lines))
            if is_start and is_end:
                continue

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
        elif tag in ("replace", "insert", "delete"):
            if first_changed_line is None:
                first_changed_line = j1 + 1
            if tag in ("replace", "delete"):
                for idx, line in enumerate([l.rstrip("\n") for l in old_lines[i1:i2]]):
                    output.append(f"-{str(i1 + idx + 1).rjust(line_num_width)} {line}")
            if tag in ("replace", "insert"):
                for idx, line in enumerate([l.rstrip("\n") for l in new_lines[j1:j2]]):
                    output.append(f"+{str(j1 + idx + 1).rjust(line_num_width)} {line}")

    return "\n".join(output), first_changed_line


__all__ = [
    "DEFAULT_MAX_LINES",
    "DEFAULT_MAX_BYTES",
    "GREP_MAX_LINE_LENGTH",
    "TruncationResult",
    "format_size",
    "truncate_head",
    "truncate_tail",
    "resolve_to_cwd",
    "resolve_read_path",
    "detect_line_ending",
    "normalize_to_lf",
    "restore_line_endings",
    "strip_bom",
    "generate_diff_string",
]
