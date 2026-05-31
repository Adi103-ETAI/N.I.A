"""File reading utilities.

Provides functions for reading files with encoding detection and line ending
detection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LineEnding(str, Enum):
    """Line ending types."""

    CRLF = "CRLF"
    LF = "LF"


class FileEncoding(str, Enum):
    """Detected file encodings."""

    UTF8 = "utf-8"
    UTF16LE = "utf-16-le"
    ASCII = "ascii"


@dataclass(frozen=True)
class FileMetadata:
    """Result of reading a file with metadata."""

    content: str
    encoding: FileEncoding
    line_endings: LineEnding


def detect_encoding(file_path: str) -> FileEncoding:
    """Detect file encoding by reading the BOM and content.

    Reads the first 4096 bytes and checks for BOM markers.
    Defaults to UTF-8 for empty files and non-BOM files.
    """
    try:
        with open(file_path, "rb") as f:
            buffer = f.read(4096)
    except (OSError, IOError):
        return FileEncoding.UTF8

    if len(buffer) == 0:
        return FileEncoding.UTF8

    # UTF-16 LE BOM
    if len(buffer) >= 2 and buffer[0] == 0xFF and buffer[1] == 0xFE:
        return FileEncoding.UTF16LE

    # UTF-8 BOM
    if (
        len(buffer) >= 3
        and buffer[0] == 0xEF
        and buffer[1] == 0xBB
        and buffer[2] == 0xBF
    ):
        return FileEncoding.UTF8

    return FileEncoding.UTF8


def detect_line_endings(content: str) -> LineEnding:
    """Detect line ending style from file content.

    Counts CRLF vs LF occurrences and returns the predominant style.
    """
    crlf_count = 0
    lf_count = 0

    for i, char in enumerate(content):
        if char == "\n":
            if i > 0 and content[i - 1] == "\r":
                crlf_count += 1
            else:
                lf_count += 1

    return LineEnding.CRLF if crlf_count > lf_count else LineEnding.LF


def read_file_with_metadata(file_path: str) -> FileMetadata:
    """Read a file and return content with encoding and line ending metadata.

    This performs a single read to extract all metadata efficiently.
    """
    encoding = detect_encoding(file_path)

    try:
        with open(file_path, "r", encoding=encoding.value) as f:
            raw = f.read()
    except (OSError, IOError) as e:
        raise FileNotFoundError(f"Cannot read file: {file_path}") from e

    # Detect line endings from the first 4096 chars
    sample = raw[:4096]
    line_endings = detect_line_endings(sample)

    # Normalize line endings to LF
    content = raw.replace("\r\n", "\n")

    return FileMetadata(content=content, encoding=encoding, line_endings=line_endings)


def read_file(file_path: str) -> str:
    """Read a file and return its content with normalized line endings."""
    return read_file_with_metadata(file_path).content


def is_binary_content(data: bytes, sample_size: int = 8192) -> bool:
    """Heuristically detect if content is binary.

    Checks for null bytes and high ratio of non-printable characters.
    """
    sample = data[:sample_size]
    if not sample:
        return False

    # Check for null bytes (strong binary indicator)
    if b"\x00" in sample:
        return True

    # Count non-printable characters (excluding common whitespace)
    non_printable = sum(
        1
        for b in sample
        if b < 32 and b not in (9, 10, 13)  # tab, LF, CR
    )

    # If more than 10% non-printable, consider binary
    return (non_printable / len(sample)) > 0.1


def has_binary_extension(file_path: str) -> bool:
    """Check if a file has a known binary extension."""
    binary_extensions = {
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".o",
        ".a",
        ".pyc",
        ".pyo",
        ".class",
        ".jar",
        ".war",
        ".ear",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".webp",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".flac",
        ".wav",
        ".ogg",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
    }
    _, ext = os.path.splitext(file_path.lower())
    return ext in binary_extensions
