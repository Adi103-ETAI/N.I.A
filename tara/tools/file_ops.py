"""
MODULE: File System Operations (3-Tier Security Model)
VERSION: 2.5.2
STRICT SCOPE: Read, Write, List, Delete, Create Dir.
CONSTRAINTS: Atomic actions only. Explicit guards on destructive operations.

TARA 2.0 Atomic Tool Module.

Verification Logic (Trust But Verify):
    - All paths validated via `_validate_path()` before any operation.
    - `must_exist=True` parameter enforces pre-existence checks.
    - Path traversal attacks blocked (no `../` escapes).

Safety Guards:
    - delete_file(): Requires `confirm=True` parameter (safety lock).
    - delete_file(): Blocks wildcard patterns (no `*` or `?` allowed).
    - delete_file(): Uses send2trash when available (recoverable).
    - write_file(): Prevents accidental overwrite unless `overwrite=True`.

Security Architecture:
    TIER 1: Safe (Eyes) - list_dir, read_file, file_exists, write_file, append_file
    TIER 2: Controlled Mutation (Hands) - create_dir, move_file, copy_file, delete_file
    TIER 3: Metadata (Brain) - get_file_info, search_files

Exports:
    - list_dir, read_file, file_exists, write_file, append_file
    - create_dir, move_file, copy_file, delete_file
    - get_file_info, search_files
"""
from __future__ import annotations

import glob
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from core.logger import setup_logger

logger = setup_logger("TARA.Tools.FileOps")

# =============================================================================
# Optional Dependencies
# =============================================================================

try:
    from send2trash import send2trash
    _HAS_TRASH = True
except ImportError:
    _HAS_TRASH = False
    logger.warning("send2trash not available - deletes will be permanent")


# =============================================================================
# Security Helpers
# =============================================================================

def _validate_path(path: str, must_exist: bool = False) -> Path:
    """
    Validate and normalize a file path.
    
    Args:
        path: Path string to validate.
        must_exist: If True, raise FileNotFoundError if not exists.
        
    Returns:
        Normalized Path object.
        
    Raises:
        ValueError: If path is invalid.
        FileNotFoundError: If must_exist and path doesn't exist.
    """
    if not path or not path.strip():
        raise ValueError("Path cannot be empty")
    
    # Convert to Path object
    p = Path(path).resolve()
    
    # Security: Block parent traversal attempts
    try:
        p.relative_to(p.anchor)  # Must be under a valid root
    except ValueError:
        raise ValueError(f"Invalid path: {path}")
    
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    return p


def _has_wildcards(path: str) -> bool:
    """Check if path contains wildcard characters."""
    return "*" in path or "?" in path


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


# =============================================================================
# TIER 1: Safe Operations (The Eyes)
# =============================================================================

def list_dir(path: str) -> str:
    """
    List contents of a directory.
    
    TIER 1: Safe, read-only operation.
    
    Args:
        path: Absolute path to directory.
        
    Returns:
        Formatted list of files and directories.
        
    Example:
        >>> list_dir("C:/Users/user/Documents")
        "[DIR]  Projects\n[FILE] notes.txt\n..."
    """
    try:
        p = _validate_path(path, must_exist=True)
        
        if not p.is_dir():
            return f"❌ Not a directory: {path}"
        
        items = []
        for item in sorted(p.iterdir()):
            if item.is_dir():
                items.append(f"[DIR]  {item.name}")
            else:
                size = _format_size(item.stat().st_size)
                items.append(f"[FILE] {item.name} ({size})")
        
        if not items:
            return f"📂 {path} (empty)"
        
        return f"📂 {path}:\n" + "\n".join(items)
        
    except FileNotFoundError as e:
        return f"❌ {e}"
    except PermissionError:
        return f"❌ Permission denied: {path}"
    except Exception as e:
        return f"❌ Error: {e}"


def read_file(path: str, max_chars: int = 5000) -> str:
    """
    Read text file contents.
    
    TIER 1: Safe, read-only operation.
    
    Args:
        path: Path to text file.
        max_chars: Maximum characters to read (default: 5000).
        
    Returns:
        File contents or error message.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    try:
        p = _validate_path(path, must_exist=True)
        
        if not p.is_file():
            return f"❌ Not a file: {path}"
        
        # Read with encoding detection fallback
        encodings = ["utf-8", "latin-1", "cp1252"]
        content = None
        
        for encoding in encodings:
            try:
                with open(p, "r", encoding=encoding) as f:
                    content = f.read(max_chars + 100)  # Read slightly more to detect truncation
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            return f"❌ Could not decode file (binary?): {path}"
        
        # Truncate if needed
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n...[TRUNCATED]"
        
        return f"📄 {p.name}:\n\n{content}"
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except PermissionError:
        return f"❌ Permission denied: {path}"
    except Exception as e:
        return f"❌ Error reading file: {e}"


def file_exists(path: str) -> bool:
    """
    Check if a file or directory exists.
    
    TIER 1: Safe, read-only operation.
    
    Args:
        path: Path to check.
        
    Returns:
        True if exists, False otherwise.
    """
    try:
        p = _validate_path(path)
        return p.exists()
    except Exception:
        return False


def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """
    Write content to a file.
    
    TIER 1: Controlled write with overwrite guard.
    
    Args:
        path: Path to file.
        content: Text content to write.
        overwrite: If False, raises error if file exists.
        
    Returns:
        Success message.
        
    Raises:
        FileExistsError: If file exists and overwrite is False.
    """
    try:
        p = _validate_path(path)
        
        # Safety guard: prevent accidental overwrite
        if p.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {path}. Set overwrite=True to replace."
            )
        
        # Create parent directories if needed
        p.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Wrote {len(content)} chars to {path}")
        return f"✅ Wrote {len(content)} characters to {p.name}"
        
    except FileExistsError:
        raise
    except PermissionError:
        return f"❌ Permission denied: {path}"
    except Exception as e:
        return f"❌ Error writing file: {e}"


def append_file(path: str, content: str) -> str:
    """
    Append content to an existing file.
    
    TIER 1: Safe append (file must exist).
    
    Args:
        path: Path to existing file.
        content: Text content to append.
        
    Returns:
        Success message.
        
    Raises:
        FileNotFoundError: If file doesn't exist (no auto-create).
    """
    try:
        p = _validate_path(path, must_exist=True)
        
        if not p.is_file():
            return f"❌ Not a file: {path}"
        
        with open(p, "a", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Appended {len(content)} chars to {path}")
        return f"✅ Appended {len(content)} characters to {p.name}"
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File does not exist (append requires existing file): {path}"
        )
    except PermissionError:
        return f"❌ Permission denied: {path}"
    except Exception as e:
        return f"❌ Error appending: {e}"


# =============================================================================
# TIER 2: Controlled Mutation (The Hands - Dangerous)
# =============================================================================

def create_dir(path: str) -> str:
    """
    Create a directory (and parent directories).
    
    TIER 2: Mutation operation.
    
    Args:
        path: Path to directory to create.
        
    Returns:
        Success message.
    """
    try:
        p = _validate_path(path)
        
        if p.exists():
            if p.is_dir():
                return f"⚠️ Directory already exists: {path}"
            else:
                return f"❌ Path exists as file: {path}"
        
        p.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
        return f"📁 Created: {path}"
        
    except PermissionError:
        return f"❌ Permission denied: {path}"
    except Exception as e:
        return f"❌ Error creating directory: {e}"


def move_file(src: str, dst: str) -> str:
    """
    Move a file or directory.
    
    TIER 2: Mutation operation.
    
    Args:
        src: Source path.
        dst: Destination path (explicit, not just directory).
        
    Returns:
        Success message.
    """
    try:
        src_p = _validate_path(src, must_exist=True)
        dst_p = _validate_path(dst)
        
        # Ensure destination parent exists
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.move(str(src_p), str(dst_p))
        logger.info(f"Moved: {src} -> {dst}")
        return f"✅ Moved: {src_p.name} → {dst_p.name}"
        
    except FileNotFoundError as e:
        return f"❌ {e}"
    except PermissionError:
        return f"❌ Permission denied"
    except Exception as e:
        return f"❌ Error moving: {e}"


def copy_file(src: str, dst: str) -> str:
    """
    Copy a file (preserves metadata).
    
    TIER 2: Mutation operation.
    
    Args:
        src: Source file path.
        dst: Destination file path.
        
    Returns:
        Success message.
    """
    try:
        src_p = _validate_path(src, must_exist=True)
        dst_p = _validate_path(dst)
        
        if not src_p.is_file():
            return f"❌ Source is not a file: {src}"
        
        # Ensure destination parent exists
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(str(src_p), str(dst_p))
        logger.info(f"Copied: {src} -> {dst}")
        return f"✅ Copied: {src_p.name} → {dst_p.name}"
        
    except FileNotFoundError as e:
        return f"❌ {e}"
    except PermissionError:
        return f"❌ Permission denied"
    except Exception as e:
        return f"❌ Error copying: {e}"


def delete_file(path: str, confirm: bool = False) -> str:
    """
    Delete a file or directory.
    
    TIER 2: DANGEROUS mutation operation with safety lock.
    
    Args:
        path: Path to delete.
        confirm: MUST be True to proceed (safety lock).
        
    Returns:
        Success message.
        
    Raises:
        ValueError: If confirm is False (safety lock).
    """
    # CRITICAL: Safety lock
    if not confirm:
        raise ValueError(
            "Safety Lock: confirm=True required to delete. "
            "Call with delete_file(path, confirm=True)"
        )
    
    # CRITICAL: Block wildcards
    if _has_wildcards(path):
        raise ValueError(
            "Security Error: Wildcards (*?) not allowed in delete path"
        )
    
    try:
        p = _validate_path(path, must_exist=True)
        
        if _HAS_TRASH:
            # Safe delete: move to trash
            send2trash(str(p))
            logger.info(f"Moved to trash: {path}")
            return f"🗑️ Moved to Trash: {p.name}"
        else:
            # Permanent delete (no recycle bin)
            if p.is_file():
                os.remove(p)
            elif p.is_dir():
                shutil.rmtree(p)
            else:
                return f"❌ Unknown file type: {path}"
            
            logger.warning(f"PERMANENTLY deleted: {path}")
            return f"⚠️ Permanently Deleted: {p.name} (no recycle bin)"
        
    except FileNotFoundError as e:
        return f"❌ {e}"
    except PermissionError:
        return f"❌ Permission denied: {path}"
    except Exception as e:
        return f"❌ Error deleting: {e}"


# =============================================================================
# TIER 3: Metadata & Verification (The Brain)
# =============================================================================

def get_file_info(path: str) -> str:
    """
    Get detailed file/directory information.
    
    TIER 3: Read-only metadata operation.
    
    Args:
        path: Path to inspect.
        
    Returns:
        Formatted info string.
    """
    try:
        p = _validate_path(path, must_exist=True)
        stat = p.stat()
        
        # Basic info
        file_type = "Directory" if p.is_dir() else "File"
        size = _format_size(stat.st_size)
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        created = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        
        info_lines = [
            f"📋 Info: {p.name}",
            f"   Type: {file_type}",
            f"   Size: {size}",
            f"   Modified: {modified}",
            f"   Created: {created}",
            f"   Path: {p}",
        ]
        
        # Directory-specific: count children
        if p.is_dir():
            try:
                children = list(p.iterdir())
                dirs = sum(1 for c in children if c.is_dir())
                files = len(children) - dirs
                info_lines.append(f"   Contents: {files} files, {dirs} folders")
            except PermissionError:
                info_lines.append("   Contents: (access denied)")
        
        return "\n".join(info_lines)
        
    except FileNotFoundError as e:
        return f"❌ {e}"
    except Exception as e:
        return f"❌ Error: {e}"


def search_files(
    root_path: str,
    pattern: str,
    max_results: int = 10,
) -> str:
    """
    Search for files matching a pattern.
    
    TIER 3: Read-only search operation.
    
    Args:
        root_path: Directory to search in.
        pattern: Glob pattern (e.g., "*.txt", "**/*.py").
        max_results: Maximum results to return (default: 10).
        
    Returns:
        Formatted list of matches.
    """
    try:
        p = _validate_path(root_path, must_exist=True)
        
        if not p.is_dir():
            return f"❌ Not a directory: {root_path}"
        
        # Safety: Limit search scope for system drives
        if str(p) in ("C:\\", "C:/", "/", "/home"):
            return f"❌ Security: Cannot search entire system drive. Be more specific."
        
        # Build search pattern
        search_pattern = str(p / pattern)
        
        # Search with early termination
        matches = []
        for match in glob.iglob(search_pattern, recursive=True):
            matches.append(match)
            if len(matches) >= max_results:
                break
        
        if not matches:
            return f"🔍 No matches for '{pattern}' in {p.name}"
        
        # Format results
        lines = [f"🔍 Found {len(matches)} matches for '{pattern}':"]
        for m in matches:
            rel = os.path.relpath(m, p)
            lines.append(f"  • {rel}")
        
        if len(matches) >= max_results:
            lines.append(f"  ... (limited to {max_results} results)")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"❌ Search error: {e}"


__all__ = [
    # Tier 1: Safe
    "list_dir",
    "read_file",
    "file_exists",
    "write_file",
    "append_file",
    # Tier 2: Controlled Mutation
    "create_dir",
    "move_file",
    "copy_file",
    "delete_file",
    # Tier 3: Metadata
    "get_file_info",
    "search_files",
]
