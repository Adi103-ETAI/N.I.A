"""
MODULE: File System Operations (3-Tier Security Model)
VERSION: 3.1.0
ACTIVE SCOPE: Read, Search, Metadata.

TARA 2.0 Atomic Tool Module.

DEPRECATED (Sandbox Era): Tier-2 mutation tools (write_file, append_file,
create_dir, move_file, copy_file, delete_file) were removed in v4.0.0.
File mutations are now handled inside Docker sandboxes via the coding-agent
skill, keeping the host filesystem untouched.

Security Architecture:
    TIER 1: Safe (Eyes)   — list_dir, read_file, file_exists
    TIER 3: Brain         — get_file_info, search_files
"""
from __future__ import annotations

import glob
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.core.logger import setup_logger
from src.core.os.platform import get_os_context
from src.capabilities.decorators import security_level

logger = setup_logger("TARA.Tools.FileOps")

# =============================================================================
# Security Helpers
# =============================================================================

def _validate_path(path: str, must_exist: bool = False) -> Path:
    """
    Validate and normalize a file path with SANDBOX ENFORCEMENT.
    
    Args:
        path: Path string to validate.
        must_exist: If True, raise FileNotFoundError if not exists.
        
    Returns:
        Normalized Path object.
        
    Raises:
        ValueError: If path is invalid or OUTSIDE SAFE ZONES.
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
    
    # SANDBOX CHECK: Ensure path is within a Safe Zone
    ctx = get_os_context()
    safe_zones = ctx.get_safe_zones()

    is_safe = False
    for zone in safe_zones:
        try:
            # Check if p is inside zone (or is the zone itself)
            p.relative_to(zone)
            is_safe = True
            break
        except ValueError:
            continue

    if not is_safe:
        # Construct helpful error message
        zone_list = "\n".join([f"  - {z}" for z in safe_zones])
        raise ValueError(
            f"🚫 Security Access Denied: Path '{path}' is outside the allowed workspace.\n"
            f"Allowed locations:\n{zone_list}"
        )

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
    # Tier 1: Safe (Active)
    "list_dir",
    "read_file",
    "file_exists",
    # Tier 3: Metadata (Active)
    "get_file_info",
    "search_files",
]
