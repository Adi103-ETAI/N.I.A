"""Git operations.

Provides git root detection, status queries, and remote URL handling.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


_GIT_ROOT_NOT_FOUND = object()

_find_git_root_cache: dict[str, Optional[str]] = {}


@functools.lru_cache(maxsize=50)
def _find_git_root_impl(start_path: str) -> Optional[str]:
    """Find git root by walking up the directory tree.

    Looks for a .git directory or file (worktrees/submodules use a file).
    Returns the directory containing .git, or None if not found.
    """
    current = Path(start_path).resolve()
    root = current.anchor or "/"

    while str(current) != root:
        git_path = current / ".git"
        if git_path.exists():
            return str(current)
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Check root directory as well
    git_path = Path(root) / ".git"
    if git_path.exists():
        return root

    return None


def find_git_root(start_path: Optional[str] = None) -> Optional[str]:
    """Find the git root directory from the given path.

    Walks up the directory tree looking for .git.
    Returns the directory containing .git, or None if not found.
    """
    path = start_path or os.getcwd()
    return _find_git_root_impl(str(Path(path).resolve()))


def find_canonical_git_root(start_path: Optional[str] = None) -> Optional[str]:
    """Find the canonical git repository root, resolving through worktrees.

    For regular repos this is the same as find_git_root.
    For worktrees, follows the .git file → gitdir: → commondir chain.
    """
    root = find_git_root(start_path)
    if root is None:
        return None

    try:
        git_path = Path(root) / ".git"
        git_content = git_path.read_text(encoding="utf-8").strip()

        if not git_content.startswith("gitdir:"):
            return root

        worktree_git_dir = Path(
            root, git_content[len("gitdir:") :].strip()
        ).resolve()

        commondir_path = worktree_git_dir / "commondir"
        common_dir = Path(
            worktree_git_dir, commondir_path.read_text(encoding="utf-8").strip()
        ).resolve()

        # Security: validate structure
        worktrees_dir = common_dir / "worktrees"
        if worktree_git_dir.parent != worktrees_dir:
            return root

        if common_dir.name != ".git":
            return str(common_dir)

        return str(common_dir.parent)
    except (OSError, IOError, ValueError):
        return root


async def _run_git(*args: str, cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode(
        "utf-8", errors="replace"
    )


async def get_is_git(cwd: Optional[str] = None) -> bool:
    """Check if the given directory is in a git repository."""
    return find_git_root(cwd or os.getcwd()) is not None


async def get_head(cwd: Optional[str] = None) -> str:
    """Get the current HEAD commit hash."""
    code, stdout, _ = await _run_git("rev-parse", "HEAD", cwd=cwd)
    return stdout.strip() if code == 0 else ""


async def get_branch(cwd: Optional[str] = None) -> str:
    """Get the current branch name."""
    code, stdout, _ = await _run_git(
        "rev-parse", "--abbrev-ref", "HEAD", cwd=cwd
    )
    return stdout.strip() if code == 0 else ""


async def get_default_branch(cwd: Optional[str] = None) -> str:
    """Get the default branch (main or master)."""
    for branch in ["main", "master", "staging"]:
        code, _, _ = await _run_git("rev-parse", "--verify", f"refs/heads/{branch}", cwd=cwd)
        if code == 0:
            return branch

    # Try remote default
    code, stdout, _ = await _run_git(
        "symbolic-ref", "refs/remotes/origin/HEAD", cwd=cwd
    )
    if code == 0:
        ref = stdout.strip()
        if ref.startswith("refs/remotes/origin/"):
            return ref[len("refs/remotes/origin/"):]

    return "main"


async def get_remote_url(cwd: Optional[str] = None) -> Optional[str]:
    """Get the URL of the 'origin' remote."""
    code, stdout, _ = await _run_git("remote", "get-url", "origin", cwd=cwd)
    return stdout.strip() if code == 0 else None


async def get_is_head_on_remote(cwd: Optional[str] = None) -> bool:
    """Check if HEAD has an upstream tracking branch."""
    code, _, _ = await _run_git("rev-parse", "@{u}", cwd=cwd)
    return code == 0


async def has_unpushed_commits(cwd: Optional[str] = None) -> bool:
    """Check if there are commits not yet pushed to upstream."""
    code, stdout, _ = await _run_git(
        "rev-list", "--count", "@{u}..HEAD", cwd=cwd
    )
    if code != 0:
        return False
    try:
        return int(stdout.strip()) > 0
    except ValueError:
        return False


async def get_is_clean(
    cwd: Optional[str] = None, ignore_untracked: bool = False
) -> bool:
    """Check if the working directory is clean."""
    args = ["--no-optional-locks", "status", "--porcelain"]
    if ignore_untracked:
        args.append("-uno")
    code, stdout, _ = await _run_git(*args, cwd=cwd)
    return stdout.strip() == ""


async def get_changed_files(cwd: Optional[str] = None) -> List[str]:
    """Get list of changed files in the working directory."""
    code, stdout, _ = await _run_git(
        "--no-optional-locks", "status", "--porcelain", cwd=cwd
    )
    if code != 0:
        return []

    files = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if len(line) > 2:
            filename = line[3:].strip()
            if filename:
                files.append(filename)
    return files


@dataclass
class GitFileStatus:
    """Git file status information."""

    tracked: List[str]
    untracked: List[str]


async def get_file_status(cwd: Optional[str] = None) -> GitFileStatus:
    """Get detailed file status (tracked and untracked)."""
    code, stdout, _ = await _run_git(
        "--no-optional-locks", "status", "--porcelain", cwd=cwd
    )

    tracked: list[str] = []
    untracked: list[str] = []

    if code == 0:
        for line in stdout.strip().splitlines():
            if not line:
                continue
            status = line[:2]
            filename = line[2:].strip()
            if status == "??":
                untracked.append(filename)
            elif filename:
                tracked.append(filename)

    return GitFileStatus(tracked=tracked, untracked=untracked)


def normalize_git_remote_url(url: str) -> Optional[str]:
    """Normalize a git remote URL to a canonical form.

    Converts SSH and HTTPS URLs to the same format: host/owner/repo
    (lowercase, no .git).
    """
    trimmed = url.strip()
    if not trimmed:
        return None

    # SSH format: git@host:owner/repo.git
    ssh_match = re.match(r"^git@([^:]+):(.+?)(?:\.git)?$", trimmed)
    if ssh_match:
        return f"{ssh_match.group(1)}/{ssh_match.group(2)}".lower()

    # HTTPS/SSH URL format
    url_match = re.match(
        r"^(?:https?|ssh)://(?:[^@]+@)?([^/]+)/(.+?)(?:\.git)?$", trimmed
    )
    if url_match:
        return f"{url_match.group(1)}/{url_match.group(2)}".lower()

    return None


async def get_repo_remote_hash(cwd: Optional[str] = None) -> Optional[str]:
    """Return a SHA256 hash (first 16 chars) of the normalized git remote URL."""
    remote_url = await get_remote_url(cwd=cwd)
    if not remote_url:
        return None

    normalized = normalize_git_remote_url(remote_url)
    if not normalized:
        return None

    hash_val = hashlib.sha256(normalized.encode()).hexdigest()
    return hash_val[:16]


async def stash_to_clean_state(
    message: Optional[str] = None, cwd: Optional[str] = None
) -> bool:
    """Stash all changes to return git to a clean state.

    Stages untracked files before stashing to prevent data loss.
    """
    try:
        stash_msg = message or f"niaharness auto-stash - {__import__('datetime').datetime.now().isoformat()}"

        # Check for untracked files
        status = await get_file_status(cwd=cwd)

        if status.untracked:
            code, _, _ = await _run_git("add", *status.untracked, cwd=cwd)
            if code != 0:
                return False

        code, _, _ = await _run_git(
            "stash", "push", "--message", stash_msg, cwd=cwd
        )
        return code == 0
    except Exception:
        return False
