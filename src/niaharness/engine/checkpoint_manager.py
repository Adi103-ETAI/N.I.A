"""Checkpoint Manager — Transparent filesystem snapshots via a single shared
shadow git store.

Ported line-by-line from hermes-agent/tools/checkpoint_manager.py (1,675 LOC).

Creates automatic snapshots of working directories before file-mutating
operations, triggered once per conversation turn. Provides rollback to any
previous checkpoint.

This is NOT a tool — the LLM never sees it. It's transparent infrastructure
controlled by the ``checkpoints`` config flag or ``--checkpoints`` CLI flag.

Storage layout (single shared store, git objects deduplicated across projects)
-----------------------------------------------------------------------------

    ~/.nia/checkpoints/
        store/                          — single bare-ish git repo
            HEAD, config, objects/      — standard git internals (shared)
            refs/nia/<hash16>           — per-project branch tip
            indexes/<hash16>            — per-project git index
            projects/<hash16>.json      — {workdir, created_at, last_touch}
            info/exclude                — default excludes (shared)
        .last_prune                     — auto-prune idempotency marker
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _get_nia_home() -> Path:
    """Return the NIA home directory."""
    try:
        from niaharness.prompts.soul import get_nia_home

        return get_nia_home()
    except Exception:
        return Path(os.path.expanduser("~/.nia"))


CHECKPOINT_BASE = _get_nia_home() / "checkpoints"

# Single shared store directory under CHECKPOINT_BASE.
_STORE_DIRNAME = "store"
_REFS_PREFIX = "refs/nia"
_INDEXES_DIRNAME = "indexes"
_PROJECTS_DIRNAME = "projects"
_LEGACY_PREFIX = "legacy-"
_PRUNE_MARKER_NAME = ".last_prune"

DEFAULT_EXCLUDES = [
    # Dependency / build output
    "node_modules/", "dist/", "build/", "target/", "out/", ".next/", ".nuxt/",
    # Caches
    "__pycache__/", "*.pyc", "*.pyo", ".cache/", ".pytest_cache/",
    ".mypy_cache/", ".ruff_cache/", "coverage/", ".coverage",
    # Virtualenvs
    ".venv/", "venv/", "env/",
    # VCS
    ".git/", ".hg/", ".svn/",
    # Worktrees (NIA convention — don't recursively snapshot siblings)
    ".worktrees/",
    # Native / compiled binaries
    "*.so", "*.dylib", "*.dll", "*.o", "*.a", "*.jar", "*.class", "*.exe", "*.obj",
    # Media / large binaries
    "*.mp4", "*.mov", "*.mkv", "*.webm", "*.zip", "*.tar", "*.tar.gz",
    "*.tgz", "*.7z", "*.rar", "*.iso",
    # Secrets
    ".env", ".env.*", ".env.local", ".env.*.local",
    # OS junk
    ".DS_Store", "Thumbs.db",
    # Logs
    "*.log",
]

# Git subprocess timeout (seconds).
_GIT_TIMEOUT: int = max(10, min(60, int(os.environ.get("NIA_CHECKPOINT_TIMEOUT", "30") or "30")))

# Max files to snapshot — skip huge directories to avoid slowdowns.
_MAX_FILES = 50_000

# Valid git commit hash pattern: 4–40 hex chars (short or full SHA-1/SHA-256).
_COMMIT_HASH_RE = re.compile(r"^[0-9a-fA-F]{4,64}$")


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _validate_commit_hash(commit_hash: str) -> Optional[str]:
    """Validate a commit hash to prevent git argument injection.

    Returns an error string if invalid, None if valid.
    """
    if not commit_hash or not commit_hash.strip():
        return "Empty commit hash"
    if commit_hash.startswith("-"):
        return f"Invalid commit hash (must not start with '-'): {commit_hash!r}"
    if not _COMMIT_HASH_RE.match(commit_hash):
        return f"Invalid commit hash (expected 4-64 hex characters): {commit_hash!r}"
    return None


def _validate_file_path(file_path: str, working_dir: str) -> Optional[str]:
    """Validate a file path to prevent path traversal outside the working directory."""
    if not file_path or not file_path.strip():
        return "Empty file path"
    if os.path.isabs(file_path):
        return f"File path must be relative, got absolute path: {file_path!r}"
    abs_workdir = _normalize_path(working_dir)
    resolved = (abs_workdir / file_path).resolve()
    try:
        resolved.relative_to(abs_workdir)
    except ValueError:
        return f"File path escapes the working directory via traversal: {file_path!r}"
    return None


# ---------------------------------------------------------------------------
# Path / hash helpers
# ---------------------------------------------------------------------------


def _normalize_path(path_value: str) -> Path:
    """Return a canonical absolute path for checkpoint operations."""
    return Path(path_value).expanduser().resolve()


def _project_hash(working_dir: str) -> str:
    """Deterministic per-project hash: sha256(abs_path)[:16]."""
    abs_path = str(_normalize_path(working_dir))
    return hashlib.sha256(abs_path.encode()).hexdigest()[:16]


def _store_path(base: Optional[Path] = None) -> Path:
    """Return the single shared shadow store path."""
    return (base or CHECKPOINT_BASE) / _STORE_DIRNAME


def _index_path(store: Path, dir_hash: str) -> Path:
    return store / _INDEXES_DIRNAME / dir_hash


def _ref_name(dir_hash: str) -> str:
    return f"{_REFS_PREFIX}/{dir_hash}"


def _project_meta_path(store: Path, dir_hash: str) -> Path:
    return store / _PROJECTS_DIRNAME / f"{dir_hash}.json"


# ---------------------------------------------------------------------------
# Git env
# ---------------------------------------------------------------------------


def _git_env(
    store: Path, working_dir: str, index_file: Optional[Path] = None,
) -> dict:
    """Build env dict that redirects git to the shared store.

    The shared store is internal NIA infrastructure — it must NOT inherit
    the user's global or system git config. User-level settings like
    ``commit.gpgsign = true``, signing hooks, or credential helpers would
    either break background snapshots or, worse, spawn interactive prompts.
    """
    normalized_working_dir = _normalize_path(working_dir)
    env = os.environ.copy()
    env["GIT_DIR"] = str(store)
    env["GIT_WORK_TREE"] = str(normalized_working_dir)
    env.pop("GIT_NAMESPACE", None)
    env.pop("GIT_ALTERNATE_OBJECT_DIRECTORIES", None)
    if index_file is not None:
        env["GIT_INDEX_FILE"] = str(index_file)
    else:
        env.pop("GIT_INDEX_FILE", None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def _repair_bare_repo_dirs(store: Path) -> None:
    """Recreate refs/ and branches/ dirs that ``git gc`` may have removed."""
    for subdir in ("refs/heads", "branches"):
        path = store / subdir
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.debug("Repaired missing %s in checkpoint store", subdir)
            except OSError as exc:
                logger.warning("Cannot create %s in checkpoint store: %s", subdir, exc)


def _run_git(
    args: List[str],
    store: Path,
    working_dir: str,
    timeout: int = _GIT_TIMEOUT,
    allowed_returncodes: Optional[Set[int]] = None,
    index_file: Optional[Path] = None,
) -> Tuple[bool, str, str]:
    """Run a git command against the shared store. Returns (ok, stdout, stderr)."""
    normalized_working_dir = _normalize_path(working_dir)
    if not normalized_working_dir.exists():
        msg = f"working directory not found: {normalized_working_dir}"
        logger.error("Git command skipped: %s (%s)", " ".join(["git"] + list(args)), msg)
        return False, "", msg
    if not normalized_working_dir.is_dir():
        msg = f"working directory is not a directory: {normalized_working_dir}"
        logger.error("Git command skipped: %s (%s)", " ".join(["git"] + list(args)), msg)
        return False, "", msg

    env = _git_env(store, str(normalized_working_dir), index_file=index_file)
    cmd = ["git"] + list(args)
    allowed_returncodes = allowed_returncodes or set()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
            cwd=str(normalized_working_dir), stdin=subprocess.DEVNULL,
        )
        ok = result.returncode == 0
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if not ok and result.returncode not in allowed_returncodes:
            logger.error(
                "Git command failed: %s (rc=%d) stderr=%s",
                " ".join(cmd), result.returncode, stderr,
            )
        return ok, stdout, stderr
    except subprocess.TimeoutExpired:
        msg = f"git timed out after {timeout}s: {' '.join(cmd)}"
        logger.error(msg, exc_info=True)
        return False, "", msg
    except FileNotFoundError as exc:
        missing_target = getattr(exc, "filename", None)
        if missing_target == "git":
            logger.error("Git executable not found: %s", " ".join(cmd), exc_info=True)
            return False, "", "git not found"
        msg = f"working directory not found: {normalized_working_dir}"
        logger.error("Git command failed before execution: %s (%s)", " ".join(cmd), msg, exc_info=True)
        return False, "", msg
    except Exception as exc:
        logger.error("Unexpected git error running %s: %s", " ".join(cmd), exc, exc_info=True)
        return False, "", str(exc)


# ---------------------------------------------------------------------------
# Store initialisation + legacy migration
# ---------------------------------------------------------------------------


def _migrate_legacy_store(base: Path) -> Optional[Path]:
    """Move pre-v2 per-project shadow repos into a ``legacy-<ts>/`` dir."""
    if not base.exists():
        return None
    store = _store_path(base)
    legacy_root: Optional[Path] = None
    reserved = {_STORE_DIRNAME, _PRUNE_MARKER_NAME}
    for child in list(base.iterdir()):
        name = child.name
        if name in reserved or name.startswith(_LEGACY_PREFIX):
            continue
        if legacy_root is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            legacy_root = base / f"{_LEGACY_PREFIX}{stamp}"
            try:
                legacy_root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("Could not create legacy archive dir: %s", exc)
                return None
        dest = legacy_root / name
        try:
            shutil.move(str(child), str(dest))
        except OSError as exc:
            logger.warning("Could not archive legacy checkpoint %s: %s", child, exc)
    _ = store
    if legacy_root is not None:
        logger.info(
            "Migrated pre-v2 checkpoint repos to %s. "
            "Clear with `nia checkpoints clear-legacy` when safe.",
            legacy_root,
        )
    return legacy_root


def _init_store(store: Path, working_dir: str) -> Optional[str]:
    """Initialise the shared shadow store if needed. Returns error or None."""
    base = store.parent
    if not store.exists():
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"Could not create checkpoint base: {exc}"
        _migrate_legacy_store(base)

    if (store / "HEAD").exists():
        return None

    store.mkdir(parents=True, exist_ok=True)
    (store / _INDEXES_DIRNAME).mkdir(exist_ok=True)
    (store / _PROJECTS_DIRNAME).mkdir(exist_ok=True)

    # ``git init --bare`` rejects GIT_WORK_TREE, so use a raw subprocess.
    init_env = os.environ.copy()
    init_env["GIT_CONFIG_GLOBAL"] = os.devnull
    init_env["GIT_CONFIG_SYSTEM"] = os.devnull
    init_env["GIT_CONFIG_NOSYSTEM"] = "1"
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_NAMESPACE",
              "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        init_env.pop(k, None)
    try:
        result = subprocess.run(
            ["git", "init", "--bare", str(store)],
            capture_output=True, text=True, env=init_env,
            timeout=_GIT_TIMEOUT, stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            return f"Shadow store init failed: {result.stderr.strip()}"
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return f"Shadow store init failed: {exc}"

    cfg_wd = str(base)
    _run_git(["config", "user.email", "nia@local"], store, cfg_wd)
    _run_git(["config", "user.name", "NIA Checkpoint"], store, cfg_wd)
    _run_git(["config", "commit.gpgsign", "false"], store, cfg_wd)
    _run_git(["config", "tag.gpgSign", "false"], store, cfg_wd)
    _run_git(["config", "gc.auto", "0"], store, cfg_wd)

    info_dir = store / "info"
    info_dir.mkdir(exist_ok=True)
    (info_dir / "exclude").write_text(
        "\n".join(DEFAULT_EXCLUDES) + "\n", encoding="utf-8"
    )

    logger.debug("Initialised checkpoint store at %s", store)
    return None


def _register_project(store: Path, working_dir: str) -> None:
    """Create or update ``projects/<hash>.json`` with workdir + timestamps."""
    dir_hash = _project_hash(working_dir)
    meta_path = _project_meta_path(store, dir_hash)
    now = time.time()
    meta: Dict = {
        "workdir": str(_normalize_path(working_dir)),
        "created_at": now,
        "last_touch": now,
    }
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                meta["created_at"] = existing.get("created_at", now)
        except (OSError, ValueError):
            pass
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write project metadata %s: %s", meta_path, exc)


def _touch_project(store: Path, working_dir: str) -> None:
    """Update last_touch for a project, preserving created_at."""
    dir_hash = _project_hash(working_dir)
    meta_path = _project_meta_path(store, dir_hash)
    if not meta_path.exists():
        _register_project(store, working_dir)
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    meta["workdir"] = str(_normalize_path(working_dir))
    meta["last_touch"] = time.time()
    meta.setdefault("created_at", meta["last_touch"])
    try:
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not update project metadata %s: %s", meta_path, exc)


def _list_projects(store: Path) -> List[Dict]:
    """Return all registered projects under the store."""
    projects_dir = store / _PROJECTS_DIRNAME
    if not projects_dir.exists():
        return []
    out: List[Dict] = []
    for meta_path in projects_dir.glob("*.json"):
        dir_hash = meta_path.stem
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        meta["_hash"] = dir_hash
        out.append(meta)
    return out


def _dir_file_count(path: str) -> int:
    """Quick file count estimate (stops early if over _MAX_FILES)."""
    count = 0
    try:
        for _ in Path(path).rglob("*"):
            count += 1
            if count > _MAX_FILES:
                return count
    except (PermissionError, OSError):
        pass
    return count


def _dir_size_bytes(path: Path) -> int:
    """Best-effort recursive size in bytes. Returns 0 on error."""
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Manages automatic filesystem checkpoints via a shared shadow git store.

    Ported from hermes-agent/tools/checkpoint_manager.py line 608.

    Designed to be owned by the QueryEngine. Call ``new_turn()`` at the start
    of each conversation turn and ``ensure_checkpoint(dir, reason)`` before
    any file-mutating tool call. The manager deduplicates so at most one
    snapshot is taken per directory per turn.
    """

    def __init__(
        self,
        enabled: bool = False,
        max_snapshots: int = 20,
        max_total_size_mb: int = 500,
        max_file_size_mb: int = 10,
    ):
        self.enabled = enabled
        self.max_snapshots = max(1, int(max_snapshots))
        self.max_total_size_mb = max(0, int(max_total_size_mb))
        self.max_file_size_mb = max(0, int(max_file_size_mb))
        self._checkpointed_dirs: Set[str] = set()
        self._git_available: Optional[bool] = None  # lazy probe

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    def new_turn(self) -> None:
        """Reset per-turn dedup. Call at the start of each agent iteration."""
        self._checkpointed_dirs.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_checkpoint(self, working_dir: str, reason: str = "auto") -> bool:
        """Take a checkpoint if enabled and not already done this turn.

        Returns True if a checkpoint was taken, False otherwise.
        Never raises — all errors are silently logged.
        """
        if not self.enabled:
            return False

        if self._git_available is None:
            self._git_available = shutil.which("git") is not None
            if not self._git_available:
                logger.debug("Checkpoints disabled: git not found")
        if not self._git_available:
            return False

        abs_dir = str(_normalize_path(working_dir))

        # Skip root, home, and other overly broad directories
        if abs_dir in {"/", str(Path.home())}:
            logger.debug("Checkpoint skipped: directory too broad (%s)", abs_dir)
            return False

        if abs_dir in self._checkpointed_dirs:
            return False

        self._checkpointed_dirs.add(abs_dir)

        try:
            return self._take(abs_dir, reason)
        except Exception as e:
            logger.debug("Checkpoint failed (non-fatal): %s", e)
            return False

    def list_checkpoints(self, working_dir: str) -> List[Dict]:
        """List available checkpoints for a directory (most recent first)."""
        abs_dir = str(_normalize_path(working_dir))
        store = _store_path(CHECKPOINT_BASE)

        if not (store / "HEAD").exists():
            return []

        ref = _ref_name(_project_hash(abs_dir))
        ok, stdout, _ = _run_git(
            ["log", ref, "--format=%H|%h|%aI|%s", "-n", str(self.max_snapshots)],
            store, abs_dir, allowed_returncodes={128, 129},
        )

        if not ok or not stdout:
            return []

        results: List[Dict] = []
        for line in stdout.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                entry = {
                    "hash": parts[0],
                    "short_hash": parts[1],
                    "timestamp": parts[2],
                    "reason": parts[3],
                    "files_changed": 0,
                    "insertions": 0,
                    "deletions": 0,
                }
                stat_ok, stat_out, _ = _run_git(
                    ["diff", "--shortstat", f"{parts[0]}~1", parts[0]],
                    store, abs_dir, allowed_returncodes={128, 129},
                )
                if stat_ok and stat_out:
                    self._parse_shortstat(stat_out, entry)
                results.append(entry)
        return results

    @staticmethod
    def _parse_shortstat(stat_line: str, entry: Dict) -> None:
        """Parse git --shortstat output into entry dict."""
        m = re.search(r"(\d+) file", stat_line)
        if m:
            entry["files_changed"] = int(m.group(1))
        m = re.search(r"(\d+) insertion", stat_line)
        if m:
            entry["insertions"] = int(m.group(1))
        m = re.search(r"(\d+) deletion", stat_line)
        if m:
            entry["deletions"] = int(m.group(1))

    def diff(self, working_dir: str, commit_hash: str) -> Dict:
        """Show diff between a checkpoint and the current working tree."""
        hash_err = _validate_commit_hash(commit_hash)
        if hash_err:
            return {"success": False, "error": hash_err}

        abs_dir = str(_normalize_path(working_dir))
        store = _store_path(CHECKPOINT_BASE)

        if not (store / "HEAD").exists():
            return {"success": False, "error": "No checkpoints exist for this directory"}

        ok, _, err = _run_git(
            ["cat-file", "-t", commit_hash], store, abs_dir,
        )
        if not ok:
            return {"success": False, "error": f"Checkpoint '{commit_hash}' not found"}

        dir_hash = _project_hash(abs_dir)
        index_file = _index_path(store, dir_hash)

        # Stage current state into the per-project index to compare.
        _run_git(["add", "-A"], store, abs_dir, timeout=_GIT_TIMEOUT * 2, index_file=index_file)

        ok_stat, stat_out, _ = _run_git(
            ["diff", "--stat", commit_hash, "--cached"],
            store, abs_dir, index_file=index_file,
        )
        ok_diff, diff_out, _ = _run_git(
            ["diff", commit_hash, "--cached", "--no-color"],
            store, abs_dir, index_file=index_file,
        )

        # Reset staged tree back to the project's last checkpoint.
        ref = _ref_name(dir_hash)
        _run_git(["read-tree", ref], store, abs_dir, index_file=index_file, allowed_returncodes={128})

        if not ok_stat and not ok_diff:
            return {"success": False, "error": "Could not generate diff"}

        return {
            "success": True,
            "stat": stat_out if ok_stat else "",
            "diff": diff_out if ok_diff else "",
        }

    def restore(self, working_dir: str, commit_hash: str, file_path: str = None) -> Dict:
        """Restore files to a checkpoint state."""
        hash_err = _validate_commit_hash(commit_hash)
        if hash_err:
            return {"success": False, "error": hash_err}

        abs_dir = str(_normalize_path(working_dir))

        if file_path:
            path_err = _validate_file_path(file_path, abs_dir)
            if path_err:
                return {"success": False, "error": path_err}

        store = _store_path(CHECKPOINT_BASE)

        if not (store / "HEAD").exists():
            return {"success": False, "error": "No checkpoints exist for this directory"}

        ok, _, err = _run_git(["cat-file", "-t", commit_hash], store, abs_dir)
        if not ok:
            return {
                "success": False,
                "error": f"Checkpoint '{commit_hash}' not found",
                "debug": err or None,
            }

        # Take a pre-rollback snapshot so you can undo the undo.
        self._take(abs_dir, f"pre-rollback snapshot (restoring to {commit_hash[:8]})")

        dir_hash = _project_hash(abs_dir)
        index_file = _index_path(store, dir_hash)

        restore_target = file_path if file_path else "."
        ok, stdout, err = _run_git(
            ["checkout", commit_hash, "--", restore_target],
            store, abs_dir, timeout=_GIT_TIMEOUT * 2, index_file=index_file,
        )

        if not ok:
            return {
                "success": False,
                "error": f"Restore failed: {err}",
                "debug": err or None,
            }

        ok2, reason_out, _ = _run_git(
            ["log", "--format=%s", "-1", commit_hash], store, abs_dir,
        )
        reason = reason_out if ok2 else "unknown"

        result: Dict = {
            "success": True,
            "restored_to": commit_hash[:8],
            "reason": reason,
            "directory": abs_dir,
        }
        if file_path:
            result["file"] = file_path
        return result

    def resolve_hash(self, working_dir: str, ref: str) -> str:
        """Resolve a short-hash / tag to a full checkpoint hash.

        Falls back to the input ref if resolution fails.
        """
        abs_dir = str(_normalize_path(working_dir))
        store = _store_path(CHECKPOINT_BASE)
        if not (store / "HEAD").exists():
            return ref
        ok, stdout, _ = _run_git(
            ["rev-parse", "--verify", f"{ref}^{{commit}}"],
            store, abs_dir, allowed_returncodes={128},
        )
        if ok and stdout:
            return stdout
        return ref

    def get_working_dir_for_path(self, file_path: str) -> str:
        """Resolve a file path to its working directory for checkpointing."""
        path = _normalize_path(file_path)
        if path.is_dir():
            candidate = path
        else:
            candidate = path.parent

        markers = {
            ".git", "pyproject.toml", "package.json", "Cargo.toml",
            "go.mod", "Makefile", "pom.xml", ".hg", "Gemfile",
        }
        check = candidate
        while check != check.parent:
            if any((check / m).exists() for m in markers):
                return str(check)
            check = check.parent
        return str(candidate)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _take(self, working_dir: str, reason: str) -> bool:
        """Take a snapshot. Returns True on success."""
        store = _store_path(CHECKPOINT_BASE)

        err = _init_store(store, working_dir)
        if err:
            logger.debug("Checkpoint store init failed: %s", err)
            return False

        _touch_project(store, working_dir)

        if _dir_file_count(working_dir) > _MAX_FILES:
            logger.debug("Checkpoint skipped: >%d files in %s", _MAX_FILES, working_dir)
            return False

        dir_hash = _project_hash(working_dir)
        index_file = _index_path(store, dir_hash)
        ref = _ref_name(dir_hash)

        if index_file.exists():
            ok_ref, ref_commit, _ = _run_git(
                ["rev-parse", "--verify", ref + "^{commit}"],
                store, working_dir, allowed_returncodes={128},
            )
            if ok_ref and ref_commit:
                _run_git(
                    ["read-tree", ref_commit], store, working_dir,
                    index_file=index_file, allowed_returncodes={128},
                )
            else:
                try:
                    index_file.unlink()
                except OSError:
                    pass
        else:
            index_file.parent.mkdir(parents=True, exist_ok=True)

        ok, _, err = _run_git(
            ["add", "-A"], store, working_dir,
            timeout=_GIT_TIMEOUT * 2, index_file=index_file,
        )
        if not ok:
            logger.debug("Checkpoint git-add failed: %s", err)
            return False

        if self.max_file_size_mb > 0:
            self._drop_oversize_from_index(store, working_dir, index_file)

        ok_ref, ref_commit, _ = _run_git(
            ["rev-parse", "--verify", ref + "^{commit}"],
            store, working_dir, allowed_returncodes={128},
        )
        has_ref = ok_ref and bool(ref_commit)

        if has_ref:
            ok_diff, _, _ = _run_git(
                ["diff-index", "--cached", "--quiet", ref_commit],
                store, working_dir, allowed_returncodes={1}, index_file=index_file,
            )
            if ok_diff:
                logger.debug("Checkpoint skipped: no changes in %s", working_dir)
                return False
        else:
            ok_ls, ls_out, _ = _run_git(
                ["ls-files", "--cached"], store, working_dir, index_file=index_file,
            )
            if ok_ls and not ls_out.strip():
                logger.debug("Checkpoint skipped: empty tree in %s", working_dir)
                return False

        ok_tree, tree_sha, err = _run_git(
            ["write-tree"], store, working_dir, index_file=index_file,
        )
        if not ok_tree or not tree_sha:
            logger.debug("Checkpoint write-tree failed: %s", err)
            return False

        commit_args = ["commit-tree", tree_sha, "-m", reason, "--no-gpg-sign"]
        if has_ref:
            commit_args = ["commit-tree", tree_sha, "-p", ref_commit, "-m", reason, "--no-gpg-sign"]
        ok_commit, new_sha, err = _run_git(
            commit_args, store, working_dir, index_file=index_file,
        )
        if not ok_commit or not new_sha:
            logger.debug("Checkpoint commit-tree failed: %s", err)
            return False

        update_args = ["update-ref", ref, new_sha]
        if has_ref:
            update_args = ["update-ref", ref, new_sha, ref_commit]
        ok_update, _, err = _run_git(update_args, store, working_dir)
        if not ok_update:
            logger.debug("Checkpoint update-ref failed: %s", err)
            return False

        logger.debug("Checkpoint taken in %s: %s (%s)", working_dir, reason, new_sha[:8])

        self._prune(store, working_dir, ref)
        self._enforce_size_cap(store)

        return True

    def _drop_oversize_from_index(
        self, store: Path, working_dir: str, index_file: Path,
    ) -> None:
        """Remove any staged file larger than ``max_file_size_mb`` from the index."""
        cap = self.max_file_size_mb * 1024 * 1024
        if cap <= 0:
            return
        ok, stdout, _ = _run_git(
            ["ls-files", "--cached", "-z"], store, working_dir, index_file=index_file,
        )
        if not ok or not stdout:
            return
        paths = [p for p in stdout.split("\x00") if p]
        abs_workdir = _normalize_path(working_dir)
        oversize: List[str] = []
        for rel in paths:
            try:
                size = (abs_workdir / rel).stat().st_size
            except OSError:
                continue
            if size > cap:
                oversize.append(rel)
        if not oversize:
            return
        logger.debug(
            "Checkpoint: dropping %d oversize file(s) (>%d MB) from index",
            len(oversize), self.max_file_size_mb,
        )
        BATCH = 200
        for i in range(0, len(oversize), BATCH):
            chunk = oversize[i:i + BATCH]
            _run_git(
                ["rm", "--cached", "--quiet", "--"] + chunk,
                store, working_dir, index_file=index_file, allowed_returncodes={128},
            )

    def _prune(self, store: Path, working_dir: str, ref: str) -> None:
        """Keep only the last ``max_snapshots`` commits on the per-project ref."""
        ok, stdout, _ = _run_git(
            ["rev-list", "--count", ref], store, working_dir, allowed_returncodes={128},
        )
        if not ok:
            return
        try:
            count = int(stdout)
        except ValueError:
            return
        if count <= self.max_snapshots:
            return

        ok_list, list_out, _ = _run_git(
            ["rev-list", "--reverse", ref], store, working_dir,
        )
        if not ok_list or not list_out:
            return
        commits = list_out.splitlines()
        keep = commits[-self.max_snapshots:]

        new_parent: Optional[str] = None
        for sha in keep:
            ok_tree, tree_sha, _ = _run_git(
                ["rev-parse", f"{sha}^{{tree}}"], store, working_dir,
            )
            if not ok_tree or not tree_sha:
                return
            ok_msg, msg, _ = _run_git(
                ["log", "--format=%s", "-1", sha], store, working_dir,
            )
            commit_msg = msg if ok_msg and msg else "checkpoint"
            args = ["commit-tree", tree_sha, "-m", commit_msg, "--no-gpg-sign"]
            if new_parent is not None:
                args = ["commit-tree", tree_sha, "-p", new_parent, "-m", commit_msg, "--no-gpg-sign"]
            ok_commit, new_sha, _ = _run_git(args, store, working_dir)
            if not ok_commit or not new_sha:
                return
            new_parent = new_sha

        if new_parent is None:
            return
        _run_git(["update-ref", ref, new_parent], store, working_dir)

        _run_git(["reflog", "expire", "--expire=now", "--all"], store, working_dir)
        _run_git(
            ["gc", "--prune=now", "--quiet"], store, working_dir, timeout=_GIT_TIMEOUT * 3,
        )
        _repair_bare_repo_dirs(store)

    def _enforce_size_cap(self, store: Path) -> None:
        """If total store size exceeds ``max_total_size_mb``, drop oldest
        checkpoints across ALL projects until under the cap.
        """
        if self.max_total_size_mb <= 0:
            return
        cap_bytes = self.max_total_size_mb * 1024 * 1024
        size = _dir_size_bytes(store)
        if size <= cap_bytes:
            return
        logger.info(
            "Checkpoint store exceeded %d MB (actual %d MB) — pruning oldest",
            self.max_total_size_mb, size // (1024 * 1024),
        )

        ok, stdout, _ = _run_git(
            ["for-each-ref", "--format=%(refname)", _REFS_PREFIX],
            store, str(store.parent), allowed_returncodes={128},
        )
        if not ok or not stdout:
            return
        refs = [r for r in stdout.splitlines() if r.strip()]

        any_dropped = False
        for _ in range(20):  # hard upper bound
            size = _dir_size_bytes(store)
            if size <= cap_bytes:
                break
            for ref in refs:
                ok_count, count_out, _ = _run_git(
                    ["rev-list", "--count", ref], store, str(store.parent),
                    allowed_returncodes={128},
                )
                try:
                    count = int(count_out) if ok_count else 0
                except ValueError:
                    count = 0
                if count <= 1:
                    continue
                ok_list, list_out, _ = _run_git(
                    ["rev-list", "--reverse", ref], store, str(store.parent),
                )
                if not ok_list or not list_out:
                    continue
                commits = list_out.splitlines()
                keep = commits[1:]
                new_parent: Optional[str] = None
                fail = False
                for sha in keep:
                    ok_tree, tree_sha, _ = _run_git(
                        ["rev-parse", f"{sha}^{{tree}}"], store, str(store.parent),
                    )
                    if not ok_tree or not tree_sha:
                        fail = True
                        break
                    ok_msg, msg, _ = _run_git(
                        ["log", "--format=%s", "-1", sha], store, str(store.parent),
                    )
                    commit_msg = msg if ok_msg and msg else "checkpoint"
                    args = ["commit-tree", tree_sha, "-m", commit_msg, "--no-gpg-sign"]
                    if new_parent is not None:
                        args = ["commit-tree", tree_sha, "-p", new_parent, "-m", commit_msg, "--no-gpg-sign"]
                    ok_commit, new_sha, _ = _run_git(args, store, str(store.parent))
                    if not ok_commit or not new_sha:
                        fail = True
                        break
                    new_parent = new_sha
                if fail or new_parent is None:
                    continue
                _run_git(["update-ref", ref, new_parent], store, str(store.parent))
                any_dropped = True
            if not any_dropped:
                break

        _run_git(["reflog", "expire", "--expire=now", "--all"], store, str(store.parent))
        _run_git(
            ["gc", "--prune=now", "--quiet"], store, str(store.parent),
            timeout=_GIT_TIMEOUT * 3,
        )
        _repair_bare_repo_dirs(store)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def format_checkpoint_list(checkpoints: List[Dict], directory: str) -> str:
    """Format checkpoint list for display to user."""
    if not checkpoints:
        return f"No checkpoints found for {directory}"

    lines = [f"Checkpoints for {directory}:\n"]
    for i, cp in enumerate(checkpoints, 1):
        ts = cp["timestamp"]
        reason = cp["reason"]
        short = cp.get("short_hash", cp["hash"][:8])
        lines.append(f"  {i}. {short} — {ts} — {reason}")
    return "\n".join(lines)


def store_status(checkpoint_base: Optional[Path] = None) -> Dict:
    """Return status info about the checkpoint store."""
    base = checkpoint_base or CHECKPOINT_BASE
    store = _store_path(base)
    if not store.exists():
        return {"enabled": False, "store_path": str(store), "projects": 0, "size_mb": 0}
    projects = _list_projects(store)
    size = _dir_size_bytes(store)
    return {
        "enabled": True,
        "store_path": str(store),
        "projects": len(projects),
        "size_mb": size // (1024 * 1024),
    }


def clear_all(checkpoint_base: Optional[Path] = None) -> Dict[str, int]:
    """Clear all checkpoints. Returns counts of what was removed."""
    base = checkpoint_base or CHECKPOINT_BASE
    store = _store_path(base)
    if not store.exists():
        return {"projects_removed": 0, "bytes_freed": 0}
    size = _dir_size_bytes(store)
    try:
        shutil.rmtree(store)
    except OSError as exc:
        logger.warning("Could not clear checkpoint store: %s", exc)
        return {"projects_removed": 0, "bytes_freed": 0}
    return {"projects_removed": 0, "bytes_freed": size}


__all__ = [
    "CheckpointManager",
    "format_checkpoint_list",
    "store_status",
    "clear_all",
    "CHECKPOINT_BASE",
    "DEFAULT_EXCLUDES",
]
