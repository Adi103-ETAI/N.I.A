"""Authoritative project → repo → lane → session tree builder.

Ported from Hermes Agent's ``tui_gateway/project_tree.py`` (558 LOC).

This is the single source of truth for how the sidebar groups sessions
into projects, repos, and lanes. It is pure (all git resolution is
injected via ``resolve``) so it can be unit-tested with fixtures.

Lane IDs (must match the renderer's persisted state):
  - explicit project id .......... ``p_<hex>`` (from projects.db)
  - auto/discovered project id ... the repo root path
  - repo node id ................. the repo root path
  - main branch lane id .......... ``<repoRoot>::branch::<branch>`` (or ``::branch::``)
  - kanban bucket lane id ........ ``<repoRoot>::kanban``
  - linked worktree lane id ...... the worktree path
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

# A cwd → git identity resolver.
Resolve = Callable[[str], Optional[dict]]

# Only KANBAN-TASK worktrees collapse into one lane.
_KANBAN_DIR_RE = re.compile(r"^(.*[/\\]\.worktrees)[/\\]t_[0-9a-f]+[/\\]?$")
_TRUNK_BRANCHES = {"main", "master", "trunk", "develop"}
DEFAULT_BRANCH_LABEL = "main"


def _branch_lane_id(repo_root: str, branch: str = "") -> str:
    """The one definition of a main-checkout lane id."""
    return f"{repo_root}::branch::{(branch or '').strip()}"


def _kanban_lane_id(repo_root: str) -> str:
    return f"{repo_root}::kanban"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _segments(path: str) -> list[str]:
    return [s for s in re.split(r"[/\\]", (path or "").rstrip("/\\")) if s]


def base_name(path: str) -> str:
    segs = _segments(path)
    return segs[-1] if segs else ""


def kanban_worktree_dir(path: str) -> Optional[str]:
    """The ``<repo>/.worktrees`` dir for a ``.../.worktrees/<task>`` path, else None."""
    m = _KANBAN_DIR_RE.match(path or "")
    return m.group(1) if m else None


def _is_path_under(folder: str, target: str) -> bool:
    """True when ``target`` equals ``folder`` or is nested under it."""
    f = _segments(folder)
    t = _segments(target)
    if not f or len(f) > len(t):
        return False
    return all(f[i] == t[i] for i in range(len(f)))


def _with_base_name(path: str, name: str) -> str:
    stripped = re.sub(r"[/\\]+$", "", path)
    return re.sub(r"[^/\\]+$", name, stripped)


# ---------------------------------------------------------------------------
# Lane placement
# ---------------------------------------------------------------------------


def _placement(
    repo_root: str,
    lane_key: str,
    lane_label: str,
    lane_path: str,
    is_main: bool,
    is_kanban: bool,
) -> dict:
    return {
        "repo_key": repo_root,
        "repo_label": base_name(repo_root) or repo_root,
        "repo_path": repo_root,
        "lane_key": lane_key,
        "lane_label": lane_label,
        "lane_path": lane_path,
        "is_main": is_main,
        "is_kanban": is_kanban,
    }


def _place_by_heuristic(path: str) -> Optional[dict]:
    """Path-only fallback when there is no git probe and no persisted root."""
    base = base_name(path)
    if not base:
        return None

    kanban_dir = kanban_worktree_dir(path)
    if kanban_dir:
        repo_path = re.sub(r"[/\\]+$", "", _with_base_name(kanban_dir, ""))
        return _placement(repo_path, _kanban_lane_id(repo_path), "kanban", kanban_dir, False, True)

    m = re.match(r"^(.+)-wt-(.+)$", base)
    if m:
        repo_path = _with_base_name(path, m.group(1))
        return _placement(repo_path, path, m.group(2), path, False, False)

    return _placement(path, path, base, path, True, False)


def _place(cwd: str, branch: str, resolve: Optional[Resolve], persisted_root: str) -> Optional[dict]:
    info = resolve(cwd) if resolve else None

    if info and info.get("repo_root") and info.get("worktree_root"):
        repo_root = info["repo_root"]
        worktree_root = info["worktree_root"]
        is_main = worktree_root == repo_root or bool(info.get("is_main"))

        if is_main:
            b = (branch or "").strip() or DEFAULT_BRANCH_LABEL
            return _placement(repo_root, _branch_lane_id(repo_root, b), b, repo_root, True, False)

        kanban_dir = kanban_worktree_dir(worktree_root)
        if kanban_dir:
            return _placement(repo_root, _kanban_lane_id(repo_root), "kanban", kanban_dir, False, True)

        label = base_name(worktree_root) or worktree_root
        return _placement(repo_root, worktree_root, label, worktree_root, False, False)

    # No live probe: trust the backend-persisted root.
    if persisted_root:
        kanban_dir = kanban_worktree_dir(cwd)
        if kanban_dir:
            return _placement(persisted_root, _kanban_lane_id(persisted_root), "kanban", kanban_dir, False, True)
        b = (branch or "").strip() or DEFAULT_BRANCH_LABEL
        return _placement(persisted_root, _branch_lane_id(persisted_root, b), b, persisted_root, True, False)

    return _place_by_heuristic(cwd)


def _session_repo_root(session: dict, resolve: Optional[Resolve]) -> str:
    """The COMMON repo root a session belongs to (folds linked worktrees)."""
    cwd = (session.get("cwd") or "").strip()
    if cwd and resolve:
        info = resolve(cwd)
        if info and info.get("repo_root"):
            return info["repo_root"]
    return (session.get("git_repo_root") or "").strip()


# ---------------------------------------------------------------------------
# Ordering + label disambiguation
# ---------------------------------------------------------------------------


def _session_time(session: dict) -> float:
    """Best-effort timestamp for a session (for sorting)."""
    try:
        return float(session.get("started_at") or session.get("last_active") or 0)
    except (TypeError, ValueError):
        return 0.0


def _lane_sort_key(group: dict) -> tuple:
    is_trunk = bool(group.get("isMain")) and group.get("label", group.get("name", "")).lower() in _TRUNK_BRANCHES
    is_kanban = bool(group.get("isKanban"))
    activity = max((_session_time(s) for s in group.get("sessions") or []), default=0.0)
    return (
        0 if is_trunk else 1,
        1 if is_kanban else 0,
        -activity,
        group.get("label", group.get("name", "")).lower(),
    )


def _sort_lanes(groups: list[dict]) -> list[dict]:
    return sorted(groups, key=_lane_sort_key)


def _disambiguate_labels(items: list[dict]) -> None:
    """Grow colliding basenames into path-prefixed labels (in place)."""
    by_label: dict[str, list[dict]] = {}
    for item in items:
        label = item.get("label", "")
        by_label.setdefault(label, []).append(item)

    for label, group in by_label.items():
        if len(group) <= 1:
            continue
        # Collision: grow each label with its parent dir basename.
        for item in group:
            path = item.get("path", "")
            parent = base_name(re.sub(r"[/\\][^/\\]+$", "", path))
            if parent:
                item["label"] = f"{parent}/{item['label']}"


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------


def build_tree(
    sessions: list[dict],
    *,
    resolve: Optional[Resolve] = None,
    explicit_projects: list[dict] | None = None,
) -> list[dict]:
    """Build the project → repo → lane → session tree.

    Args:
        sessions: List of session dicts (each must have ``cwd``, optionally
            ``git_branch``, ``git_repo_root``, ``id``, ``started_at``).
        resolve: Optional git probe function (``cwd → {repo_root, worktree_root}``).
        explicit_projects: Optional list of explicit project definitions
            (from projects.db), each with ``id``, ``name``, ``cwd``.

    Returns:
        A list of project nodes, each containing:
        - ``id``: project id (``p_<hex>`` or repo root path)
        - ``name``: display name
        - ``repos``: list of repo nodes
        Each repo node contains:
        - ``id``: repo root path
        - ``name``: display name
        - ``lanes``: list of lane nodes
        Each lane node contains:
        - ``id``: lane key
        - ``name``: display name
        - ``isMain``: True if this is the trunk lane
        - ``isKanban``: True if this is a kanban lane
        - ``sessions``: list of session dicts
    """
    # Group sessions by explicit project first, then by repo root.
    explicit_map: dict[str, list[dict]] = {}
    auto_sessions: list[dict] = []

    if explicit_projects:
        for proj in explicit_projects:
            explicit_map.setdefault(proj.get("id", ""), [])
        for session in sessions:
            cwd = (session.get("cwd") or "").strip()
            matched = False
            for proj in explicit_projects:
                proj_cwd = (proj.get("cwd") or "").strip()
                if proj_cwd and _is_path_under(proj_cwd, cwd):
                    explicit_map.setdefault(proj["id"], []).append(session)
                    matched = True
                    break
            if not matched:
                auto_sessions.append(session)
    else:
        auto_sessions = list(sessions)

    # Group auto sessions by repo root.
    repo_groups: dict[str, list[dict]] = {}
    for session in auto_sessions:
        root = _session_repo_root(session, resolve) or (session.get("cwd") or "")
        repo_groups.setdefault(root, []).append(session)

    # Build project nodes.
    projects: list[dict] = []

    # Explicit projects first.
    for proj in explicit_projects or []:
        proj_sessions = explicit_map.get(proj.get("id", []), [])
        if not proj_sessions:
            continue
        repos = _build_repos(proj_sessions, resolve)
        projects.append({
            "id": proj.get("id", ""),
            "name": proj.get("name", "Project"),
            "repos": repos,
        })

    # Auto-discovered projects (one per repo root).
    for root, root_sessions in sorted(repo_groups.items(), key=lambda kv: -max((_session_time(s) for s in kv[1]), default=0)):
        repos = _build_repos(root_sessions, resolve)
        projects.append({
            "id": root,
            "name": base_name(root) or root,
            "repos": repos,
        })

    return projects


def _build_repos(sessions: list[dict], resolve: Optional[Resolve]) -> list[dict]:
    """Build repo nodes for a set of sessions."""
    # Group sessions by repo root.
    by_repo: dict[str, list[dict]] = {}
    for session in sessions:
        root = _session_repo_root(session, resolve) or (session.get("cwd") or "")
        by_repo.setdefault(root, []).append(session)

    repos: list[dict] = []
    for repo_root, repo_sessions in by_repo.items():
        lanes = _build_lanes(repo_sessions, resolve, repo_root)
        repos.append({
            "id": repo_root,
            "name": base_name(repo_root) or repo_root,
            "lanes": lanes,
        })

    return repos


def _build_lanes(sessions: list[dict], resolve: Optional[Resolve], repo_root: str) -> list[dict]:
    """Build lane nodes for sessions within a repo."""
    # Group sessions by lane.
    by_lane: dict[str, dict] = {}
    for session in sessions:
        cwd = (session.get("cwd") or "").strip()
        branch = (session.get("git_branch") or "").strip()
        persisted_root = (session.get("git_repo_root") or "").strip()

        placement = _place(cwd, branch, resolve, persisted_root)
        if placement is None:
            continue

        lane_key = placement["lane_key"]
        if lane_key not in by_lane:
            by_lane[lane_key] = {
                "id": lane_key,
                "name": placement["lane_label"],
                "path": placement["lane_path"],
                "isMain": placement["is_main"],
                "isKanban": placement["is_kanban"],
                "sessions": [],
            }
        by_lane[lane_key]["sessions"].append(session)

    # Sort lanes: trunk first, kanban last, then by activity.
    lanes = list(by_lane.values())
    for lane in lanes:
        lane["sessions"].sort(key=lambda s: -_session_time(s))

    return _sort_lanes(lanes)


__all__ = [
    "Resolve",
    "base_name",
    "build_tree",
    "kanban_worktree_dir",
]
