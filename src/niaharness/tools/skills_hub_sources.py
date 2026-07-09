"""GitHub skill source adapter — fetch skills from GitHub repos.

Ported from Hermes Agent's ``tools/skills_hub.py`` (4,109 LOC), scoped
to the GitHubSource + GitHubAuth classes. Provides:

  - :class:`GitHubAuth` — four-tier auth resolution (PAT → gh CLI →
    GitHub App → anonymous).
  - :class:`GitHubSource` — fetches skills via the GitHub Contents API.
    Supports search across default taps + arbitrary repos. Returns
    :class:`SkillBundle` objects that feed NIA's existing quarantine →
    scan → install pipeline.
  - :func:`get_default_sources` — factory returning
    ``[OptionalSkillSource(), GitHubSource(auth=None)]``.

Default taps: openai/skills, anthropics/skills, huggingface/skills,
NVIDIA/skills, garrytan/gstack.

Usage::

    from niaharness.tools.skills_hub_sources import GitHubSource, GitHubAuth

    source = GitHubSource(GitHubAuth())
    results = source.search("git")
    bundle = source.fetch("anthropics/skills/skill-name")
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INDEX_CACHE_TTL = 3600  # 1 hour.
_GITHUB_API = "https://api.github.com"
_DEFAULT_TIMEOUT = 15.0
_MAX_RETRIES = 3

# Trusted repos (higher trust level).
TRUSTED_REPOS = frozenset({
    "openai/skills",
    "anthropics/skills",
    "huggingface/skills",
    "nvidia/skills",
})

# Provider labels per repo.
GITHUB_TAP_PROVIDERS = {
    "openai/skills": "OpenAI",
    "anthropics/skills": "Anthropic",
    "huggingface/skills": "HuggingFace",
    "nvidia/skills": "NVIDIA",
    "garrytan/gstack": "gstack",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SkillMeta:
    """Metadata for a skill available from a source."""
    name: str
    description: str
    source: str  # "github", "well-known", etc.
    identifier: str  # e.g. "openai/skills/skill-creator"
    trust_level: str  # "builtin" | "trusted" | "community"
    repo: Optional[str] = None
    path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillBundle:
    """A downloaded skill ready for quarantine/scanning/installation."""
    name: str
    files: Dict[str, str]  # relative_path -> file content
    source: str  # "github", "well-known", ...
    identifier: str
    trust_level: str  # "builtin" | "trusted" | "community"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GitHubAuth
# ---------------------------------------------------------------------------


class GitHubAuth:
    """GitHub API authentication with four-tier resolution.

    Priority order:
      1. ``GITHUB_TOKEN`` / ``GH_TOKEN`` env var (PAT — raises limit to 5000/hr).
      2. ``gh auth token`` subprocess (if gh CLI is installed).
      3. GitHub App JWT + installation token (if app credentials configured).
      4. Anonymous (60 req/hr, public repos only).
    """

    def __init__(self) -> None:
        self._cached_token: Optional[str] = None
        self._cached_method: Optional[str] = None
        self._app_token_expiry: float = 0

    def get_headers(self) -> Dict[str, str]:
        """Return authorization headers for GitHub API requests."""
        token = self._resolve_token()
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"
        return headers

    def is_authenticated(self) -> bool:
        return self._resolve_token() is not None

    def auth_method(self) -> str:
        """Return which auth method is active."""
        self._resolve_token()
        return self._cached_method or "anonymous"

    def _resolve_token(self) -> Optional[str]:
        if self._cached_token:
            if self._cached_method != "github-app" or time.time() < self._app_token_expiry:
                return self._cached_token

        # 1. Environment variable (PAT).
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            self._cached_token = token
            self._cached_method = "pat"
            return token

        # 2. gh CLI.
        token = self._try_gh_cli()
        if token:
            self._cached_token = token
            self._cached_method = "gh-cli"
            return token

        # 3. GitHub App.
        token = self._try_github_app()
        if token:
            self._cached_token = token
            self._cached_method = "github-app"
            self._app_token_expiry = time.time() + 3500  # ~58 min.
            return token

        self._cached_method = "anonymous"
        return None

    def _try_gh_cli(self) -> Optional[str]:
        """Try to get a token from the gh CLI."""
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True, text=True, timeout=5,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    def _try_github_app(self) -> Optional[str]:
        """Try GitHub App JWT authentication."""
        app_id = os.environ.get("GITHUB_APP_ID")
        key_path = os.environ.get("GITHUB_APP_PRIVATE_KEY_PATH")
        installation_id = os.environ.get("GITHUB_APP_INSTALLATION_ID")
        if not all([app_id, key_path, installation_id]):
            return None
        try:
            import jwt as pyjwt  # type: ignore[import-untyped]
        except ImportError:
            return None
        try:
            key_file = Path(key_path)
            if not key_file.exists():
                return None
            private_key = key_file.read_text(encoding="utf-8")
            now = int(time.time())
            payload = {"iat": now - 60, "exp": now + 600, "iss": app_id}
            encoded_jwt = pyjwt.encode(payload, private_key, algorithm="RS256")
            import httpx
            resp = httpx.post(
                f"{_GITHUB_API}/app/installations/{installation_id}/access_tokens",
                headers={
                    "Authorization": f"Bearer {encoded_jwt}",
                    "Accept": "application/vnd.github.v3+json",
                },
                timeout=10,
            )
            if resp.status_code == 201:
                return resp.json().get("token")
        except Exception as exc:
            logger.debug("GitHub App auth failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# GitHubSource
# ---------------------------------------------------------------------------


class GitHubSource:
    """Fetch skills from GitHub repos via the Contents API.

    Supports searching across default taps (openai/skills,
    anthropics/skills, etc.) and fetching individual skills by
    identifier (``owner/repo/path/to/skill``).
    """

    DEFAULT_TAPS: List[Dict[str, str]] = [
        {"repo": "openai/skills", "path": "skills/.curated/"},
        {"repo": "openai/skills", "path": "skills/.system/"},
        {"repo": "anthropics/skills", "path": "skills/"},
        {"repo": "huggingface/skills", "path": "skills/"},
        {"repo": "NVIDIA/skills", "path": "skills/"},
        {"repo": "garrytan/gstack", "path": ""},
    ]

    def __init__(
        self,
        auth: Optional[GitHubAuth] = None,
        *,
        extra_taps: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self.auth = auth or GitHubAuth()
        self.taps = list(self.DEFAULT_TAPS)
        if extra_taps:
            self.taps.extend(extra_taps)
        self._tree_cache: Dict[str, Tuple[str, List[dict]]] = {}
        self._rate_limited: bool = False

    def source_id(self) -> str:
        return "github"

    @property
    def is_rate_limited(self) -> bool:
        return self._rate_limited

    def trust_level_for(self, identifier: str) -> str:
        """Return 'trusted' or 'community' based on the repo."""
        parts = identifier.split("/", 2)
        if len(parts) >= 2:
            repo = f"{parts[0]}/{parts[1]}"
            if repo in TRUSTED_REPOS:
                return "trusted"
        return "community"

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        """Search all taps for skills matching *query*."""
        results: List[SkillMeta] = []
        query_lower = query.lower()

        for tap in self.taps:
            try:
                skills = self._list_skills_in_repo(tap["repo"], tap.get("path", ""))
                for skill in skills:
                    searchable = f"{skill.name} {skill.description} {' '.join(skill.tags)}".lower()
                    if query_lower in searchable:
                        results.append(skill)
            except Exception as exc:
                logger.debug("Failed to search %s: %s", tap["repo"], exc)
                continue

        # Deduplicate by identifier, preferring higher trust.
        _trust_rank = {"builtin": 2, "trusted": 1, "community": 0}
        seen: Dict[str, SkillMeta] = {}
        for r in results:
            if r.identifier not in seen:
                seen[r.identifier] = r
            elif _trust_rank.get(r.trust_level, 0) > _trust_rank.get(seen[r.identifier].trust_level, 0):
                seen[r.identifier] = r

        return list(seen.values())[:limit]

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        """Download a skill from GitHub.

        Args:
            identifier: Format ``owner/repo/path/to/skill-dir``.

        Returns:
            :class:`SkillBundle` or ``None`` if not found / no SKILL.md.
        """
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            return None

        repo = f"{parts[0]}/{parts[1]}"
        skill_path = parts[2]

        files = self._download_directory(repo, skill_path)
        if not files or "SKILL.md" not in files:
            return None

        skill_name = skill_path.rstrip("/").split("/")[-1]
        trust = self.trust_level_for(identifier)

        return SkillBundle(
            name=skill_name,
            files=files,
            source="github",
            identifier=identifier,
            trust_level=trust,
        )

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        """Fetch just the SKILL.md for preview."""
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            return None
        repo = f"{parts[0]}/{parts[1]}"
        skill_path = parts[2]
        skill_name = skill_path.rstrip("/").split("/")[-1]

        content = self._fetch_file_content(repo, f"{skill_path.rstrip('/')}/SKILL.md")
        if not content:
            return None

        frontmatter = _parse_frontmatter_quick(content)
        description = frontmatter.get("description", "") if frontmatter else ""
        tags = frontmatter.get("tags", []) if frontmatter else []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        provider = GITHUB_TAP_PROVIDERS.get(repo.lower())
        return SkillMeta(
            name=skill_name,
            description=description or content[:200],
            source="github",
            identifier=identifier,
            trust_level=self.trust_level_for(identifier),
            repo=repo,
            path=skill_path,
            tags=tags if isinstance(tags, list) else [],
            extra={"provider": provider} if provider else {},
        )

    # ------------------------------------------------------------------
    # Internal: API calls
    # ------------------------------------------------------------------

    def _list_skills_in_repo(self, repo: str, path: str) -> List[SkillMeta]:
        """List skills in a repo path via the Contents API."""
        url = f"{_GITHUB_API}/repos/{repo}/contents/{path}" if path else f"{_GITHUB_API}/repos/{repo}/contents"
        resp = self._github_get(url)
        if resp is None or resp.status_code != 200:
            return []

        results: List[SkillMeta] = []
        for entry in resp.json():
            if not isinstance(entry, dict):
                continue
            if entry.get("type") != "dir":
                continue
            name = entry.get("name", "")
            if name.startswith(".") or name.startswith("_"):
                continue
            identifier = f"{repo}/{path.rstrip('/')}/{name}" if path else f"{repo}/{name}"
            meta = self.inspect(identifier)
            if meta:
                results.append(meta)
            else:
                # Still list it even if SKILL.md fetch failed.
                results.append(SkillMeta(
                    name=name,
                    description="",
                    source="github",
                    identifier=identifier,
                    trust_level=self.trust_level_for(identifier),
                    repo=repo,
                    path=f"{path}/{name}" if path else name,
                ))
        return results

    def _download_directory(self, repo: str, path: str) -> Dict[str, str]:
        """Download all files in a directory from a GitHub repo."""
        # Primary: use the trees API for a single round-trip.
        tree_result = self._download_directory_via_tree(repo, path)
        if tree_result is not None:
            return tree_result
        # Fallback: recursive Contents API walk.
        return self._download_directory_recursive(repo, path)

    def _download_directory_via_tree(self, repo: str, path: str) -> Optional[Dict[str, str]]:
        """Download via the git trees API (single call)."""
        tree = self._get_repo_tree(repo)
        if tree is None:
            return None
        _branch, entries = tree
        prefix = path.rstrip("/") + "/"
        files: Dict[str, str] = {}
        for entry in entries:
            entry_path = entry.get("path", "")
            if not entry_path.startswith(prefix):
                continue
            if entry.get("type") != "blob":
                continue
            rel_path = entry_path[len(prefix):]
            content = self._fetch_file_content(repo, entry_path)
            if content is not None:
                files[rel_path] = content
        return files

    def _download_directory_recursive(self, repo: str, path: str) -> Dict[str, str]:
        """Fallback: walk Contents API one directory at a time."""
        url = f"{_GITHUB_API}/repos/{repo}/contents/{path}"
        resp = self._github_get(url)
        if resp is None or resp.status_code != 200:
            return {}
        files: Dict[str, str] = {}
        for entry in resp.json():
            if not isinstance(entry, dict):
                continue
            entry_type = entry.get("type")
            entry_name = entry.get("name", "")
            entry_path = f"{path}/{entry_name}" if path else entry_name
            if entry_type == "file":
                content = self._fetch_file_content(repo, entry_path)
                if content is not None:
                    files[entry_name] = content
            elif entry_type == "dir":
                sub_files = self._download_directory_recursive(repo, entry_path)
                for sub_name, sub_content in sub_files.items():
                    files[f"{entry_name}/{sub_name}"] = sub_content
        return files

    def _get_repo_tree(self, repo: str) -> Optional[Tuple[str, List[dict]]]:
        """Get the recursive file tree for a repo (cached per instance)."""
        if repo in self._tree_cache:
            return self._tree_cache[repo]
        # Get default branch.
        resp = self._github_get(f"{_GITHUB_API}/repos/{repo}")
        if resp is None or resp.status_code != 200:
            return None
        default_branch = resp.json().get("default_branch", "main")
        # Get tree.
        tree_resp = self._github_get(
            f"{_GITHUB_API}/repos/{repo}/git/trees/{default_branch}",
            params={"recursive": "1"},
            timeout=30.0,
        )
        if tree_resp is None or tree_resp.status_code != 200:
            return None
        tree_data = tree_resp.json()
        if tree_data.get("truncated"):
            logger.warning("GitHub tree for %s is truncated — falling back to recursive", repo)
            return None
        entries = tree_data.get("tree", [])
        self._tree_cache[repo] = (default_branch, entries)
        return (default_branch, entries)

    def _fetch_file_content(self, repo: str, path: str) -> Optional[str]:
        """Fetch raw file content from GitHub."""
        url = f"{_GITHUB_API}/repos/{repo}/contents/{quote(path, safe='/')}"
        resp = self._github_get(url, headers={"Accept": "application/vnd.github.v3.raw"})
        if resp is None or resp.status_code != 200:
            return None
        return resp.text

    def _github_get(
        self,
        url: str,
        *,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _MAX_RETRIES,
    ) -> Optional[Any]:
        """HTTP GET with retry/backoff for rate limits + 5xx."""
        import httpx

        merged_headers = self.auth.get_headers()
        if headers:
            merged_headers.update(headers)

        for attempt in range(max_retries):
            try:
                resp = httpx.get(
                    url,
                    headers=merged_headers,
                    params=params,
                    timeout=timeout,
                    follow_redirects=True,
                )
                # Check rate limit.
                if resp.status_code in (403, 429):
                    remaining = resp.headers.get("X-RateLimit-Remaining", "")
                    if remaining == "0" or resp.status_code == 429:
                        self._rate_limited = True
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait = min(int(retry_after), 60)
                                time.sleep(wait)
                                continue
                            except (ValueError, TypeError):
                                pass
                        # Exponential backoff.
                        time.sleep(min(2 ** attempt, 30))
                        continue
                if resp.status_code >= 500:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                return resp
            except Exception as exc:
                logger.debug("GitHub GET failed (attempt %d): %s", attempt + 1, exc)
                if attempt < max_retries - 1:
                    time.sleep(min(2 ** attempt, 30))
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_frontmatter_quick(content: str) -> Optional[dict]:
    """Parse YAML frontmatter from a SKILL.md file."""
    if not content.startswith("---"):
        return None
    end = content.find("\n---", 3)
    if end == -1:
        return None
    frontmatter_text = content[3:end].strip()
    try:
        import yaml
        return yaml.safe_load(frontmatter_text) or {}
    except ImportError:
        return None
    except Exception:
        return None


def github_provider_for(repo: str) -> Optional[str]:
    """Return the provider label for a GitHub tap repo."""
    return GITHUB_TAP_PROVIDERS.get(repo.strip().lower())


def get_default_sources() -> List[Any]:
    """Factory returning the default skill sources.

    Returns ``[GitHubSource(auth=GitHubAuth())]`` — the OptionalSkillSource
    (local well-known skills) is handled by NIA's existing skills loader.
    """
    return [GitHubSource(GitHubAuth())]


__all__ = [
    "GitHubAuth",
    "GitHubSource",
    "SkillBundle",
    "SkillMeta",
    "TRUSTED_REPOS",
    "GITHUB_TAP_PROVIDERS",
    "get_default_sources",
    "github_provider_for",
]
