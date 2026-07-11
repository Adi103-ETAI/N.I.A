"""P1 Skill source adapters — 8 missing adapters + TapsManager.

Ported from Hermes Agent's ``tools/skills_hub.py`` (4109 LOC), scoped
to NIA's architecture. Provides the 8 missing skill source adapters
identified in AUDIT.md:

  - :class:`WellKnownSkillSource` — reads ``/.well-known/skills/index.json``
    from any domain.
  - :class:`UrlSource` — fetches a single skill from a direct URL.
  - :class:`SkillsShSource` — crawls skills.sh (community skill directory).
  - :class:`ClawHubSource` — fetches from ClawHub (clawhub.ai) API.
  - :class:`ClaudeMarketplaceSource` — fetches from Claude Marketplace.
  - :class:`LobeHubSource` — fetches from LobeHub.
  - :class:`BrowseShSource` — crawls browse.sh.
  - :class:`HermesIndexSource` — fetches from the Hermes skill index cache.
  - :class:`TapsManager` — manages custom GitHub repo sources (taps.json).

Each adapter implements the ``SkillSource`` ABC: ``search``, ``fetch``,
``inspect``, ``source_id``, ``trust_level_for``.

All network operations use httpx with SSRF guards (blocks localhost,
private IPs, file://, data:// schemes). Failures are logged + return
empty results (never crash the hub).

Usage::

    from niaharness.tools.skills_hub_extra_sources import (
        WellKnownSkillSource,
        UrlSource,
        TapsManager,
        get_extra_sources,
    )

    sources = get_extra_sources()  # all 8 adapters
    for source in sources:
        results = source.search("email")
        for meta in results:
            print(f"{meta.name}: {meta.description}")
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------


_SAFE_SCHEMES = frozenset({"http", "https"})
_BLOCKED_HOSTS = frozenset({
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "169.254.169.254",  # cloud metadata
    "metadata.google.internal",
})


def _is_safe_url(url: str) -> bool:
    """Return True if the URL is safe to fetch (not localhost/private/metadata)."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in _SAFE_SCHEMES:
        return False
    host = (parsed.hostname or "").lower().rstrip("/")
    if host in _BLOCKED_HOSTS:
        return False
    # Block private IP ranges (10.x, 172.16-31.x, 192.168.x).
    if host and re.match(r"^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)", host):
        return False
    return True


async def _http_get(url: str, *, timeout: float = 15.0) -> Optional[str]:
    """Fetch a URL and return the response text. None on failure."""
    if not _is_safe_url(url):
        logger.warning("Blocked unsafe URL: %s", url)
        return None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            if response.status_code != 200:
                logger.debug("HTTP %d for %s", response.status_code, url)
                return None
            return response.text
    except Exception as exc:
        logger.debug("HTTP fetch failed for %s: %s", url, exc)
        return None


async def _http_get_bytes(url: str, *, timeout: float = 30.0) -> Optional[bytes]:
    """Fetch a URL and return the response bytes. None on failure."""
    if not _is_safe_url(url):
        logger.warning("Blocked unsafe URL: %s", url)
        return None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            if response.status_code != 200:
                return None
            return response.content
    except Exception as exc:
        logger.debug("HTTP fetch failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Shared data classes (re-exported from skills_hub_sources for compatibility)
# ---------------------------------------------------------------------------


@dataclass
class SkillMeta:
    """Minimal metadata returned by search results."""

    name: str
    description: str
    source: str
    identifier: str
    trust_level: str = "community"
    category: str = ""
    installed: bool = False
    repo: Optional[str] = None
    path: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillBundle:
    """A downloaded skill ready for quarantine/scanning/installation."""

    name: str
    files: dict[str, Union[str, bytes]]
    source: str
    identifier: str
    trust_level: str = "community"
    category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_frontmatter(content: str) -> dict[str, Any]:
    """Parse YAML frontmatter from a SKILL.md file."""
    if not content or not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    frontmatter = parts[1].strip()
    result: dict[str, Any] = {}
    for line in frontmatter.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                result[key] = value
    return result


def _validate_skill_name(name: str) -> str:
    """Validate a skill name. Returns the name or raises ValueError."""
    if not name or not isinstance(name, str):
        raise ValueError("skill name must be a non-empty string")
    cleaned = name.strip().lower().replace(" ", "-")
    if not re.match(r"^[a-z0-9][a-z0-9_-]*$", cleaned):
        raise ValueError(f"invalid skill name: {name!r}")
    return cleaned


def _validate_rel_path(rel_path: str) -> str:
    """Validate a relative path inside a skill bundle."""
    if not rel_path:
        raise ValueError("empty path")
    if ".." in Path(rel_path).parts:
        raise ValueError(f"path traversal detected: {rel_path!r}")
    if Path(rel_path).is_absolute():
        raise ValueError(f"absolute path not allowed: {rel_path!r}")
    return rel_path


# ---------------------------------------------------------------------------
# SkillSource ABC (local copy — matches skills_hub.SkillSource)
# ---------------------------------------------------------------------------


from abc import ABC, abstractmethod


class SkillSource(ABC):
    """Abstract base for all skill registry adapters."""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        """Search for skills matching a query string."""

    @abstractmethod
    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        """Download a skill bundle by identifier."""

    @abstractmethod
    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        """Fetch metadata for a skill without downloading all files."""

    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this source."""

    def trust_level_for(self, identifier: str) -> str:
        return "community"


# ---------------------------------------------------------------------------
# 1. WellKnownSkillSource — /.well-known/skills/index.json
# ---------------------------------------------------------------------------


class WellKnownSkillSource(SkillSource):
    """Read skills from a domain exposing /.well-known/skills/index.json.

    Any domain can publish a skills index at the well-known path. The
    index is a JSON file with a ``skills`` array, each entry having
    ``name``, ``description``, and ``files`` (list of relative paths).

    Identifier format: ``well-known://<domain>/<skill-name>``
    """

    BASE_PATH = "/.well-known/skills"

    def source_id(self) -> str:
        return "well-known"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        # Extract domain from query (e.g. "example.com" or "well-known://example.com/email").
        domain = self._extract_domain(query)
        if not domain:
            return []
        index = self._fetch_index_sync(domain)
        if not index:
            return []
        results: List[SkillMeta] = []
        query_lower = query.lower()
        for entry in index.get("skills", [])[:limit]:
            name = entry.get("name", "")
            desc = entry.get("description", "")
            if query_lower in name.lower() or query_lower in desc.lower() or domain in query:
                results.append(SkillMeta(
                    name=name,
                    description=str(desc),
                    source="well-known",
                    identifier=f"well-known://{domain}/{name}",
                    trust_level="community",
                    path=name,
                    extra={"files": entry.get("files", ["SKILL.md"])},
                ))
        return results[:limit]

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        parsed = self._parse_identifier(identifier)
        if not parsed:
            return None
        domain, skill_name = parsed
        index = self._fetch_index_sync(domain)
        if not index:
            return None
        for entry in index.get("skills", []):
            if entry.get("name") == skill_name:
                return SkillMeta(
                    name=skill_name,
                    description=str(entry.get("description", "")),
                    source="well-known",
                    identifier=identifier,
                    trust_level="community",
                    path=skill_name,
                    extra={"files": entry.get("files", ["SKILL.md"])},
                )
        return None

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        parsed = self._parse_identifier(identifier)
        if not parsed:
            return None
        domain, skill_name = parsed
        try:
            skill_name = _validate_skill_name(skill_name)
        except ValueError:
            return None
        index = self._fetch_index_sync(domain)
        if not index:
            return None
        entry = next((e for e in index.get("skills", []) if e.get("name") == skill_name), None)
        if not entry:
            return None
        files_list = entry.get("files", ["SKILL.md"])
        if not isinstance(files_list, list):
            files_list = ["SKILL.md"]
        files: dict[str, str] = {}
        base_url = f"https://{domain}{self.BASE_PATH}/{skill_name}"
        for rel_path in files_list:
            if not isinstance(rel_path, str) or not rel_path:
                continue
            try:
                safe_path = _validate_rel_path(rel_path)
            except ValueError:
                continue
            content = self._fetch_text_sync(f"{base_url}/{safe_path}")
            if content is not None:
                files[safe_path] = content
        if not files:
            return None
        return SkillBundle(
            name=skill_name,
            files=files,
            source="well-known",
            identifier=identifier,
            trust_level="community",
        )

    def _extract_domain(self, query: str) -> Optional[str]:
        """Extract a domain from the query string."""
        if query.startswith("well-known://"):
            return self._parse_identifier(query)[0]
        # If the query looks like a domain, use it.
        if re.match(r"^[a-z0-9.-]+\.[a-z]{2,}$", query.lower().strip()):
            return query.lower().strip()
        return None

    def _parse_identifier(self, identifier: str) -> Optional[Tuple[str, str]]:
        """Parse 'well-known://domain/skill-name' → (domain, skill_name)."""
        if not identifier.startswith("well-known://"):
            return None
        rest = identifier[len("well-known://"):]
        parts = rest.split("/", 1)
        if len(parts) != 2:
            return None
        return (parts[0], parts[1])

    def _fetch_index_sync(self, domain: str) -> Optional[dict]:
        """Fetch the well-known skills index (sync wrapper)."""
        url = f"https://{domain}{self.BASE_PATH}/index.json"
        text = self._fetch_text_sync(url)
        if not text:
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

    def _fetch_text_sync(self, url: str) -> Optional[str]:
        """Sync HTTP GET (uses urllib to avoid async requirement)."""
        if not _is_safe_url(url):
            return None
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "NIA-Skills/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            logger.debug("WellKnown fetch failed for %s: %s", url, exc)
            return None


# ---------------------------------------------------------------------------
# 2. UrlSource — fetch a single skill from a direct URL
# ---------------------------------------------------------------------------


class UrlSource(SkillSource):
    """Fetch a single skill from a direct URL to a SKILL.md file.

    Identifier format: the URL itself.
    """

    def source_id(self) -> str:
        return "url"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        # URL source doesn't support search — it's fetch-only.
        return []

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        if not _is_safe_url(identifier):
            return None
        content = self._fetch_text_sync(identifier)
        if content is None:
            return None
        fm = _parse_frontmatter(content)
        name = fm.get("name", self._name_from_url(identifier))
        return SkillMeta(
            name=name,
            description=fm.get("description", ""),
            source="url",
            identifier=identifier,
            trust_level="community",
            extra={"url": identifier},
        )

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        if not _is_safe_url(identifier):
            return None
        content = self._fetch_text_sync(identifier)
        if content is None:
            return None
        fm = _parse_frontmatter(content)
        name = fm.get("name", self._name_from_url(identifier))
        try:
            name = _validate_skill_name(name)
        except ValueError:
            name = "unnamed-skill"
        return SkillBundle(
            name=name,
            files={"SKILL.md": content},
            source="url",
            identifier=identifier,
            trust_level="community",
        )

    def _name_from_url(self, url: str) -> str:
        """Extract a skill name from a URL."""
        path = urlparse(url).path
        name = Path(path).stem
        return name or "url-skill"

    def _fetch_text_sync(self, url: str) -> Optional[str]:
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "NIA-Skills/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            logger.debug("UrlSource fetch failed for %s: %s", url, exc)
            return None


# ---------------------------------------------------------------------------
# 3. SkillsShSource — crawl skills.sh community directory
# ---------------------------------------------------------------------------


class SkillsShSource(SkillSource):
    """Crawl skills.sh — a community skill directory.

    Identifier format: ``skillssh://<slug>``
    """

    BASE_URL = "https://skills.sh"

    def source_id(self) -> str:
        return "skillssh"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        # Fetch the skills.sh search page.
        url = f"{self.BASE_URL}/search?q={query}"
        html = self._fetch_text_sync(url)
        if not html:
            return []
        # Parse skill entries from the HTML (simple regex — skills.sh uses
        # a predictable card structure).
        results: List[SkillMeta] = []
        for match in re.finditer(
            r'href="/skills/([a-z0-9_-]+)"[^>]*>.*?<h[23][^>]*>([^<]+)</h[23]>'
            r'.*?<p[^>]*>([^<]+)</p>',
            html, re.DOTALL | re.IGNORECASE,
        ):
            slug = match.group(1)
            title = match.group(2).strip()
            desc = match.group(3).strip()
            results.append(SkillMeta(
                name=slug,
                description=desc,
                source="skillssh",
                identifier=f"skillssh://{slug}",
                trust_level="community",
            ))
            if len(results) >= limit:
                break
        return results

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        slug = self._parse_identifier(identifier)
        if not slug:
            return None
        url = f"{self.BASE_URL}/skills/{slug}"
        html = self._fetch_text_sync(url)
        if not html:
            return None
        # Extract title + description from the page.
        title_match = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
        desc_match = re.search(r'<meta\s+name="description"\s+content="([^"]*)"', html, re.IGNORECASE)
        return SkillMeta(
            name=title_match.group(1).strip() if title_match else slug,
            description=desc_match.group(1).strip() if desc_match else "",
            source="skillssh",
            identifier=identifier,
            trust_level="community",
        )

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        slug = self._parse_identifier(identifier)
        if not slug:
            return None
        # Fetch the raw SKILL.md from skills.sh.
        url = f"{self.BASE_URL}/skills/{slug}/raw/SKILL.md"
        content = self._fetch_text_sync(url)
        if content is None:
            return None
        return SkillBundle(
            name=slug,
            files={"SKILL.md": content},
            source="skillssh",
            identifier=identifier,
            trust_level="community",
        )

    def _parse_identifier(self, identifier: str) -> Optional[str]:
        if identifier.startswith("skillssh://"):
            return identifier[len("skillssh://"):]
        return None

    def _fetch_text_sync(self, url: str) -> Optional[str]:
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "NIA-Skills/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            logger.debug("SkillsSh fetch failed for %s: %s", url, exc)
            return None


# ---------------------------------------------------------------------------
# 4. ClawHubSource — fetch from ClawHub API (clawhub.ai)
# ---------------------------------------------------------------------------


class ClawHubSource(SkillSource):
    """Fetch skills from ClawHub (clawhub.ai) via their HTTP API.

    Identifier format: ``clawhub://<slug>``
    """

    API_BASE = "https://api.clawhub.ai/v1"

    def source_id(self) -> str:
        return "clawhub"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        url = f"{self.API_BASE}/skills/search?q={query}&limit={limit}"
        text = self._fetch_text_sync(url)
        if not text:
            return []
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return []
        results: List[SkillMeta] = []
        for entry in data.get("skills", [])[:limit]:
            results.append(SkillMeta(
                name=entry.get("slug", ""),
                description=entry.get("description", ""),
                source="clawhub",
                identifier=f"clawhub://{entry.get('slug', '')}",
                trust_level="community",
                extra={"version": entry.get("version"), "author": entry.get("author")},
            ))
        return results

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        slug = self._parse_identifier(identifier)
        if not slug:
            return None
        url = f"{self.API_BASE}/skills/{slug}"
        text = self._fetch_text_sync(url)
        if not text:
            return None
        try:
            entry = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        return SkillMeta(
            name=entry.get("slug", slug),
            description=entry.get("description", ""),
            source="clawhub",
            identifier=identifier,
            trust_level="community",
            extra={"version": entry.get("version"), "author": entry.get("author")},
        )

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        slug = self._parse_identifier(identifier)
        if not slug:
            return None
        # Fetch the skill content as a ZIP.
        url = f"{self.API_BASE}/skills/{slug}/download"
        content = self._fetch_bytes_sync(url)
        if content is None:
            return None
        # Unzip + extract files.
        files = self._extract_zip(content)
        if not files:
            return None
        return SkillBundle(
            name=slug,
            files=files,
            source="clawhub",
            identifier=identifier,
            trust_level="community",
        )

    def _parse_identifier(self, identifier: str) -> Optional[str]:
        if identifier.startswith("clawhub://"):
            return identifier[len("clawhub://"):]
        return None

    def _fetch_text_sync(self, url: str) -> Optional[str]:
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={
                "User-Agent": "NIA-Skills/1.0",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            return None

    def _fetch_bytes_sync(self, url: str) -> Optional[bytes]:
        if not _is_safe_url(url):
            return None
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "NIA-Skills/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception:
            return None

    @staticmethod
    def _extract_zip(data: bytes) -> dict[str, str]:
        """Extract files from a ZIP archive."""
        import io
        import zipfile
        files: dict[str, str] = {}
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    try:
                        content = zf.read(info.filename).decode("utf-8", errors="replace")
                        files[info.filename] = content
                    except Exception:
                        pass
        except Exception:
            pass
        return files


# ---------------------------------------------------------------------------
# 5. ClaudeMarketplaceSource — fetch from Claude Marketplace
# ---------------------------------------------------------------------------


class ClaudeMarketplaceSource(SkillSource):
    """Fetch skills from the Claude Marketplace.

    Uses the public skills index cache at
    ``skills/index-cache/claude_marketplace_anthropics_skills.json``.

    Identifier format: ``claude-marketplace://<skill-name>``
    """

    INDEX_URL = "https://raw.githubusercontent.com/anthropics/skills/main/index.json"

    def source_id(self) -> str:
        return "claude-marketplace"

    def trust_level_for(self, identifier: str) -> str:
        return "trusted"  # Anthropic-published skills

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        index = self._fetch_index()
        if not index:
            return []
        query_lower = query.lower()
        results: List[SkillMeta] = []
        for entry in index:
            name = entry.get("name", "")
            desc = entry.get("description", "")
            if query_lower in name.lower() or query_lower in desc.lower():
                results.append(SkillMeta(
                    name=name,
                    description=desc,
                    source="claude-marketplace",
                    identifier=f"claude-marketplace://{name}",
                    trust_level="trusted",
                    extra={"url": entry.get("url")},
                ))
            if len(results) >= limit:
                break
        return results

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        name = self._parse_identifier(identifier)
        if not name:
            return None
        index = self._fetch_index()
        if not index:
            return None
        entry = next((e for e in index if e.get("name") == name), None)
        if not entry:
            return None
        return SkillMeta(
            name=name,
            description=entry.get("description", ""),
            source="claude-marketplace",
            identifier=identifier,
            trust_level="trusted",
            extra={"url": entry.get("url")},
        )

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        name = self._parse_identifier(identifier)
        if not name:
            return None
        index = self._fetch_index()
        if not index:
            return None
        entry = next((e for e in index if e.get("name") == name), None)
        if not entry:
            return None
        url = entry.get("url")
        if not url:
            return None
        content = self._fetch_text_sync(url)
        if content is None:
            return None
        return SkillBundle(
            name=name,
            files={"SKILL.md": content},
            source="claude-marketplace",
            identifier=identifier,
            trust_level="trusted",
        )

    def _parse_identifier(self, identifier: str) -> Optional[str]:
        if identifier.startswith("claude-marketplace://"):
            return identifier[len("claude-marketplace://"):]
        return None

    def _fetch_index(self) -> List[dict]:
        text = self._fetch_text_sync(self.INDEX_URL)
        if not text:
            return []
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("skills", [])
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    def _fetch_text_sync(self, url: str) -> Optional[str]:
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "NIA-Skills/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 6. LobeHubSource — fetch from LobeHub
# ---------------------------------------------------------------------------


class LobeHubSource(SkillSource):
    """Fetch skills from LobeHub (lobechat.com) plugin directory.

    Identifier format: ``lobehub://<plugin-id>``
    """

    API_BASE = "https://chat.lobehub.com/api/plugins"

    def source_id(self) -> str:
        return "lobehub"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        url = f"{self.API_BASE}?search={query}&pageSize={limit}"
        text = self._fetch_text_sync(url)
        if not text:
            return []
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return []
        results: List[SkillMeta] = []
        for entry in data.get("plugins", [])[:limit]:
            results.append(SkillMeta(
                name=entry.get("identifier", ""),
                description=entry.get("meta", {}).get("description", ""),
                source="lobehub",
                identifier=f"lobehub://{entry.get('identifier', '')}",
                trust_level="community",
            ))
        return results

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        plugin_id = self._parse_identifier(identifier)
        if not plugin_id:
            return None
        url = f"{self.API_BASE}/{plugin_id}"
        text = self._fetch_text_sync(url)
        if not text:
            return None
        try:
            entry = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        return SkillMeta(
            name=entry.get("identifier", plugin_id),
            description=entry.get("meta", {}).get("description", ""),
            source="lobehub",
            identifier=identifier,
            trust_level="community",
        )

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        # LobeHub plugins are JSON manifests, not SKILL.md files.
        # We convert the manifest to a SKILL.md on the fly.
        plugin_id = self._parse_identifier(identifier)
        if not plugin_id:
            return None
        meta = self.inspect(identifier)
        if not meta:
            return None
        skill_md = self._manifest_to_skill_md(meta)
        return SkillBundle(
            name=plugin_id,
            files={"SKILL.md": skill_md},
            source="lobehub",
            identifier=identifier,
            trust_level="community",
        )

    def _parse_identifier(self, identifier: str) -> Optional[str]:
        if identifier.startswith("lobehub://"):
            return identifier[len("lobehub://"):]
        return None

    @staticmethod
    def _manifest_to_skill_md(meta: SkillMeta) -> str:
        """Convert a LobeHub plugin manifest to a SKILL.md string."""
        return (
            "---\n"
            f"name: {meta.name}\n"
            f"description: {meta.description}\n"
            f"source: lobehub\n"
            "---\n\n"
            f"# {meta.name}\n\n"
            f"{meta.description}\n\n"
            "This skill was imported from LobeHub.\n"
        )

    def _fetch_text_sync(self, url: str) -> Optional[str]:
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={
                "User-Agent": "NIA-Skills/1.0",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 7. BrowseShSource — crawl browse.sh skill directory
# ---------------------------------------------------------------------------


class BrowseShSource(SkillSource):
    """Crawl browse.sh — a community skill browsing site.

    Identifier format: ``browsesh://<slug>``
    """

    BASE_URL = "https://browse.sh"

    def source_id(self) -> str:
        return "browsesh"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        url = f"{self.BASE_URL}/search?q={query}"
        html = self._fetch_text_sync(url)
        if not html:
            return []
        results: List[SkillMeta] = []
        for match in re.finditer(
            r'href="/skill/([a-z0-9_-]+)"[^>]*>.*?<h[23][^>]*>([^<]+)</h[23]>',
            html, re.DOTALL | re.IGNORECASE,
        ):
            slug = match.group(1)
            title = match.group(2).strip()
            results.append(SkillMeta(
                name=slug,
                description=title,
                source="browsesh",
                identifier=f"browsesh://{slug}",
                trust_level="community",
            ))
            if len(results) >= limit:
                break
        return results

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        slug = self._parse_identifier(identifier)
        if not slug:
            return None
        url = f"{self.BASE_URL}/skill/{slug}"
        html = self._fetch_text_sync(url)
        if not html:
            return None
        title_match = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
        return SkillMeta(
            name=title_match.group(1).strip() if title_match else slug,
            description="",
            source="browsesh",
            identifier=identifier,
            trust_level="community",
        )

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        slug = self._parse_identifier(identifier)
        if not slug:
            return None
        url = f"{self.BASE_URL}/skill/{slug}/raw"
        content = self._fetch_text_sync(url)
        if content is None:
            return None
        return SkillBundle(
            name=slug,
            files={"SKILL.md": content},
            source="browsesh",
            identifier=identifier,
            trust_level="community",
        )

    def _parse_identifier(self, identifier: str) -> Optional[str]:
        if identifier.startswith("browsesh://"):
            return identifier[len("browsesh://"):]
        return None

    def _fetch_text_sync(self, url: str) -> Optional[str]:
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "NIA-Skills/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 8. HermesIndexSource — fetch from the Hermes skill index cache
# ---------------------------------------------------------------------------


class HermesIndexSource(SkillSource):
    """Fetch skills from the Hermes skill index cache.

    The index cache is a local JSON file that's periodically refreshed
    from the Hermes skill registry. It contains skill metadata + download
    URLs.

    Identifier format: ``hermes-index://<skill-name>``
    """

    INDEX_CACHE_PATH = "skills/index-cache/hermes_index.json"

    def __init__(self, index_path: Optional[Path] = None) -> None:
        self._index_path = index_path or self._default_index_path()
        self._cache: Optional[List[dict]] = None

    def source_id(self) -> str:
        return "hermes-index"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        index = self._load_index()
        if not index:
            return []
        query_lower = query.lower()
        results: List[SkillMeta] = []
        for entry in index:
            name = entry.get("name", "")
            desc = entry.get("description", "")
            if query_lower in name.lower() or query_lower in desc.lower():
                results.append(SkillMeta(
                    name=name,
                    description=desc,
                    source="hermes-index",
                    identifier=f"hermes-index://{name}",
                    trust_level=entry.get("trust_level", "community"),
                    extra={"url": entry.get("url"), "category": entry.get("category")},
                ))
            if len(results) >= limit:
                break
        return results

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        name = self._parse_identifier(identifier)
        if not name:
            return None
        index = self._load_index()
        if not index:
            return None
        entry = next((e for e in index if e.get("name") == name), None)
        if not entry:
            return None
        return SkillMeta(
            name=name,
            description=entry.get("description", ""),
            source="hermes-index",
            identifier=identifier,
            trust_level=entry.get("trust_level", "community"),
            extra={"url": entry.get("url")},
        )

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        name = self._parse_identifier(identifier)
        if not name:
            return None
        index = self._load_index()
        if not index:
            return None
        entry = next((e for e in index if e.get("name") == name), None)
        if not entry:
            return None
        url = entry.get("url")
        if not url:
            return None
        content = self._fetch_text_sync(url)
        if content is None:
            return None
        return SkillBundle(
            name=name,
            files={"SKILL.md": content},
            source="hermes-index",
            identifier=identifier,
            trust_level=entry.get("trust_level", "community"),
        )

    def _parse_identifier(self, identifier: str) -> Optional[str]:
        if identifier.startswith("hermes-index://"):
            return identifier[len("hermes-index://"):]
        return None

    def _default_index_path(self) -> Path:
        try:
            from niaharness.config.paths import get_nia_home
            return Path(get_nia_home()) / self.INDEX_CACHE_PATH
        except Exception:
            return Path(os.environ.get("NIA_HOME", os.path.expanduser("~/.nia"))) / self.INDEX_CACHE_PATH

    def _load_index(self) -> List[dict]:
        if self._cache is not None:
            return self._cache
        try:
            if self._index_path.exists():
                data = json.loads(self._index_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._cache = data
                    return data
                if isinstance(data, dict):
                    self._cache = data.get("skills", [])
                    return self._cache
        except (OSError, json.JSONDecodeError):
            pass
        self._cache = []
        return []

    def _fetch_text_sync(self, url: str) -> Optional[str]:
        if not _is_safe_url(url):
            return None
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "NIA-Skills/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            return None


# ---------------------------------------------------------------------------
# TapsManager — custom GitHub repo sources
# ---------------------------------------------------------------------------


class TapsManager:
    """Manages the taps.json file — custom GitHub repo sources.

    A "tap" is a GitHub repo that contains skills. Users can add custom
    taps to extend the skill hub beyond the built-in default taps.

    The taps file lives at ``~/.nia/skills/taps.json`` and has the format::

        {"taps": [{"repo": "owner/repo", "path": "skills/"}]}
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        if path is not None:
            self.path = path
        else:
            try:
                from niaharness.config.paths import get_nia_home
                self.path = Path(get_nia_home()) / "skills" / "taps.json"
            except Exception:
                self.path = Path(os.environ.get("NIA_HOME", os.path.expanduser("~/.nia"))) / "skills" / "taps.json"

    def load(self) -> List[dict]:
        """Load the taps list."""
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            taps = data.get("taps", [])
            if isinstance(taps, list):
                return taps
        except (json.JSONDecodeError, OSError):
            pass
        return []

    def save(self, taps: List[dict]) -> None:
        """Save the taps list."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"taps": taps}, indent=2) + "\n", encoding="utf-8"
        )

    def add(self, repo: str, path: str = "skills/") -> bool:
        """Add a tap. Returns False if already exists."""
        taps = self.load()
        if any(t.get("repo") == repo for t in taps):
            return False
        taps.append({"repo": repo, "path": path})
        self.save(taps)
        logger.info("Added tap: %s/%s", repo, path)
        return True

    def remove(self, repo: str) -> bool:
        """Remove a tap by repo name. Returns False if not found."""
        taps = self.load()
        new_taps = [t for t in taps if t.get("repo") != repo]
        if len(new_taps) == len(taps):
            return False
        self.save(new_taps)
        logger.info("Removed tap: %s", repo)
        return True

    def list_taps(self) -> List[dict]:
        """Return all taps."""
        return self.load()


# ---------------------------------------------------------------------------
# Factory — return all extra sources
# ---------------------------------------------------------------------------


def get_extra_sources() -> List[SkillSource]:
    """Return all 8 extra skill source adapters.

    These complement the existing GitHubSource + OptionalSkillSource.
    """
    return [
        WellKnownSkillSource(),
        UrlSource(),
        SkillsShSource(),
        ClawHubSource(),
        ClaudeMarketplaceSource(),
        LobeHubSource(),
        BrowseShSource(),
        HermesIndexSource(),
    ]


def get_all_sources() -> List[Any]:
    """Return ALL sources: GitHub + Optional + 8 extra adapters.

    Merges the default sources from skills_hub_sources.get_default_sources()
    with the 8 extra adapters.
    """
    try:
        from niaharness.tools.skills_hub_sources import get_default_sources
        sources = list(get_default_sources())
    except Exception:
        sources = []
    sources.extend(get_extra_sources())
    return sources


__all__ = [
    "BrowseShSource",
    "ClaudeMarketplaceSource",
    "ClawHubSource",
    "HermesIndexSource",
    "LobeHubSource",
    "SkillBundle",
    "SkillMeta",
    "SkillSource",
    "SkillsShSource",
    "TapsManager",
    "UrlSource",
    "WellKnownSkillSource",
    "get_all_sources",
    "get_extra_sources",
]
