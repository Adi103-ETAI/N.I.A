"""Skill hub — browse, search, install, and uninstall optional skills.

Adapted from the reference project's tools/skills_hub.py.

Provides:
  - SkillSource ABC: Interface for all skill registry adapters
  - OptionalSkillSource: Official optional skills shipped with the repo
  - HubLockFile: Track provenance of installed hub skills
  - quarantine_bundle: Write skills to quarantine for scanning before install
  - install_from_quarantine: Move scanned skills to the user skills directory
  - Path validation: _validate_skill_name, _normalize_bundle_path, etc.

Optional skills are shipped in bundled/optional/<category>/<skill>/SKILL.md
but not loaded by default. Users browse, search, and install them via the
skill_hub tool or /skills slash command.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Union

from niaharness.tools.skills_loader import get_user_skills_dir, _parse_skill_markdown

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_OPTIONAL_DIR = Path(__file__).parent.parent / "skills" / "bundled" / "optional"


def get_optional_skills_dir() -> Path:
    """Return the optional skills directory (shipped but not activated)."""
    return _OPTIONAL_DIR


def _hub_dir() -> Path:
    """Return the hub state directory (~/.niaharness/skills/.hub/)."""
    d = get_user_skills_dir() / ".hub"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lock_file() -> Path:
    return _hub_dir() / "lock.json"


def _quarantine_dir() -> Path:
    d = _hub_dir() / "quarantine"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Data classes (adapted from reference SkillMeta + SkillBundle)
# ---------------------------------------------------------------------------


@dataclass
class SkillMeta:
    """Minimal metadata returned by search results."""

    name: str
    description: str
    source: str  # "official" | "user"
    identifier: str  # source-specific ID
    trust_level: str  # "builtin" | "trusted" | "community"
    category: str = ""
    installed: bool = False
    repo: Optional[str] = None
    path: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class SkillBundle:
    """A downloaded skill ready for quarantine/scanning/installation."""

    name: str
    files: dict[str, Union[str, bytes]]  # relative_path -> file content
    source: str
    identifier: str
    trust_level: str
    category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Path validation (adapted from reference _normalize_bundle_path etc.)
# ---------------------------------------------------------------------------


def _validate_skill_name(name: str) -> str:
    """Validate a skill name. Returns the validated name or raises ValueError."""
    if not name or not isinstance(name, str):
        raise ValueError(f"Unsafe skill name: {name!r}")
    raw = name.strip()
    if not raw:
        raise ValueError("Empty skill name")
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$", raw):
        raise ValueError(f"Unsafe skill name: {name!r}")
    return raw


def _validate_bundle_rel_path(rel_path: str) -> str:
    """Validate a relative path inside a skill bundle."""
    if not isinstance(rel_path, str):
        raise ValueError(f"Unsafe bundle path: {rel_path!r}")
    raw = rel_path.strip().replace("\\", "/")
    if not raw:
        raise ValueError("Empty bundle path")
    path = PurePosixPath(raw)
    parts = [p for p in path.parts if p not in {"", "."}]
    if raw.startswith("/") or path.is_absolute():
        raise ValueError(f"Unsafe bundle path: {rel_path!r}")
    if not parts or any(p == ".." for p in parts):
        raise ValueError(f"Unsafe bundle path: {rel_path!r}")
    return raw


def _is_path_redirect(path: Path) -> bool:
    """Check if a path is a symlink (which could redirect outside the skill dir)."""
    try:
        return path.is_symlink()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# SkillSource ABC (adapted from reference SkillSource)
# ---------------------------------------------------------------------------


class SkillSource(ABC):
    """Abstract base for all skill registry adapters."""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[SkillMeta]:
        """Search for skills matching a query string."""

    @abstractmethod
    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        """Download a skill bundle by identifier."""

    @abstractmethod
    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        """Fetch metadata for a skill without downloading all files."""

    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this source (e.g. 'official')."""

    def trust_level_for(self, identifier: str) -> str:
        """Determine trust level for a skill from this source."""
        return "community"


# ---------------------------------------------------------------------------
# OptionalSkillSource (adapted from reference OptionalSkillSource)
# ---------------------------------------------------------------------------


class OptionalSkillSource(SkillSource):
    """Fetch skills from the optional/ directory shipped with the repo.

    These skills are official but not activated by default. They are
    discoverable via the Skills Hub (search / install / inspect) and
    labelled "official" with "builtin" trust.
    """

    def __init__(self) -> None:
        self._optional_dir = _OPTIONAL_DIR

    def source_id(self) -> str:
        return "official"

    def trust_level_for(self, identifier: str) -> str:
        return "builtin"

    # -- search -----------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> list[SkillMeta]:
        results: list[SkillMeta] = []
        query_lower = query.lower()

        for meta in self._scan_all():
            searchable = f"{meta.name} {meta.description} {' '.join(meta.tags)}".lower()
            if query_lower in searchable:
                results.append(meta)
            if len(results) >= limit:
                break

        return results

    # -- fetch ------------------------------------------------------------

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        """Fetch a skill bundle by identifier.

        Identifier format: "official/category/skill" or "official/skill"
        or just "skill".
        """
        rel = identifier.split("/", 1)[-1] if identifier.startswith("official/") else identifier
        skill_dir = self._optional_dir / rel

        # Guard against path traversal.
        try:
            resolved = skill_dir.resolve()
            optional_root = self._optional_dir.resolve()
            if not resolved.is_relative_to(optional_root):
                return None
        except (OSError, ValueError):
            return None

        if not resolved.is_dir():
            skill_name = rel.rsplit("/", 1)[-1]
            skill_dir = self._find_skill_dir(skill_name)
            if not skill_dir:
                return None
        else:
            skill_dir = resolved

        files: dict[str, Union[str, bytes]] = {}
        for f in skill_dir.rglob("*"):
            if (
                f.is_file()
                and not f.name.startswith(".")
                and "__pycache__" not in f.parts
                and f.suffix != ".pyc"
            ):
                rel_path = str(f.relative_to(skill_dir))
                try:
                    safe_rel = _validate_bundle_rel_path(rel_path)
                    files[safe_rel] = f.read_bytes()
                except (ValueError, OSError):
                    continue

        if not files:
            return None

        name = skill_dir.name
        rel_to_optional = skill_dir.relative_to(self._optional_dir).as_posix()
        category = rel_to_optional.split("/")[0] if "/" in rel_to_optional else ""

        return SkillBundle(
            name=name,
            files=files,
            source="official",
            identifier=f"official/{rel_to_optional}",
            trust_level="builtin",
            category=category,
        )

    # -- inspect ----------------------------------------------------------

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        rel = identifier.split("/", 1)[-1] if identifier.startswith("official/") else identifier
        skill_name = rel.rsplit("/", 1)[-1]

        for meta in self._scan_all():
            if meta.name == skill_name:
                return meta
        return None

    # -- internal helpers -------------------------------------------------

    def _find_skill_dir(self, name: str) -> Optional[Path]:
        """Find a skill directory by name anywhere in optional/."""
        if not self._optional_dir.is_dir():
            return None
        for skill_md in self._optional_dir.rglob("SKILL.md"):
            if skill_md.parent.name == name:
                return skill_md.parent
        return None

    def _scan_all(self) -> list[SkillMeta]:
        """Enumerate all optional skills with metadata."""
        if not self._optional_dir.is_dir():
            return []

        # Load installed skill names to check status.
        from niaharness.tools.skills_loader import load_skill_registry

        registry = load_skill_registry()
        installed_names = {s.name for s in registry.list_skills()}

        results: list[SkillMeta] = []
        for skill_md in sorted(self._optional_dir.rglob("SKILL.md")):
            parent = skill_md.parent
            try:
                content = skill_md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            name, description = _parse_skill_markdown(parent.name, content)
            rel_path = parent.relative_to(self._optional_dir).as_posix()
            category = rel_path.split("/")[0] if "/" in rel_path else ""

            results.append(
                SkillMeta(
                    name=name,
                    description=description[:200],
                    source="official",
                    identifier=f"official/{rel_path}",
                    trust_level="builtin",
                    category=category,
                    installed=name in installed_names,
                    path=str(skill_md),
                )
            )

        return results


# ---------------------------------------------------------------------------
# HubLockFile (adapted from reference HubLockFile)
# ---------------------------------------------------------------------------


class HubLockFile:
    """Manages skills/.hub/lock.json — tracks provenance of installed hub skills."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path if path is not None else _lock_file()

    def load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "installed": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"version": 1, "installed": {}}

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def record_install(
        self,
        name: str,
        source: str,
        identifier: str,
        trust_level: str,
        skill_hash: str,
        install_path: str,
        files: list[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        safe_name = _validate_skill_name(name)
        data = self.load()
        data["installed"][safe_name] = {
            "source": source,
            "identifier": identifier,
            "trust_level": trust_level,
            "content_hash": skill_hash,
            "install_path": install_path,
            "files": files,
            "metadata": metadata or {},
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save(data)

    def record_uninstall(self, name: str) -> None:
        data = self.load()
        data["installed"].pop(name, None)
        self.save(data)

    def get_installed(self, name: str) -> Optional[dict]:
        data = self.load()
        return data["installed"].get(name)

    def list_installed(self) -> list[dict]:
        data = self.load()
        return [{"name": name, **entry} for name, entry in data["installed"].items()]


# ---------------------------------------------------------------------------
# Quarantine + install (adapted from reference quarantine_bundle + install_from_quarantine)
# ---------------------------------------------------------------------------


def _content_hash(skill_dir: Path) -> str:
    """Compute a SHA-256 hash of all files in a skill directory."""
    h = hashlib.sha256()
    for f in sorted(skill_dir.rglob("*")):
        if f.is_file() and not f.name.startswith(".") and f.suffix != ".pyc":
            h.update(f.read_bytes())
    return h.hexdigest()


def quarantine_bundle(bundle: SkillBundle) -> Path:
    """Write a skill bundle to the quarantine directory for scanning.

    Adapted from reference quarantine_bundle.
    """
    skill_name = _validate_skill_name(bundle.name)
    validated_files: list[tuple[str, Union[str, bytes]]] = []
    for rel_path, file_content in bundle.files.items():
        safe_rel = _validate_bundle_rel_path(rel_path)
        validated_files.append((safe_rel, file_content))

    dest = _quarantine_dir() / skill_name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for rel_path, file_content in validated_files:
        file_dest = dest.joinpath(*rel_path.split("/"))
        file_dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(file_content, bytes):
            file_dest.write_bytes(file_content)
        else:
            file_dest.write_text(file_content, encoding="utf-8")

    return dest


def install_from_quarantine(
    quarantine_path: Path,
    skill_name: str,
    bundle: SkillBundle,
) -> Path:
    """Move a scanned skill from quarantine into the user skills directory.

    Includes symlink rejection and lock file recording.
    Adapted from reference install_from_quarantine.
    """
    safe_skill_name = _validate_skill_name(skill_name)

    # Resolve quarantine path safely.
    quarantine_resolved = quarantine_path.resolve()
    quarantine_root = _quarantine_dir().resolve()
    if not quarantine_resolved.is_relative_to(quarantine_root):
        raise ValueError(f"Unsafe quarantine path: {quarantine_path}")

    install_dir = get_user_skills_dir() / safe_skill_name

    if install_dir.exists():
        shutil.rmtree(install_dir)

    # Warn if SKILL.md is very large.
    skill_md = quarantine_path / "SKILL.md"
    if skill_md.exists():
        try:
            skill_size = skill_md.stat().st_size
            if skill_size > 100_000:
                logger.warning(
                    "Skill '%s' has a large SKILL.md (%s chars). "
                    "Consider splitting into smaller files.",
                    safe_skill_name,
                    f"{skill_size:,}",
                )
        except OSError:
            pass

    # Reject symlinks inside the quarantined skill.
    for entry in quarantine_path.rglob("*"):
        if _is_path_redirect(entry):
            try:
                rel = entry.relative_to(quarantine_resolved)
            except ValueError:
                rel = entry
            raise ValueError(
                f"Installed skill contains symlinks, which is not allowed: {rel}"
            )

    install_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(quarantine_path), str(install_dir))

    # Record in lock file.
    lock = HubLockFile()
    lock.record_install(
        name=safe_skill_name,
        source=bundle.source,
        identifier=bundle.identifier,
        trust_level=bundle.trust_level,
        skill_hash=_content_hash(install_dir),
        install_path=str(install_dir.relative_to(get_user_skills_dir())),
        files=list(bundle.files.keys()),
        metadata={"category": bundle.category} if bundle.category else {},
    )

    return install_dir


# ---------------------------------------------------------------------------
# Public API (high-level convenience functions)
# ---------------------------------------------------------------------------


def list_optional_skills() -> list[SkillMeta]:
    """List all optional skills available for installation."""
    source = OptionalSkillSource()
    return source._scan_all()


def search_optional_skills(query: str, limit: int = 20) -> list[SkillMeta]:
    """Search optional skills by keyword."""
    source = OptionalSkillSource()
    return source.search(query, limit=limit)


def install_skill(skill_name: str) -> tuple[bool, str]:
    """Install an optional skill by name.

    Flow: fetch → quarantine → scan (basic) → install → lock.
    """
    source = OptionalSkillSource()

    # Check if already installed.
    from niaharness.tools.skills_loader import load_skill_registry

    registry = load_skill_registry()
    existing = registry.get(skill_name) or registry.get(skill_name.lower())
    if existing is not None and existing.source == "user":
        return True, f"Skill '{skill_name}' is already installed."

    # Fetch the skill bundle.
    bundle = source.fetch(skill_name)
    if bundle is None:
        return False, f"Skill '{skill_name}' not found in the optional catalog."

    # Quarantine.
    try:
        quarantine_path = quarantine_bundle(bundle)
    except ValueError as exc:
        return False, f"Quarantine failed: {exc}"

    # Install (with symlink rejection + lock file recording).
    try:
        install_dir = install_from_quarantine(quarantine_path, skill_name, bundle)
        return True, f"Installed skill '{skill_name}' to {install_dir}"
    except ValueError as exc:
        return False, f"Install failed: {exc}"
    except Exception as exc:
        return False, f"Install failed: {exc}"


def uninstall_skill(skill_name: str) -> tuple[bool, str]:
    """Uninstall a user-installed skill by name."""
    safe_name = _validate_skill_name(skill_name)
    user_dir = get_user_skills_dir()
    skill_dir = user_dir / safe_name
    if not skill_dir.exists():
        return False, f"Skill '{safe_name}' is not installed in the user directory."

    try:
        shutil.rmtree(skill_dir)
        # Remove from lock file.
        lock = HubLockFile()
        lock.record_uninstall(safe_name)
        return True, f"Uninstalled skill '{safe_name}'"
    except Exception as exc:
        return False, f"Failed to uninstall skill '{safe_name}': {exc}"
