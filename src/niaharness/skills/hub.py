"""Skill hub — browse, search, and install optional skills.

Adapted from the reference project's tools/skills_hub.py pattern.

Provides a catalog of optional skills that ship with NIA but aren't
activated by default. Users can browse, search, and install them via
the ``/skills`` slash command or the ``skill_hub`` tool.

Directory structure:
  bundled/<category>/<skill>/SKILL.md        ← active (always loaded)
  optional/<category>/<skill>/SKILL.md       ← optional (install on demand)

When a user installs an optional skill, its directory is copied from
``optional/`` to the user skills directory (``~/.niaharness/skills/``),
making it loadable by the skill registry.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from niaharness.skills.loader import get_user_skills_dir
from niaharness.skills.types import SkillDefinition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SkillMeta:
    """Metadata for a skill in the hub catalog."""

    name: str
    description: str
    category: str
    source: str  # "official" | "user"
    installed: bool = False
    path: Optional[str] = None
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Optional skills directory
# ---------------------------------------------------------------------------

_OPTIONAL_DIR = Path(__file__).parent / "bundled" / "optional"


def get_optional_skills_dir() -> Path:
    """Return the optional skills directory (shipped but not activated)."""
    return _OPTIONAL_DIR


# ---------------------------------------------------------------------------
# Hub operations
# ---------------------------------------------------------------------------


def list_optional_skills() -> list[SkillMeta]:
    """List all optional skills available for installation.

    Scans the optional/ directory for SKILL.md files.
    """
    if not _OPTIONAL_DIR.exists():
        return []

    # Load installed skill names to check status.
    from niaharness.skills import load_skill_registry

    registry = load_skill_registry()
    installed_names = {s.name for s in registry.list_skills()}

    results: list[SkillMeta] = []
    for skill_md in sorted(_OPTIONAL_DIR.rglob("SKILL.md")):
        try:
            content = skill_md.read_text(encoding="utf-8")
            from niaharness.skills.loader import _parse_skill_markdown

            name, description = _parse_skill_markdown(skill_md.parent.name, content)
            # Extract category from path: optional/<category>/<skill>/SKILL.md
            rel = skill_md.relative_to(_OPTIONAL_DIR)
            category = rel.parts[0] if len(rel.parts) > 1 else "uncategorized"

            results.append(
                SkillMeta(
                    name=name,
                    description=description,
                    category=category,
                    source="official",
                    installed=name in installed_names,
                    path=str(skill_md),
                )
            )
        except Exception as exc:
            logger.debug("Failed to scan optional skill %s: %s", skill_md, exc)

    return results


def search_optional_skills(query: str, limit: int = 20) -> list[SkillMeta]:
    """Search optional skills by name, description, or category.

    Args:
        query: Search string (case-insensitive substring match).
        limit: Max results to return.
    """
    query_lower = query.lower().strip()
    if not query_lower:
        return list_optional_skills()[:limit]

    results: list[SkillMeta] = []
    for meta in list_optional_skills():
        searchable = f"{meta.name} {meta.description} {meta.category}".lower()
        if query_lower in searchable:
            results.append(meta)
            if len(results) >= limit:
                break

    return results


def install_skill(skill_name: str) -> tuple[bool, str]:
    """Install an optional skill by name.

    Copies the skill directory from optional/ to the user skills directory.

    Args:
        skill_name: Skill name (e.g. "arxiv" or "docker-management").

    Returns:
        (success, message) tuple.
    """
    # Find the skill in optional/.
    skill_dir = _find_optional_skill_dir(skill_name)
    if skill_dir is None:
        return False, f"Skill '{skill_name}' not found in the optional catalog."

    # Check if already installed.
    from niaharness.skills import load_skill_registry

    registry = load_skill_registry()
    existing = registry.get(skill_name) or registry.get(skill_name.lower())
    if existing is not None and existing.source == "user":
        return True, f"Skill '{skill_name}' is already installed."

    # Copy to user skills directory.
    user_dir = get_user_skills_dir()
    dest_dir = user_dir / skill_name
    if dest_dir.exists():
        return False, f"Skill directory already exists: {dest_dir}"

    try:
        shutil.copytree(skill_dir, dest_dir)
        return True, f"Installed skill '{skill_name}' to {dest_dir}"
    except Exception as exc:
        return False, f"Failed to install skill '{skill_name}': {exc}"


def uninstall_skill(skill_name: str) -> tuple[bool, str]:
    """Uninstall a user-installed skill by name.

    Removes the skill directory from the user skills directory.
    Only removes user-installed skills — bundled skills cannot be uninstalled.

    Args:
        skill_name: Skill name.

    Returns:
        (success, message) tuple.
    """
    user_dir = get_user_skills_dir()
    skill_dir = user_dir / skill_name
    if not skill_dir.exists():
        return False, f"Skill '{skill_name}' is not installed in the user directory."

    try:
        shutil.rmtree(skill_dir)
        return True, f"Uninstalled skill '{skill_name}'"
    except Exception as exc:
        return False, f"Failed to uninstall skill '{skill_name}': {exc}"


def _find_optional_skill_dir(skill_name: str) -> Optional[Path]:
    """Find an optional skill directory by name.

    Searches optional/<category>/<skill_name>/ recursively.
    """
    if not _OPTIONAL_DIR.exists():
        return None

    # Try exact match on directory name.
    for skill_md in _OPTIONAL_DIR.rglob("SKILL.md"):
        if skill_md.parent.name == skill_name:
            return skill_md.parent

    # Try case-insensitive match.
    for skill_md in _OPTIONAL_DIR.rglob("SKILL.md"):
        if skill_md.parent.name.lower() == skill_name.lower():
            return skill_md.parent

    return None
