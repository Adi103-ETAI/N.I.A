"""Bundled skill definitions loaded from directory-based SKILL.md files.

Mirrors Hermes Agent's skills/ directory structure:
  skills/bundled/skills/<skill-name>/SKILL.md
  skills/bundled/skills/<skill-name>/references/*.md
  skills/bundled/skills/<skill-name>/templates/*.md
  skills/bundled/skills/<skill-name>/scripts/*.sh

The loader scans the ``skills/`` subdirectory for ``SKILL.md`` files.
Support files (references, templates, scripts) are NOT loaded into the
skill content — they're accessible via the ``skill_manage`` tool's
``write_file`` / ``remove_file`` actions and via the ``read_file`` tool
using the skill directory path.

For backward compatibility, the old flat ``content/*.md`` directory is
also scanned if it exists.
"""

from __future__ import annotations

import logging
from pathlib import Path

from niaharness.skills.types import SkillDefinition

logger = logging.getLogger(__name__)

# Primary: directory-based skills (skills/<name>/SKILL.md)
_SKILLS_DIR = Path(__file__).parent / "skills"

# Legacy: flat content/*.md files (backward compat)
_CONTENT_DIR = Path(__file__).parent / "content"


def get_bundled_skills() -> list[SkillDefinition]:
    """Load all bundled skills.

    Scans two locations:
    1. ``skills/`` subdirectory for ``<name>/SKILL.md`` files (primary,
       mirrors Hermes's directory structure).
    2. ``content/`` directory for flat ``*.md`` files (backward compat).
    """
    skills: list[SkillDefinition] = []
    seen_names: set[str] = set()

    # Primary: directory-based skills.
    if _SKILLS_DIR.exists():
        for skill_dir in sorted(_SKILLS_DIR.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                content = skill_md.read_text(encoding="utf-8")
                name, description = _parse_frontmatter(skill_dir.name, content)
                if name in seen_names:
                    continue
                seen_names.add(name)
                skills.append(
                    SkillDefinition(
                        name=name,
                        description=description,
                        content=content,
                        source="bundled",
                        path=str(skill_md),
                    )
                )
            except Exception as exc:
                logger.warning("Failed to load bundled skill %s: %s", skill_dir.name, exc)

    # Legacy: flat content/*.md files (only if not already loaded by name).
    if _CONTENT_DIR.exists():
        for path in sorted(_CONTENT_DIR.glob("*.md")):
            try:
                content = path.read_text(encoding="utf-8")
                name, description = _parse_frontmatter(path.stem, content)
                if name in seen_names:
                    continue
                seen_names.add(name)
                skills.append(
                    SkillDefinition(
                        name=name,
                        description=description,
                        content=content,
                        source="bundled",
                        path=str(path),
                    )
                )
            except Exception as exc:
                logger.warning("Failed to load legacy skill %s: %s", path.stem, exc)

    return skills


def get_bundled_skills_dir() -> Path:
    """Return the directory-based skills root (for skill_manage write_file)."""
    return _SKILLS_DIR


def _parse_frontmatter(default_name: str, content: str) -> tuple[str, str]:
    """Extract name and description from a skill markdown file.

    Supports both YAML frontmatter (---) and simple # heading format.
    """
    from niaharness.skills.loader import _parse_skill_markdown

    return _parse_skill_markdown(default_name, content)
