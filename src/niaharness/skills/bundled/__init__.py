"""Bundled skill definitions loaded from categorized SKILL.md files.

Directory structure (mirrors the reference project's layout):
  bundled/<category>/<skill-name>/SKILL.md
  bundled/<category>/<skill-name>/references/*.md
  bundled/<category>/<skill-name>/templates/*.md
  bundled/<category>/<skill-name>/scripts/*.sh

The loader recursively scans for SKILL.md files under the bundled directory.
Categories (github, software-development, etc.) are just organizational —
skills are identified by name, not by category path.
"""

from __future__ import annotations

import logging
from pathlib import Path

from niaharness.skills.types import SkillDefinition

logger = logging.getLogger(__name__)

_BUNDLED_DIR = Path(__file__).parent


def get_bundled_skills() -> list[SkillDefinition]:
    """Load all bundled skills by recursively scanning for SKILL.md files.

    Scans: bundled/<category>/<skill-name>/SKILL.md
    Returns skills sorted by name, deduplicated.
    """
    skills: list[SkillDefinition] = []
    seen_names: set[str] = set()

    if not _BUNDLED_DIR.exists():
        return skills

    # Recursively find all SKILL.md files, excluding the optional/ directory
    # (optional skills are managed by the skill hub, not loaded by default).
    for skill_md in sorted(_BUNDLED_DIR.rglob("SKILL.md")):
        # Skip skills in the optional/ directory.
        if "optional" in skill_md.parts:
            continue
        try:
            content = skill_md.read_text(encoding="utf-8")
            name, description = _parse_frontmatter(skill_md.parent.name, content)
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
            logger.warning("Failed to load bundled skill %s: %s", skill_md, exc)

    return skills


def get_bundled_skills_dir() -> Path:
    """Return the bundled skills root directory."""
    return _BUNDLED_DIR


def _parse_frontmatter(default_name: str, content: str) -> tuple[str, str]:
    """Extract name and description from a skill markdown file."""
    from niaharness.skills.loader import _parse_skill_markdown

    return _parse_skill_markdown(default_name, content)
