"""Bundled skill definitions loaded from .md files."""

from __future__ import annotations

from pathlib import Path

from niaharness.skills.types import SkillDefinition

_CONTENT_DIR = Path(__file__).parent / "content"


def get_bundled_skills() -> list[SkillDefinition]:
    """Load all bundled skills from the content/ directory."""
    skills: list[SkillDefinition] = []
    if not _CONTENT_DIR.exists():
        return skills
    for path in sorted(_CONTENT_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        name, description = _parse_frontmatter(path.stem, content)
        skills.append(
            SkillDefinition(
                name=name,
                description=description,
                content=content,
                source="bundled",
                path=str(path),
            )
        )
    return skills


def _parse_frontmatter(default_name: str, content: str) -> tuple[str, str]:
    """Extract name and description from a skill markdown file.

    Supports both YAML frontmatter (---) and simple # heading format.
    Uses the loader's _parse_skill_markdown for frontmatter support.
    """
    from niaharness.skills.loader import _parse_skill_markdown

    return _parse_skill_markdown(default_name, content)
