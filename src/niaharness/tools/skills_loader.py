"""Skill loading from bundled and user directories."""

from __future__ import annotations

from pathlib import Path

from niaharness.config.paths import get_config_dir
from niaharness.config.settings import load_settings
from niaharness.skills.bundled import get_bundled_skills
from niaharness.tools.skills_registry import SkillRegistry
from niaharness.tools.skills_types import SkillDefinition


def get_user_skills_dir() -> Path:
    """Return the user skills directory."""
    path = get_config_dir() / "skills"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_skill_registry(cwd: str | Path | None = None) -> SkillRegistry:
    """Load bundled and user-defined skills."""
    registry = SkillRegistry()
    for skill in get_bundled_skills():
        registry.register(skill)
    for skill in load_user_skills():
        registry.register(skill)
    if cwd is not None:
        from niaharness.plugins.loader import load_plugins

        settings = load_settings()
        for plugin in load_plugins(settings, cwd):
            if not plugin.enabled:
                continue
            for skill in plugin.skills:
                registry.register(skill)
    return registry


def load_user_skills() -> list[SkillDefinition]:
    """Load skills from the user config directory.

    Scans for directory-based skills (<name>/SKILL.md) first (mirrors
    Hermes's structure), then falls back to flat *.md files for backward
    compatibility.
    """
    skills: list[SkillDefinition] = []
    seen_names: set[str] = set()
    user_dir = get_user_skills_dir()

    # Primary: directory-based skills (<name>/SKILL.md).
    if user_dir.exists():
        for skill_dir in sorted(user_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                content = skill_md.read_text(encoding="utf-8")
                name, description = _parse_skill_markdown(skill_dir.name, content)
                if name in seen_names:
                    continue
                seen_names.add(name)
                skills.append(
                    SkillDefinition(
                        name=name,
                        description=description,
                        content=content,
                        source="user",
                        path=str(skill_md),
                    )
                )
            except Exception:
                continue

    # Legacy: flat *.md files (only if not already loaded by name).
    for path in sorted(user_dir.glob("*.md")):
        try:
            content = path.read_text(encoding="utf-8")
            name, description = _parse_skill_markdown(path.stem, content)
            if name in seen_names:
                continue
            seen_names.add(name)
            skills.append(
                SkillDefinition(
                    name=name,
                    description=description,
                    content=content,
                    source="user",
                    path=str(path),
                )
            )
        except Exception:
            continue

    return skills


def _parse_skill_markdown(default_name: str, content: str) -> tuple[str, str]:
    """Parse name and description from a skill markdown file.

    Uses proper YAML frontmatter parsing via skill_utils.parse_frontmatter.
    Falls back to heading/first-paragraph extraction for legacy skills.
    """
    from niaharness.tools.skill_utils import parse_frontmatter, extract_skill_description

    fm, body = parse_frontmatter(content)

    name = fm.get("name", default_name) if fm else default_name
    if not isinstance(name, str) or not name.strip():
        name = default_name
    name = name.strip()

    description = ""
    if fm and fm.get("description"):
        description = str(fm["description"]).strip()
    if not description:
        # Check for # heading as name source (legacy skills without frontmatter).
        for line in body.strip().splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                heading_name = stripped[2:].strip()
                if heading_name and name == default_name:
                    name = heading_name
                continue
            if stripped and not stripped.startswith("---") and not stripped.startswith("#"):
                description = stripped[:200]
                break
    if not description:
        description = extract_skill_description(content)
    if not description:
        description = f"Skill: {name}"
    return name, description
