"""Skill loading from bundled and user directories."""

from __future__ import annotations

from pathlib import Path

from niaharness.config.paths import get_config_dir
from niaharness.config.settings import load_settings
from niaharness.skills.bundled import get_bundled_skills
from niaharness.skills.registry import SkillRegistry
from niaharness.skills.types import SkillDefinition


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
    """Parse name and description from a skill markdown file with YAML frontmatter support."""
    name = default_name
    description = ""

    lines = content.splitlines()

    # Try YAML frontmatter first (--- ... ---)
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                # Parse frontmatter fields
                for fm_line in lines[1:i]:
                    fm_stripped = fm_line.strip()
                    if fm_stripped.startswith("name:"):
                        val = fm_stripped[5:].strip().strip("'\"")
                        if val:
                            name = val
                    elif fm_stripped.startswith("description:"):
                        val = fm_stripped[12:].strip().strip("'\"")
                        if val:
                            description = val
                break

    # Fallback: extract from headings and first paragraph
    if not description:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# "):
                if not name or name == default_name:
                    name = stripped[2:].strip() or default_name
                continue
            if stripped and not stripped.startswith("---") and not stripped.startswith("#"):
                description = stripped[:200]
                break

    if not description:
        description = f"Skill: {name}"
    return name, description
