"""Skill utility functions — path exclusion, support-dir detection, frontmatter parsing.

Adapted from the reference project's agent/skill_utils.py.
Provides the shared primitives every skill-scanning site needs.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePath
from typing import Any, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants (adapted from reference EXCLUDED_SKILL_DIRS + SKILL_SUPPORT_DIRS)
# ---------------------------------------------------------------------------

EXCLUDED_SKILL_DIRS = frozenset(
    (
        ".git",
        ".github",
        ".hub",
        ".archive",
        ".venv",
        "venv",
        "node_modules",
        "site-packages",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".eggs",
        "build",
        "dist",
        "optional",  # NIA-specific: optional skills are managed by the hub
    )
)

SKILL_SUPPORT_DIRS = frozenset(("references", "templates", "assets", "scripts"))


# ---------------------------------------------------------------------------
# Path exclusion (adapted from reference is_excluded_skill_path + is_skill_support_path)
# ---------------------------------------------------------------------------


def is_excluded_skill_path(path) -> bool:
    """True if *path* should be skipped by active skill scanners.

    Prunes dependency, virtualenv, VCS, cache, and progressive-disclosure
    support-package paths.
    """
    try:
        parts = path.parts  # Path
    except AttributeError:
        parts = PurePath(str(path)).parts
    return any(part in EXCLUDED_SKILL_DIRS for part in parts) or is_skill_support_path(path)


def is_skill_support_path(path) -> bool:
    """True if *path* is under a support dir of an actual skill root.

    ``references/``, ``templates/``, ``assets/``, and ``scripts/`` are
    progressive-disclosure support areas when they sit directly inside a skill
    directory containing ``SKILL.md``. They are not active discovery roots for
    standalone skills.
    """
    try:
        parts = path.parts
    except AttributeError:
        parts = PurePath(str(path)).parts
    for idx, part in enumerate(parts):
        if part not in SKILL_SUPPORT_DIRS or idx == 0:
            continue
        skill_root = Path(*parts[:idx])
        if (skill_root / "SKILL.md").exists():
            return True
    return False


def is_external_skill_path(path) -> bool:
    """Return True when ``path`` lives under a configured external skills dir.

    NIA doesn't have external_dirs config yet — always returns False.
    """
    return False


# ---------------------------------------------------------------------------
# YAML frontmatter parsing (adapted from reference parse_frontmatter)
# ---------------------------------------------------------------------------


def parse_frontmatter(content: str) -> Tuple[dict, str]:
    """Parse YAML frontmatter from SKILL.md content.

    Returns (frontmatter_dict, remaining_body).
    Falls back to empty dict if YAML is not available or content has no frontmatter.
    """
    if not content.startswith("---"):
        return {}, content
    # Find the closing ---
    match = re.search(r"\n---\s*\n", content[3:])
    if not match:
        return {}, content
    yaml_text = content[3 : match.start() + 3]
    body = content[match.end() + 3 :]
    if yaml is None:
        # Fallback: simple key: value parsing
        fm: dict[str, Any] = {}
        for line in yaml_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                fm[key.strip()] = val.strip().strip("'\"")
        return fm, body
    try:
        parsed = yaml.safe_load(yaml_text)
        return (parsed if isinstance(parsed, dict) else {}), body
    except Exception:
        return {}, body


def extract_skill_description(content: str) -> str:
    """Extract description from skill content (frontmatter or first paragraph)."""
    fm, body = parse_frontmatter(content)
    if fm.get("description"):
        return str(fm["description"])[:200]
    # Fallback: first non-empty, non-heading line
    for line in body.strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("---"):
            return line[:200]
    return ""


def extract_skill_tags(content: str) -> list[str]:
    """Extract tags from skill frontmatter (metadata.hermes.tags or metadata.niaharness.tags)."""
    fm, _ = parse_frontmatter(content)
    meta_block = fm.get("metadata", {})
    if not isinstance(meta_block, dict):
        return []
    for key in ("hermes", "niaharness", "nia"):
        section = meta_block.get(key, {})
        if isinstance(section, dict):
            tags = section.get("tags", [])
            if isinstance(tags, list):
                return tags
    return []


def skill_matches_platform(frontmatter: dict, platform: Optional[str] = None) -> bool:
    """Check if a skill matches the current platform.

    If no platform is specified, always returns True.
    If the skill has no 'platforms' field, always returns True.
    """
    if platform is None:
        return True
    platforms = frontmatter.get("platforms")
    if not platforms:
        return True
    if isinstance(platforms, str):
        platforms = [platforms]
    return platform.lower() in [p.lower() for p in platforms]


# ---------------------------------------------------------------------------
# Skill index iteration (adapted from reference iter_skill_index_files)
# ---------------------------------------------------------------------------


def iter_skill_index_files(root: Path):
    """Yield SKILL.md paths under *root*, skipping excluded and support dirs.

    This is the canonical scanner used by all skill-loading code paths.
    """
    if not root.is_dir():
        return
    for skill_md in root.rglob("SKILL.md"):
        if is_excluded_skill_path(skill_md):
            continue
        yield skill_md


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_category_from_path(skill_md_path: Path, root: Path) -> str:
    """Extract category from path: <root>/<category>/<skill>/SKILL.md → 'category'."""
    try:
        rel = skill_md_path.relative_to(root)
        parts = rel.parts
        if len(parts) >= 3:
            return parts[0]  # <category>/<skill>/SKILL.md
        return ""
    except ValueError:
        return ""


def validate_skill_name(name: str) -> Optional[str]:
    """Validate a skill name. Returns error message or None if valid."""
    if not name:
        return "Skill name is required."
    if len(name) > 64:
        return f"Skill name exceeds 64 characters."
    if not re.match(r"^[a-z0-9][a-z0-9._-]*$", name):
        return (
            f"Invalid skill name '{name}'. Use lowercase letters, numbers, "
            f"hyphens, dots, and underscores."
        )
    return None
