"""Skill management tool — create, update, edit, delete user skills.

The audit (P0 Task 2) flagged that NIA's ``skill`` tool is read-only.  This
tool fills that gap, letting the agent manage its own procedural memory.

Skills are markdown files (with optional YAML frontmatter) stored in the
user skills directory (``~/.niaharness/skills/`` by default, or
``$NIAHARNESS_CONFIG_DIR/skills/``).  Bundled skills (shipped with the
package) are read-only and cannot be modified through this tool.

Operations:
- ``create``  — create a new user skill markdown file
- ``update``  — replace a skill's content entirely (full rewrite)
- ``edit``    — find-and-replace within a skill (scoped to skills)
- ``delete``  — delete a user skill
- ``list``    — list all skills (bundled + user) with source info
- ``info``    — show skill metadata (name, description, source, path)

Safety:
- Only writes to the USER skills dir — never bundled.
- Skill names validated: ``^[a-z0-9][a-z0-9._-]*$``, max 64 chars.
- Frontmatter required for create/update: ``name`` + ``description`` fields.
- Max content size: 100K chars (~36K tokens at 2.75 chars/token).
- Path traversal blocked: skill name can't contain ``/`` or ``..``.

Reference: Hermes Agent's ``tools/skill_manager_tool.py`` (6 ops:
create/patch/edit/delete/write_file/remove_file). NIA's skills are single
markdown files (not directory-based), so write_file/remove_file don't apply.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from niaharness.tools.skills_loader import load_skill_registry
from niaharness.tools import skills_loader
from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


# ---------------------------------------------------------------------------
# Constants (ported from Hermes)
# ---------------------------------------------------------------------------

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_SKILL_CONTENT_CHARS = 100_000  # ~36k tokens at 2.75 chars/token

VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

# Frontmatter template for new skills.
_SKILL_TEMPLATE = """\
---
name: {name}
description: {description}
---

# {title}

{body}
"""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class SkillManageToolInput(BaseModel):
    """Arguments for the skill_manage tool.

    Supports 8 operations (adapted from reference skill_manager_tool):
    create, update, edit, delete, list, info, write_file, remove_file.
    """

    action: Literal["create", "update", "edit", "delete", "list", "info", "write_file", "remove_file"] = Field(
        description=(
            "The skill management operation. write_file/remove_file manage "
            "support files (references/, templates/, scripts/, assets/)."
        )
    )
    name: str | None = Field(
        default=None,
        description="Skill name (required for create/update/edit/delete/info/write_file/remove_file). "
        "Must match ^[a-z0-9][a-z0-9._-]*$ — lowercase letters, digits, hyphens, dots, underscores.",
    )
    description: str | None = Field(
        default=None,
        description="One-line skill description (required for create, optional for update). "
        "Max 1024 chars. Used in skill listings and the system prompt.",
    )
    content: str | None = Field(
        default=None,
        description=(
            "Full skill content (required for create/update). "
            "For create: if this includes YAML frontmatter (---), it's used as-is; "
            "otherwise the tool wraps name+description+content in frontmatter automatically. "
            "For update: replaces the entire file content."
        ),
    )
    old_string: str | None = Field(
        default=None,
        description="For edit: the exact text to find in the skill content",
    )
    new_string: str | None = Field(
        default=None,
        description="For edit: the replacement text (must differ from old_string)",
    )
    replace_all: bool = Field(
        default=False,
        description="For edit: replace all occurrences of old_string (default false = first only)",
    )
    file_path: str | None = Field(
        default=None,
        description=(
            "For write_file/remove_file: path relative to the skill directory "
            "(e.g. 'references/notes.md', 'templates/pr-body.md', 'scripts/run.sh'). "
            "Must be under references/, templates/, scripts/, or assets/."
        ),
    )
    file_content: str | None = Field(
        default=None,
        description="For write_file: the content to write to the support file.",
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_name(name: str | None) -> tuple[str | None, str | None]:
    """Validate a skill name. Returns (normalized_name, error_message)."""
    if not name:
        return None, "Skill name is required."
    name = name.strip().lower()
    if len(name) > MAX_NAME_LENGTH:
        return None, f"Skill name exceeds {MAX_NAME_LENGTH} characters."
    if not VALID_NAME_RE.match(name):
        return None, (
            f"Invalid skill name '{name}'. Use lowercase letters, digits, "
            f"hyphens, dots, and underscores. Must start with a letter or digit."
        )
    # Block path traversal.
    if "/" in name or "\\" in name or ".." in name:
        return None, f"Skill name contains forbidden path characters: {name!r}"
    return name, None


def _validate_content_size(content: str) -> str | None:
    """Check that content doesn't exceed the character limit."""
    if len(content) > MAX_SKILL_CONTENT_CHARS:
        return (
            f"Skill content is {len(content):,} characters "
            f"(limit: {MAX_SKILL_CONTENT_CHARS:,})."
        )
    return None


def _validate_frontmatter(content: str) -> str | None:
    """Validate that content has proper frontmatter with required fields.

    Uses proper YAML parsing via skill_utils.parse_frontmatter (adapted from
    reference _validate_frontmatter which uses yaml.safe_load).
    """
    if not content.strip():
        return "Content cannot be empty."

    if not content.startswith("---"):
        return "Skill content must start with YAML frontmatter (---)."

    # Find closing ---
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return "Skill frontmatter is not closed. Ensure you have a closing '---' line."

    body = content[end_match.end() + 3 :].strip()
    if not body:
        return "Skill must have content after the frontmatter (instructions, procedures, etc.)."

    # Use proper YAML parsing (adapted from reference).
    from niaharness.tools.skill_utils import parse_frontmatter

    parsed, _ = parse_frontmatter(content)

    if "name" not in parsed:
        return "Frontmatter must include 'name' field."
    if "description" not in parsed:
        return "Frontmatter must include 'description' field."
    if len(str(parsed.get("description", ""))) > MAX_DESCRIPTION_LENGTH:
        return f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters."

    return None


def _skill_file_path(name: str) -> Path:
    """Return the path to a user skill's SKILL.md file.

    Uses directory-based structure: <user_skills_dir>/<name>/SKILL.md
    (mirrors Hermes's skills/<name>/SKILL.md layout).
    """
    return skills_loader.get_user_skills_dir() / name / "SKILL.md"


def _skill_dir_path(name: str) -> Path:
    """Return the directory for a user skill (references/, templates/, etc.)."""
    return skills_loader.get_user_skills_dir() / name


def _is_user_skill(name: str) -> bool:
    """Return True if a user skill file exists for ``name``."""
    return _skill_file_path(name).exists()


def _wrap_in_frontmatter(name: str, description: str, body: str) -> str:
    """Wrap name + description + body in YAML frontmatter."""
    title = name.replace("-", " ").replace("_", " ").title()
    return _SKILL_TEMPLATE.format(
        name=name,
        description=description,
        title=title,
        body=body,
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class SkillManageTool(BaseTool):
    """Create, update, edit, and delete user skills."""

    name = "skill_manage"
    description = (
        "Manage user skills — create, update, edit, delete, list, or inspect. "
        "Skills are the agent's procedural memory: reusable approaches for "
        "recurring task types. Only user skills (in ~/.niaharness/skills/) "
        "are writable; bundled skills are read-only."
    )
    input_model = SkillManageToolInput

    def is_read_only(self, arguments: SkillManageToolInput) -> bool:
        return arguments.action in ("list", "info")

    async def execute(self, arguments: SkillManageToolInput, context: ToolExecutionContext) -> ToolResult:
        action = arguments.action

        if action == "list":
            return self._list_skills(context)
        if action == "info":
            return self._info(arguments, context)

        # All write actions require a valid name.
        name, err = _validate_name(arguments.name)
        if err:
            return ToolResult(output=err, is_error=True)
        assert name is not None

        if action == "create":
            return self._create(name, arguments)
        if action == "update":
            return self._update(name, arguments)
        if action == "edit":
            return self._edit(name, arguments)
        if action == "delete":
            return self._delete(name)
        if action == "write_file":
            return self._write_file(name, arguments)
        if action == "remove_file":
            return self._remove_file(name, arguments)

        return ToolResult(output=f"Unknown action: {action}", is_error=True)

    # ---- actions -------------------------------------------------------

    def _list_skills(self, context: ToolExecutionContext) -> ToolResult:
        """List all skills (bundled + user) with source info."""
        registry = load_skill_registry(context.cwd)
        skills = registry.list_skills()
        if not skills:
            return ToolResult(output="No skills found.")

        lines = [f"Skills ({len(skills)} total):", ""]
        bundled = [s for s in skills if s.source == "bundled"]
        user = [s for s in skills if s.source == "user"]
        plugin = [s for s in skills if s.source not in ("bundled", "user")]

        if bundled:
            lines.append(f"Bundled ({len(bundled)}):")
            for s in bundled:
                lines.append(f"  {s.name}: {s.description[:80]}")
            lines.append("")
        if user:
            lines.append(f"User ({len(user)}):")
            for s in user:
                lines.append(f"  {s.name}: {s.description[:80]}")
            lines.append("")
        if plugin:
            lines.append(f"Plugin ({len(plugin)}):")
            for s in plugin:
                lines.append(f"  {s.name}: {s.description[:80]}")
            lines.append("")

        lines.append(f"User skills dir: {skills_loader.get_user_skills_dir()}")
        return ToolResult(output="\n".join(lines))

    def _info(self, arguments: SkillManageToolInput, context: ToolExecutionContext) -> ToolResult:
        """Show skill metadata."""
        name, err = _validate_name(arguments.name)
        if err:
            return ToolResult(output=err, is_error=True)
        assert name is not None

        registry = load_skill_registry(context.cwd)
        # Try exact, lowercase, and title-case matches.
        skill = registry.get(name) or registry.get(name.lower()) or registry.get(name.title())
        if skill is None:
            return ToolResult(output=f"Skill not found: {name}", is_error=True)

        lines = [
            f"Name: {skill.name}",
            f"Description: {skill.description}",
            f"Source: {skill.source}",
        ]
        if skill.path:
            lines.append(f"Path: {skill.path}")
        lines.append(f"Content length: {len(skill.content)} chars")
        return ToolResult(output="\n".join(lines))

    def _create(self, name: str, arguments: SkillManageToolInput) -> ToolResult:
        """Create a new user skill."""
        # Must not already exist.
        if _is_user_skill(name):
            return ToolResult(
                output=f"Skill already exists: {name}. Use 'update' to replace it or 'edit' to modify part of it.",
                is_error=True,
            )

        # Need content.
        if not arguments.content:
            return ToolResult(
                output="create requires 'content' (the skill body — instructions, procedures, etc.).",
                is_error=True,
            )

        # Build the full file content.
        if arguments.content.strip().startswith("---"):
            # Caller supplied their own frontmatter — validate it.
            full_content = arguments.content.strip()
            fm_err = _validate_frontmatter(full_content)
            if fm_err:
                return ToolResult(output=fm_err, is_error=True)
        else:
            # No frontmatter — need a description to auto-wrap.
            if not arguments.description:
                return ToolResult(
                    output=(
                        "create requires a 'description' field when content has no YAML frontmatter. "
                        "Either include frontmatter in the content or provide description separately."
                    ),
                    is_error=True,
                )
            full_content = _wrap_in_frontmatter(
                name, arguments.description, arguments.content.strip()
            )

        size_err = _validate_content_size(full_content)
        if size_err:
            return ToolResult(output=size_err, is_error=True)

        # Write.
        path = _skill_file_path(name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(full_content, encoding="utf-8")
        except OSError as exc:
            return ToolResult(output=f"Failed to write skill: {exc}", is_error=True)

        return ToolResult(
            output=f"Created skill '{name}' at {path} ({len(full_content)} chars).",
            metadata={"action": "create", "name": name, "path": str(path)},
        )

    def _update(self, name: str, arguments: SkillManageToolInput) -> ToolResult:
        """Replace a skill's content entirely."""
        if not _is_user_skill(name):
            return ToolResult(
                output=f"Skill not found in user dir: {name}. Use 'create' to make a new skill.",
                is_error=True,
            )
        if not arguments.content:
            return ToolResult(output="update requires 'content' (the new full content).", is_error=True)

        full_content = arguments.content.strip()
        # If the caller supplied frontmatter, validate it.
        if full_content.startswith("---"):
            fm_err = _validate_frontmatter(full_content)
            if fm_err:
                return ToolResult(output=fm_err, is_error=True)
        elif arguments.description:
            # No frontmatter but description given — wrap it.
            full_content = _wrap_in_frontmatter(
                name, arguments.description, full_content
            )
        else:
            return ToolResult(
                output="update content must either include YAML frontmatter or be accompanied by a 'description' field.",
                is_error=True,
            )

        size_err = _validate_content_size(full_content)
        if size_err:
            return ToolResult(output=size_err, is_error=True)

        path = _skill_file_path(name)
        try:
            path.write_text(full_content, encoding="utf-8")
        except OSError as exc:
            return ToolResult(output=f"Failed to write skill: {exc}", is_error=True)

        return ToolResult(
            output=f"Updated skill '{name}' at {path} ({len(full_content)} chars).",
            metadata={"action": "update", "name": name, "path": str(path)},
        )

    def _edit(self, name: str, arguments: SkillManageToolInput) -> ToolResult:
        """Find-and-replace within a skill."""
        if not _is_user_skill(name):
            return ToolResult(
                output=f"Skill not found in user dir: {name}. 'edit' only works on user skills.",
                is_error=True,
            )
        if not arguments.old_string:
            return ToolResult(output="edit requires 'old_string' (the text to find).", is_error=True)
        if arguments.new_string is None:
            return ToolResult(output="edit requires 'new_string' (the replacement text).", is_error=True)
        if arguments.old_string == arguments.new_string:
            return ToolResult(output="old_string and new_string must differ.", is_error=True)

        path = _skill_file_path(name)
        try:
            original = path.read_text(encoding="utf-8")
        except OSError as exc:
            return ToolResult(output=f"Failed to read skill: {exc}", is_error=True)

        if arguments.old_string not in original:
            return ToolResult(
                output=f"old_string not found in skill '{name}'.",
                is_error=True,
            )

        if arguments.replace_all:
            count = original.count(arguments.old_string)
            updated = original.replace(arguments.old_string, arguments.new_string)
        else:
            count = 1
            updated = original.replace(arguments.old_string, arguments.new_string, 1)

        size_err = _validate_content_size(updated)
        if size_err:
            return ToolResult(output=size_err, is_error=True)

        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return ToolResult(output=f"Failed to write skill: {exc}", is_error=True)

        return ToolResult(
            output=f"Edited skill '{name}' — {count} replacement(s) at {path}.",
            metadata={
                "action": "edit",
                "name": name,
                "path": str(path),
                "replacements": count,
            },
        )

    def _delete(self, name: str) -> ToolResult:
        """Delete a user skill (including its directory and support files)."""
        if not _is_user_skill(name):
            return ToolResult(
                output=f"Skill not found in user dir: {name}. Only user skills can be deleted.",
                is_error=True,
            )

        skill_dir = _skill_dir_path(name)
        path = _skill_file_path(name)
        try:
            # Remove the SKILL.md file.
            path.unlink()
            # Remove the skill directory if it's now empty (or only has empty subdirs).
            # Adapted from reference _delete_skill which does shutil.rmtree.
            import shutil
            # Only rmtree if the directory contains no other skills.
            other_skills = list(skill_dir.rglob("SKILL.md"))
            if not other_skills:
                shutil.rmtree(skill_dir, ignore_errors=True)
        except OSError as exc:
            return ToolResult(output=f"Failed to delete skill: {exc}", is_error=True)

        return ToolResult(
            output=f"Deleted skill '{name}' ({path}).",
            metadata={"action": "delete", "name": name, "path": str(path)},
        )

    def _write_file(self, name: str, arguments: SkillManageToolInput) -> ToolResult:
        """Add or overwrite a support file in a user skill directory.

        Adapted from reference _write_file. The file must be under one of the
        support directories: references/, templates/, scripts/, assets/.
        """
        if not arguments.file_path:
            return ToolResult(output="write_file requires file_path", is_error=True)
        if arguments.file_content is None:
            return ToolResult(output="write_file requires file_content", is_error=True)

        # Validate file_path (adapted from reference _validate_file_path).
        from niaharness.tools.skill_utils import SKILL_SUPPORT_DIRS

        safe_path = Path(arguments.file_path)
        if safe_path.is_absolute() or ".." in safe_path.parts:
            return ToolResult(output=f"Invalid file_path: {arguments.file_path}", is_error=True)

        parts = safe_path.parts
        if len(parts) < 2 or parts[0] not in SKILL_SUPPORT_DIRS:
            return ToolResult(
                output=(
                    f"file_path must start with one of: {', '.join(sorted(SKILL_SUPPORT_DIRS))}. "
                    f"Got: {arguments.file_path}"
                ),
                is_error=True,
            )

        # Check skill exists.
        if not _is_user_skill(name):
            return ToolResult(
                output=f"Skill not found in user dir: {name}. write_file only works on user skills.",
                is_error=True,
            )

        skill_dir = _skill_dir_path(name)
        target = (skill_dir / safe_path).resolve()
        try:
            skill_root = skill_dir.resolve()
            if not target.is_relative_to(skill_root):
                return ToolResult(output=f"File path escapes skill directory: {arguments.file_path}", is_error=True)
        except (ValueError, OSError):
            return ToolResult(output=f"Invalid file path: {arguments.file_path}", is_error=True)

        # Size check (1 MiB max per support file).
        if len(arguments.file_content) > 1_048_576:
            return ToolResult(
                output=f"File content too large ({len(arguments.file_content):,} bytes, max 1 MiB).",
                is_error=True,
            )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(arguments.file_content, encoding="utf-8")
        except OSError as exc:
            return ToolResult(output=f"Failed to write file: {exc}", is_error=True)

        return ToolResult(
            output=f"Wrote {target.relative_to(skill_dir)} in skill '{name}' ({len(arguments.file_content)} bytes).",
            metadata={"action": "write_file", "name": name, "file_path": str(target)},
        )

    def _remove_file(self, name: str, arguments: SkillManageToolInput) -> ToolResult:
        """Remove a support file from a user skill directory.

        Adapted from reference _remove_file.
        """
        if not arguments.file_path:
            return ToolResult(output="remove_file requires file_path", is_error=True)

        from niaharness.tools.skill_utils import SKILL_SUPPORT_DIRS

        safe_path = Path(arguments.file_path)
        if safe_path.is_absolute() or ".." in safe_path.parts:
            return ToolResult(output=f"Invalid file_path: {arguments.file_path}", is_error=True)

        parts = safe_path.parts
        if len(parts) < 2 or parts[0] not in SKILL_SUPPORT_DIRS:
            return ToolResult(
                output=f"file_path must start with one of: {', '.join(sorted(SKILL_SUPPORT_DIRS))}.",
                is_error=True,
            )

        if not _is_user_skill(name):
            return ToolResult(
                output=f"Skill not found in user dir: {name}. remove_file only works on user skills.",
                is_error=True,
            )

        skill_dir = _skill_dir_path(name)
        target = (skill_dir / safe_path).resolve()
        try:
            skill_root = skill_dir.resolve()
            if not target.is_relative_to(skill_root):
                return ToolResult(output=f"File path escapes skill directory: {arguments.file_path}", is_error=True)
        except (ValueError, OSError):
            return ToolResult(output=f"Invalid file path: {arguments.file_path}", is_error=True)

        if not target.exists():
            return ToolResult(output=f"File not found: {arguments.file_path}", is_error=True)

        try:
            target.unlink()
            # Clean up empty parent directories.
            parent = target.parent
            while parent != skill_dir and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
        except OSError as exc:
            return ToolResult(output=f"Failed to remove file: {exc}", is_error=True)

        return ToolResult(
            output=f"Removed {arguments.file_path} from skill '{name}'.",
            metadata={"action": "remove_file", "name": name, "file_path": arguments.file_path},
        )
