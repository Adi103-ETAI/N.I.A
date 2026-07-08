"""Skill viewing tool — read skill content and support files.

Adapted from the reference project's tools/skills_tool.py (skill_view + skills_list).

Two operations:
- ``view``: Load a skill's SKILL.md content (or a specific support file via
  ``file_path``). Also enumerates linked_files (references/, templates/,
  scripts/, assets/) so the agent knows what's available.
- ``list``: List all installed skills with name, description, source, category.

The ``file_path`` parameter enables progressive disclosure: the agent first
calls ``skill(action="view", name="github-pr-workflow")`` to read the SKILL.md,
then calls ``skill(action="view", name="github-pr-workflow", file_path="references/conventional-commits.md")``
to read a specific support file.

Support file directories (adapted from reference):
- ``references/`` — session-specific detail, knowledge banks
- ``templates/`` — starter files meant to be copied
- ``scripts/`` — runnable scripts the skill can invoke
- ``assets/`` — images, configs, other binary assets
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult
from niaharness.tools.skills_loader import load_skill_registry
from niaharness.tools.skill_utils import SKILL_SUPPORT_DIRS

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 64
MAX_FILE_SIZE = 1_048_576  # 1 MiB per support file


class SkillToolInput(BaseModel):
    """Arguments for the skill tool."""

    action: Literal["view", "list"] = Field(
        default="view",
        description="view: read a skill's content or a support file. list: list all skills.",
    )
    name: str = Field(
        description="Skill name (for 'view' action).",
    )
    file_path: str | None = Field(
        default=None,
        description=(
            "Optional path to a support file within the skill directory "
            "(e.g. 'references/conventional-commits.md', 'templates/pr-body-feature.md', "
            "'scripts/gh-env.sh'). When omitted, reads the SKILL.md."
        ),
    )


class SkillTool(BaseTool):
    """Read skill content and support files with progressive disclosure."""

    name = "skill"
    description = (
        "Read a skill's SKILL.md content or a specific support file "
        "(references/, templates/, scripts/, assets/). When reading SKILL.md, "
        "also lists available support files so you know what else is in the skill."
    )
    input_model = SkillToolInput

    def is_read_only(self, arguments: SkillToolInput) -> bool:
        del arguments
        return True

    async def execute(self, arguments: SkillToolInput, context: ToolExecutionContext) -> ToolResult:
        if arguments.action == "list":
            return self._list_skills(context)

        # action == "view"
        if not arguments.name:
            return ToolResult(output="view requires 'name'", is_error=True)

        registry = load_skill_registry(context.cwd)
        skill = (
            registry.get(arguments.name)
            or registry.get(arguments.name.lower())
            or registry.get(arguments.name.title())
        )
        if skill is None:
            return ToolResult(output=f"Skill not found: {arguments.name}", is_error=True)

        # If file_path is specified, read a support file (adapted from reference skill_view).
        if arguments.file_path:
            return self._read_support_file(skill, arguments.file_path)

        # Read SKILL.md content.
        content = skill.content

        # Enumerate linked files (adapted from reference skill_view linked_files).
        linked_files = self._enumerate_support_files(skill)

        # Build output with skill content + support file listing.
        lines = [content]
        if linked_files:
            lines.append("")
            lines.append("---")
            lines.append("Supporting files (use file_path to read):")
            for f in linked_files:
                lines.append(f"  {f}")

        return ToolResult(
            output="\n".join(lines),
            metadata={
                "name": skill.name,
                "source": skill.source,
                "path": skill.path,
                "linked_files": linked_files,
            },
        )

    def _read_support_file(self, skill, file_path: str) -> ToolResult:
        """Read a support file from a skill directory.

        Adapted from reference skill_view file_path handling.
        Includes path traversal protection.
        """
        if not skill.path:
            return ToolResult(output="Skill has no path — cannot read support files.", is_error=True)

        skill_dir = Path(skill.path).parent

        # Path traversal protection (adapted from reference _validate_file_path).
        safe_path = Path(file_path)
        if safe_path.is_absolute() or ".." in safe_path.parts:
            return ToolResult(output=f"Invalid file_path: {file_path}", is_error=True)

        # Ensure the file is under a support directory.
        parts = safe_path.parts
        if len(parts) < 2 or parts[0] not in SKILL_SUPPORT_DIRS:
            return ToolResult(
                output=(
                    f"file_path must be under a support directory "
                    f"({', '.join(sorted(SKILL_SUPPORT_DIRS))}). Got: {file_path}"
                ),
                is_error=True,
            )

        target = (skill_dir / safe_path).resolve()
        try:
            skill_root = skill_dir.resolve()
            if not target.is_relative_to(skill_root):
                return ToolResult(output=f"File path escapes skill directory: {file_path}", is_error=True)
        except (ValueError, OSError):
            return ToolResult(output=f"Invalid file path: {file_path}", is_error=True)

        if not target.exists():
            return ToolResult(output=f"Support file not found: {file_path}", is_error=True)
        if not target.is_file():
            return ToolResult(output=f"Not a file: {file_path}", is_error=True)

        # Size check.
        try:
            size = target.stat().st_size
            if size > MAX_FILE_SIZE:
                return ToolResult(
                    output=f"File too large ({size:,} bytes, max {MAX_FILE_SIZE:,}). Use read_file with offset/limit.",
                    is_error=True,
                )
        except OSError:
            pass

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Binary file — return metadata only.
            return ToolResult(
                output=f"Binary file: {file_path} ({target.stat().st_size} bytes). Use read_file to access.",
                metadata={"file_path": str(target), "binary": True},
            )

        return ToolResult(
            output=content,
            metadata={
                "skill_name": skill.name,
                "file_path": str(target),
                "relative_path": file_path,
            },
        )

    def _enumerate_support_files(self, skill) -> list[str]:
        """Enumerate support files in a skill directory.

        Adapted from reference skill_view linked_files enumeration.
        Scans references/, templates/, scripts/, assets/ subdirectories.
        """
        if not skill.path:
            return []

        skill_dir = Path(skill.path).parent
        files: list[str] = []

        for support_dir_name in sorted(SKILL_SUPPORT_DIRS):
            support_dir = skill_dir / support_dir_name
            if not support_dir.is_dir():
                continue
            for f in sorted(support_dir.rglob("*")):
                if not f.is_file():
                    continue
                if f.name.startswith(".") or f.suffix == ".pyc":
                    continue
                rel = f.relative_to(skill_dir)
                files.append(str(rel))

        return files

    def _list_skills(self, context: ToolExecutionContext) -> ToolResult:
        """List all installed skills with metadata."""
        registry = load_skill_registry(context.cwd)
        skills = registry.list_skills()
        if not skills:
            return ToolResult(output="No skills available.")

        lines = [f"Installed skills ({len(skills)}):", ""]
        # Group by source.
        bundled = [s for s in skills if s.source == "bundled"]
        user = [s for s in skills if s.source == "user"]
        plugin = [s for s in skills if s.source not in ("bundled", "user")]

        if bundled:
            lines.append(f"Bundled ({len(bundled)}):")
            for s in sorted(bundled, key=lambda x: x.name):
                lines.append(f"  {s.name}: {s.description[:70]}")
            lines.append("")
        if user:
            lines.append(f"User ({len(user)}):")
            for s in sorted(user, key=lambda x: x.name):
                lines.append(f"  {s.name}: {s.description[:70]}")
            lines.append("")
        if plugin:
            lines.append(f"Plugin ({len(plugin)}):")
            for s in sorted(plugin, key=lambda x: x.name):
                lines.append(f"  {s.name}: {s.description[:70]}")
            lines.append("")

        lines.append("Use: skill(action='view', name='<skill-name>') to read a skill.")
        lines.append("Use: skill(action='view', name='<skill>', file_path='references/...') to read support files.")
        return ToolResult(output="\n".join(lines))
