"""Standalone ``skill_view`` tool — read a skill's SKILL.md or a support file.

Split from ``skill_manage`` so the agent has a small, focused read-only
tool for skill content access (matching the reference project's toolset
split). This tool is the read counterpart to ``skills_list``.

The ``file_path`` parameter enables progressive disclosure: the agent
first calls ``skill_view(name="github-pr-workflow")`` to read SKILL.md,
then calls ``skill_view(name="github-pr-workflow", file_path="references/
conventional-commits.md")`` to read a specific support file.

On a successful view, the skill's usage counter is bumped (best-effort,
telemetry never breaks the tool call). This mirrors the reference
project's ``_skill_view_with_bump`` pattern.

Support file directories (adapted from reference):
- ``references/`` — session-specific detail, knowledge banks
- ``templates/`` — starter files meant to be copied
- ``scripts/`` — runnable scripts the skill can invoke
- ``assets/`` — images, configs, other binary assets
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult
from niaharness.tools.skills_loader import load_skill_registry
from niaharness.tools.skill_utils import SKILL_SUPPORT_DIRS

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 64
MAX_FILE_SIZE = 1_048_576  # 1 MiB per support file


class SkillViewToolInput(BaseModel):
    """Arguments for the skill_view tool."""

    name: str = Field(
        description=(
            "The skill name (use skills_list to see available skills). "
            "For plugin-provided skills, use the qualified form 'plugin:skill'."
        ),
    )
    file_path: Optional[str] = Field(
        default=None,
        description=(
            "OPTIONAL: Path to a linked file within the skill "
            "(e.g. 'references/api.md', 'templates/config.yaml', "
            "'scripts/validate.py'). Omit to get the main SKILL.md content. "
            "Must be under one of: references/, templates/, scripts/, assets/."
        ),
    )


class SkillViewTool(BaseTool):
    """Read a skill's SKILL.md content or a specific support file.

    First call returns SKILL.md content plus a list of available support
    files (references/, templates/, scripts/, assets/) so the agent knows
    what else is in the skill. Subsequent calls with ``file_path`` read
    individual support files (progressive disclosure).
    """

    name = "skill_view"
    description = (
        "Skills allow for loading information about specific tasks and "
        "workflows, as well as scripts and templates. Load a skill's full "
        "content or access its linked files (references, templates, scripts). "
        "First call returns SKILL.md content plus a 'linked_files' listing "
        "showing available references/templates/scripts. To access those, "
        "call again with file_path parameter."
    )
    input_model = SkillViewToolInput

    def is_read_only(self, arguments: SkillViewToolInput) -> bool:
        del arguments
        return True

    async def execute(
        self, arguments: SkillViewToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        if not arguments.name:
            return ToolResult(output="skill_view requires 'name'", is_error=True)

        registry = load_skill_registry(context.cwd)
        skill = (
            registry.get(arguments.name)
            or registry.get(arguments.name.lower())
            or registry.get(arguments.name.title())
        )
        if skill is None:
            return ToolResult(
                output=f"Skill not found: {arguments.name}", is_error=True
            )

        # If file_path is specified, read a support file.
        if arguments.file_path:
            result = self._read_support_file(skill, arguments.file_path)
        else:
            result = self._read_skill_md(skill)

        # Best-effort: bump usage counter on successful view.
        if not result.is_error:
            try:
                from niaharness.tools.skill_usage import bump_view, bump_use

                bump_view(skill.name)
                # A skill_view tool call is the agent actively loading the
                # skill to act on it — that counts as use, not just browse.
                bump_use(skill.name)
            except Exception as exc:
                logger.debug("skill_usage bump failed (non-fatal): %s", exc)

        return result

    # ---- internal helpers --------------------------------------------------

    def _read_skill_md(self, skill) -> ToolResult:
        """Read SKILL.md content + enumerate support files."""
        content = skill.content
        linked_files = self._enumerate_support_files(skill)

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

        Includes path traversal protection (adapted from reference
        _validate_file_path). The file must be under one of the support
        directories: references/, templates/, scripts/, assets/.
        """
        if not skill.path:
            return ToolResult(
                output="Skill has no path — cannot read support files.",
                is_error=True,
            )

        skill_dir = Path(skill.path).parent

        # Path traversal protection.
        safe_path = Path(file_path)
        if safe_path.is_absolute() or ".." in safe_path.parts:
            return ToolResult(
                output=f"Invalid file_path (absolute or traversal): {file_path}",
                is_error=True,
            )

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
                return ToolResult(
                    output=f"File path escapes skill directory: {file_path}",
                    is_error=True,
                )
        except (ValueError, OSError):
            return ToolResult(
                output=f"Invalid file path: {file_path}", is_error=True
            )

        # Reject symlinks (defense in depth — even after resolve()).
        try:
            if target.is_symlink():
                return ToolResult(
                    output=f"Refusing to read symlinked support file: {file_path}",
                    is_error=True,
                )
        except OSError:
            return ToolResult(
                output=f"Cannot stat support file: {file_path}", is_error=True
            )

        if not target.exists():
            return ToolResult(
                output=f"Support file not found: {file_path}", is_error=True
            )
        if not target.is_file():
            return ToolResult(
                output=f"Not a file: {file_path}", is_error=True
            )

        # Size check.
        try:
            size = target.stat().st_size
            if size > MAX_FILE_SIZE:
                return ToolResult(
                    output=(
                        f"File too large ({size:,} bytes, max {MAX_FILE_SIZE:,}). "
                        "Use read_file with offset/limit."
                    ),
                    is_error=True,
                )
        except OSError:
            pass

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Binary file — return metadata only.
            try:
                size = target.stat().st_size
            except OSError:
                size = 0
            return ToolResult(
                output=f"Binary file: {file_path} ({size} bytes). Use read_file to access.",
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


__all__ = ["SkillViewTool", "SkillViewToolInput"]
