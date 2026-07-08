"""Standalone ``skills_list`` tool — list installed skills with metadata.

Split from ``skill_manage`` so the agent has a small, focused read-only
tool for discovery (matching the reference project's toolset split).
``skill_manage`` continues to handle write operations (create/update/edit/
delete/write_file/remove_file); the existing combined ``skill`` tool
remains for backward compatibility.

Adapted from the reference project's tools/skills_tool.py:skills_list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult
from niaharness.tools.skills_loader import load_skill_registry


class SkillsListToolInput(BaseModel):
    """Arguments for the skills_list tool."""

    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional category filter to narrow results "
            "(e.g. 'github', 'software-development', 'productivity'). "
            "When omitted, all skills are listed."
        ),
    )


class SkillsListTool(BaseTool):
    """List all installed skills (read-only, no arguments required).

    Returns a compact summary grouped by source (bundled / user / plugin)
    with name + description + category. Use ``skill_view(name=...)`` to
    load a skill's full content.
    """

    name = "skills_list"
    description = (
        "List available skills (name + description + category). "
        "Use skill_view(name) to load full content. Use skill(action='view', "
        "name=..., file_path=...) to read support files (references/, "
        "templates/, scripts/, assets/)."
    )
    input_model = SkillsListToolInput

    def is_read_only(self, arguments: SkillsListToolInput) -> bool:
        del arguments
        return True

    async def execute(
        self, arguments: SkillsListToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        registry = load_skill_registry(context.cwd)
        skills = registry.list_skills()

        # Optional category filter.
        if arguments.category:
            cat_lower = arguments.category.lower()
            filtered = []
            for s in skills:
                # Derive category from path: bundled/<category>/<skill>/SKILL.md.
                cat = _derive_category(s.path) if s.path else ""
                if cat.lower() == cat_lower:
                    filtered.append(s)
            skills = filtered

        if not skills:
            if arguments.category:
                return ToolResult(
                    output=f"No skills found in category '{arguments.category}'."
                )
            return ToolResult(output="No skills available.")

        # Group by source.
        bundled = [s for s in skills if s.source == "bundled"]
        user = [s for s in skills if s.source == "user"]
        plugin = [s for s in skills if s.source not in ("bundled", "user")]

        lines = [f"Installed skills ({len(skills)} total):", ""]

        if bundled:
            lines.append(f"Bundled ({len(bundled)}):")
            for s in sorted(bundled, key=lambda x: x.name):
                cat = _derive_category(s.path) if s.path else ""
                cat_str = f" [{cat}]" if cat else ""
                lines.append(f"  {s.name}{cat_str}: {s.description[:80]}")
            lines.append("")
        if user:
            lines.append(f"User ({len(user)}):")
            for s in sorted(user, key=lambda x: x.name):
                lines.append(f"  {s.name}: {s.description[:80]}")
            lines.append("")
        if plugin:
            lines.append(f"Plugin ({len(plugin)}):")
            for s in sorted(plugin, key=lambda x: x.name):
                lines.append(f"  {s.name}: {s.description[:80]}")
            lines.append("")

        lines.append("Use: skill_view(name='<skill-name>') to read full content.")
        lines.append(
            "Use: skill(action='view', name='<skill>', file_path='references/...') "
            "to read support files."
        )
        return ToolResult(output="\n".join(lines))


def _derive_category(skill_path: str) -> str:
    """Derive a category from a skill path: bundled/<cat>/<skill>/SKILL.md → 'cat'."""
    try:
        parts = Path(skill_path).parts
        # Look for 'bundled' or 'optional' in the path, then take the next segment.
        for marker in ("bundled", "optional"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 2 < len(parts):  # marker/<cat>/<skill>/SKILL.md
                    return parts[idx + 1]
        return ""
    except (ValueError, IndexError):
        return ""


__all__ = ["SkillsListTool", "SkillsListToolInput"]
