"""Skill hub tool — browse, search, and install optional skills.

Adapted from the reference project's skills hub pattern.

Lets the agent browse the optional skill catalog, search for skills by
keyword, install skills on demand, and uninstall them. Installed skills
are copied to the user skills directory and become immediately loadable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from niaharness.skills.hub import (
    install_skill,
    list_optional_skills,
    search_optional_skills,
    uninstall_skill,
)
from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class SkillHubInput(BaseModel):
    """Arguments for the skill_hub tool."""

    action: Literal["browse", "search", "install", "uninstall"] = Field(
        description=(
            "browse: list all optional skills. "
            "search: find skills by keyword. "
            "install: install a skill to the user directory. "
            "uninstall: remove a user-installed skill."
        )
    )
    query: str | None = Field(
        default=None,
        description="Search query (for 'search' action).",
    )
    name: str | None = Field(
        default=None,
        description="Skill name (for 'install' and 'uninstall' actions).",
    )


class SkillHubTool(BaseTool):
    """Browse, search, and install optional skills."""

    name = "skill_hub"
    description = (
        "Browse the optional skill catalog, search for skills by keyword, "
        "install skills on demand, or uninstall them. Installed skills are "
        "immediately available via the skill tool and slash commands."
    )
    input_model = SkillHubInput

    def is_read_only(self, arguments: SkillHubInput) -> bool:
        return arguments.action in ("browse", "search")

    async def execute(self, arguments: SkillHubInput, context: ToolExecutionContext) -> ToolResult:
        if arguments.action == "browse":
            return self._browse()
        if arguments.action == "search":
            return self._search(arguments)
        if arguments.action == "install":
            return self._install(arguments)
        if arguments.action == "uninstall":
            return self._uninstall(arguments)
        return ToolResult(output=f"Unknown action: {arguments.action}", is_error=True)

    def _browse(self) -> ToolResult:
        """List all optional skills."""
        skills = list_optional_skills()
        if not skills:
            return ToolResult(output="No optional skills available.")

        installed_count = sum(1 for s in skills if s.installed)
        lines = [
            f"Optional skills ({len(skills)} total, {installed_count} already installed):",
            "",
        ]
        for s in sorted(skills, key=lambda x: (x.category, x.name)):
            status = "✓ installed" if s.installed else "  available"
            lines.append(f"  [{s.category}] {s.name:<25} {status}")
            lines.append(f"    {s.description[:70]}")

        lines.append("")
        lines.append("Use: skill_hub(action='install', name='<skill-name>') to install.")
        return ToolResult(output="\n".join(lines), metadata={"count": len(skills)})

    def _search(self, args: SkillHubInput) -> ToolResult:
        """Search optional skills by keyword."""
        if not args.query:
            return ToolResult(output="search requires 'query'", is_error=True)

        results = search_optional_skills(args.query)
        if not results:
            return ToolResult(output=f"No skills found matching '{args.query}'.")

        lines = [f"Found {len(results)} skill(s) matching '{args.query}':", ""]
        for s in results:
            status = "✓ installed" if s.installed else "  available"
            lines.append(f"  [{s.category}] {s.name:<25} {status}")
            lines.append(f"    {s.description[:70]}")

        return ToolResult(output="\n".join(lines), metadata={"count": len(results)})

    def _install(self, args: SkillHubInput) -> ToolResult:
        """Install a skill."""
        if not args.name:
            return ToolResult(output="install requires 'name'", is_error=True)

        success, message = install_skill(args.name)
        return ToolResult(output=message, is_error=not success)

    def _uninstall(self, args: SkillHubInput) -> ToolResult:
        """Uninstall a skill."""
        if not args.name:
            return ToolResult(output="uninstall requires 'name'", is_error=True)

        success, message = uninstall_skill(args.name)
        return ToolResult(output=message, is_error=not success)
