"""Tests for the skill_manage tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from niaharness.tools.base import ToolExecutionContext
from niaharness.tools.skill_manage_tool import SkillManageTool, SkillManageToolInput


@pytest.fixture
def isolated_skills_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the user skills dir to a temp directory."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "niaharness.skills.loader.get_user_skills_dir",
        lambda: skills_dir,
    )
    return skills_dir


@pytest.fixture
def context(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestCreate:
    @pytest.mark.asyncio
    async def test_create_with_auto_frontmatter(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        result = await SkillManageTool().execute(
            SkillManageToolInput(
                action="create",
                name="my-skill",
                description="A test skill for unit testing.",
                content="## Steps\n1. Do thing one\n2. Do thing two\n",
            ),
            context,
        )
        assert result.is_error is False
        assert "Created skill 'my-skill'" in result.output
        path = isolated_skills_dir / "my-skill.md"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert content.startswith("---")
        assert "name: my-skill" in content
        assert "description: A test skill for unit testing." in content
        assert "Do thing one" in content

    @pytest.mark.asyncio
    async def test_create_with_explicit_frontmatter(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        full_content = """\
---
name: custom-skill
description: Custom skill with explicit frontmatter.
---

# Custom Skill

Do the thing.
"""
        result = await SkillManageTool().execute(
            SkillManageToolInput(
                action="create",
                name="custom-skill",
                content=full_content,
            ),
            context,
        )
        assert result.is_error is False
        path = isolated_skills_dir / "custom-skill.md"
        assert path.exists()
        assert path.read_text(encoding="utf-8").strip() == full_content.strip()

    @pytest.mark.asyncio
    async def test_create_already_exists(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        # Pre-create the skill.
        (isolated_skills_dir / "exists.md").write_text("---\nname: exists\ndescription: x\n---\nbody", encoding="utf-8")

        result = await SkillManageTool().execute(
            SkillManageToolInput(
                action="create",
                name="exists",
                description="desc",
                content="body",
            ),
            context,
        )
        assert result.is_error is True
        assert "already exists" in result.output

    @pytest.mark.asyncio
    async def test_create_missing_description_without_frontmatter(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        """When content has no frontmatter, description is required."""
        result = await SkillManageTool().execute(
            SkillManageToolInput(
                action="create",
                name="no-desc",
                content="body without frontmatter",
            ),
            context,
        )
        assert result.is_error is True
        assert "description" in result.output.lower()

    @pytest.mark.asyncio
    async def test_create_missing_content(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        result = await SkillManageTool().execute(
            SkillManageToolInput(
                action="create",
                name="no-content",
                description="desc",
            ),
            context,
        )
        assert result.is_error is True
        assert "content" in result.output.lower()

    @pytest.mark.asyncio
    async def test_create_invalid_name(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        result = await SkillManageTool().execute(
            SkillManageToolInput(
                action="create",
                name="UPPERCASE!",
                description="desc",
                content="body",
            ),
            context,
        )
        assert result.is_error is True
        assert "Invalid skill name" in result.output

    @pytest.mark.asyncio
    async def test_create_path_traversal_blocked(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        result = await SkillManageTool().execute(
            SkillManageToolInput(
                action="create",
                name="../escape",
                description="desc",
                content="body",
            ),
            context,
        )
        assert result.is_error is True
        # Could fail at name validation or path-traversal check
        assert "Invalid" in result.output or "forbidden" in result.output


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    @pytest.mark.asyncio
    async def test_update_replaces_content(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        # Pre-create.
        path = isolated_skills_dir / "upd.md"
        path.write_text("---\nname: upd\ndescription: old\n---\nold body", encoding="utf-8")

        result = await SkillManageTool().execute(
            SkillManageToolInput(
                action="update",
                name="upd",
                description="new desc",
                content="## New body\nUpdated content.",
            ),
            context,
        )
        assert result.is_error is False
        content = path.read_text(encoding="utf-8")
        assert "new desc" in content
        assert "Updated content" in content
        assert "old body" not in content

    @pytest.mark.asyncio
    async def test_update_not_found(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        result = await SkillManageTool().execute(
            SkillManageToolInput(
                action="update",
                name="missing",
                content="body",
            ),
            context,
        )
        assert result.is_error is True
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


class TestEdit:
    @pytest.mark.asyncio
    async def test_edit_first_occurrence(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        path = isolated_skills_dir / "edit.md"
        path.write_text("---\nname: edit\ndescription: d\n---\nfoo bar foo bar", encoding="utf-8")

        result = await SkillManageTool().execute(
            SkillManageToolInput(
                action="edit",
                name="edit",
                old_string="foo",
                new_string="baz",
            ),
            context,
        )
        assert result.is_error is False
        content = path.read_text(encoding="utf-8")
        assert "baz bar foo bar" == content.split("---\n")[-1]

    @pytest.mark.asyncio
    async def test_edit_replace_all(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        path = isolated_skills_dir / "editall.md"
        path.write_text("---\nname: editall\ndescription: d\n---\nfoo bar foo bar", encoding="utf-8")

        result = await SkillManageTool().execute(
            SkillManageToolInput(
                action="edit",
                name="editall",
                old_string="foo",
                new_string="baz",
                replace_all=True,
            ),
            context,
        )
        assert result.is_error is False
        content = path.read_text(encoding="utf-8")
        assert "baz bar baz bar" == content.split("---\n")[-1]

    @pytest.mark.asyncio
    async def test_edit_string_not_found(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        path = isolated_skills_dir / "notfound.md"
        path.write_text("---\nname: notfound\ndescription: d\n---\nhello world", encoding="utf-8")

        result = await SkillManageTool().execute(
            SkillManageToolInput(
                action="edit",
                name="notfound",
                old_string="missing",
                new_string="present",
            ),
            context,
        )
        assert result.is_error is True
        assert "not found" in result.output.lower()

    @pytest.mark.asyncio
    async def test_edit_same_old_new(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        path = isolated_skills_dir / "same.md"
        path.write_text("---\nname: same\ndescription: d\n---\nbody", encoding="utf-8")

        result = await SkillManageTool().execute(
            SkillManageToolInput(
                action="edit",
                name="same",
                old_string="body",
                new_string="body",
            ),
            context,
        )
        assert result.is_error is True
        assert "must differ" in result.output


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_user_skill(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        path = isolated_skills_dir / "del.md"
        path.write_text("---\nname: del\ndescription: d\n---\nbody", encoding="utf-8")
        assert path.exists()

        result = await SkillManageTool().execute(
            SkillManageToolInput(action="delete", name="del"),
            context,
        )
        assert result.is_error is False
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_delete_not_found(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        result = await SkillManageTool().execute(
            SkillManageToolInput(action="delete", name="missing"),
            context,
        )
        assert result.is_error is True
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# list + info
# ---------------------------------------------------------------------------


class TestListInfo:
    @pytest.mark.asyncio
    async def test_list_includes_bundled_and_user(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        # Add a user skill.
        (isolated_skills_dir / "user-skill.md").write_text(
            "---\nname: user-skill\ndescription: A user skill.\n---\nbody", encoding="utf-8"
        )

        result = await SkillManageTool().execute(
            SkillManageToolInput(action="list"),
            context,
        )
        assert result.is_error is False
        assert "Bundled" in result.output
        assert "User" in result.output
        assert "user-skill" in result.output

    @pytest.mark.asyncio
    async def test_info_shows_metadata(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        (isolated_skills_dir / "info-test.md").write_text(
            "---\nname: info-test\ndescription: Info test skill.\n---\n## Body\nContent here.",
            encoding="utf-8",
        )

        result = await SkillManageTool().execute(
            SkillManageToolInput(action="info", name="info-test"),
            context,
        )
        assert result.is_error is False
        assert "Name: info-test" in result.output
        assert "Source: user" in result.output
        assert "Content length:" in result.output

    @pytest.mark.asyncio
    async def test_info_not_found(
        self, isolated_skills_dir: Path, context: ToolExecutionContext
    ):
        result = await SkillManageTool().execute(
            SkillManageToolInput(action="info", name="missing"),
            context,
        )
        assert result.is_error is True
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# read-only flag
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_list_is_read_only(self):
        tool = SkillManageTool()
        assert tool.is_read_only(SkillManageToolInput(action="list")) is True

    def test_info_is_read_only(self):
        tool = SkillManageTool()
        assert tool.is_read_only(SkillManageToolInput(action="info", name="x")) is True

    def test_create_is_not_read_only(self):
        tool = SkillManageTool()
        assert tool.is_read_only(SkillManageToolInput(action="create", name="x", description="d", content="c")) is False

    def test_delete_is_not_read_only(self):
        tool = SkillManageTool()
        assert tool.is_read_only(SkillManageToolInput(action="delete", name="x")) is False
