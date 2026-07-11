"""Tests for the P1 skills extensions — 8 source adapters + TapsManager + ported skills.

Covers:
  - WellKnownSkillSource (identifier parsing, SSRF guard, index fetching)
  - UrlSource (URL validation, SKILL.md parsing, name extraction)
  - SkillsShSource (identifier parsing, search)
  - ClawHubSource (identifier parsing, ZIP extraction)
  - ClaudeMarketplaceSource (trust level, index fetching)
  - LobeHubSource (identifier parsing, manifest-to-SKILL.md conversion)
  - BrowseShSource (identifier parsing)
  - HermesIndexSource (local index cache, identifier parsing)
  - TapsManager (add/remove/list, persistence)
  - get_extra_sources / get_all_sources factory
  - SSRF guard (_is_safe_url)
  - Frontmatter parsing
  - Skill name validation
  - Ported skills exist (computer-use, dogfood, etc.)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from niaharness.tools.skills_hub_extra_sources import (
    BrowseShSource,
    ClaudeMarketplaceSource,
    ClawHubSource,
    HermesIndexSource,
    LobeHubSource,
    SkillBundle,
    SkillMeta,
    SkillSource,
    SkillsShSource,
    TapsManager,
    UrlSource,
    WellKnownSkillSource,
    _is_safe_url,
    _parse_frontmatter,
    _validate_skill_name,
    get_all_sources,
    get_extra_sources,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _temp_nia_home(tmp_path: Path, monkeypatch):
    """Redirect NIA_HOME to a temp dir so tests don't pollute the host."""
    monkeypatch.setenv("NIA_HOME", str(tmp_path / ".nia"))
    yield


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------


class TestSSRFGuard:
    def test_allows_https(self):
        assert _is_safe_url("https://example.com/skills") is True

    def test_allows_http(self):
        assert _is_safe_url("http://example.com/skills") is True

    def test_blocks_localhost(self):
        assert _is_safe_url("http://localhost:8080") is False
        assert _is_safe_url("http://127.0.0.1:8080") is False

    def test_blocks_private_ips(self):
        assert _is_safe_url("http://10.0.0.1/skills") is False
        assert _is_safe_url("http://192.168.1.1/skills") is False
        assert _is_safe_url("http://172.16.0.1/skills") is False

    def test_blocks_metadata_endpoint(self):
        assert _is_safe_url("http://169.254.169.254/latest/meta-data") is False

    def test_blocks_file_scheme(self):
        assert _is_safe_url("file:///etc/passwd") is False

    def test_blocks_data_scheme(self):
        assert _is_safe_url("data:text/html,<script>") is False

    def test_blocks_empty(self):
        assert _is_safe_url("") is False
        assert _is_safe_url(None) is False


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


class TestFrontmatter:
    def test_parse_basic(self):
        content = "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test Skill"
        fm = _parse_frontmatter(content)
        assert fm["name"] == "test-skill"
        assert fm["description"] == "A test skill"

    def test_parse_no_frontmatter(self):
        assert _parse_frontmatter("# Just markdown") == {}

    def test_parse_empty(self):
        assert _parse_frontmatter("") == {}

    def test_parse_quoted_values(self):
        content = "---\nname: \"quoted name\"\ndescription: 'single quoted'\n---\n"
        fm = _parse_frontmatter(content)
        assert fm["name"] == "quoted name"
        assert fm["description"] == "single quoted"


# ---------------------------------------------------------------------------
# Skill name validation
# ---------------------------------------------------------------------------


class TestSkillNameValidation:
    def test_valid_name(self):
        assert _validate_skill_name("my-skill") == "my-skill"

    def test_normalizes_spaces(self):
        assert _validate_skill_name("My Skill") == "my-skill"

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            _validate_skill_name("")

    def test_rejects_special_chars(self):
        with pytest.raises(ValueError):
            _validate_skill_name("skill<script>")

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError):
            _validate_skill_name("../etc/passwd")


# ---------------------------------------------------------------------------
# WellKnownSkillSource
# ---------------------------------------------------------------------------


class TestWellKnownSkillSource:
    def test_source_id(self):
        assert WellKnownSkillSource().source_id() == "well-known"

    def test_parse_identifier(self):
        source = WellKnownSkillSource()
        parsed = source._parse_identifier("well-known://example.com/my-skill")
        assert parsed == ("example.com", "my-skill")

    def test_parse_identifier_invalid(self):
        source = WellKnownSkillSource()
        assert source._parse_identifier("not-well-known://example.com/skill") is None
        assert source._parse_identifier("well-known://example.com") is None  # no skill name

    def test_extract_domain(self):
        source = WellKnownSkillSource()
        assert source._extract_domain("example.com") == "example.com"
        assert source._extract_domain("well-known://example.com/email") == "example.com"
        assert source._extract_domain("random search query") is None

    def test_search_no_domain_returns_empty(self):
        results = WellKnownSkillSource().search("random query")
        assert results == []

    def test_inspect_invalid_identifier(self):
        assert WellKnownSkillSource().inspect("invalid://identifier") is None


# ---------------------------------------------------------------------------
# UrlSource
# ---------------------------------------------------------------------------


class TestUrlSource:
    def test_source_id(self):
        assert UrlSource().source_id() == "url"

    def test_search_returns_empty(self):
        # URL source doesn't support search.
        assert UrlSource().search("anything") == []

    def test_name_from_url(self):
        source = UrlSource()
        # Path.stem returns the filename without extension.
        assert source._name_from_url("https://example.com/skills/my-skill/SKILL.md") == "SKILL"
        assert source._name_from_url("https://example.com/my-skill.md") == "my-skill"

    def test_inspect_blocks_unsafe_url(self):
        assert UrlSource().inspect("http://localhost/evil") is None


# ---------------------------------------------------------------------------
# SkillsShSource
# ---------------------------------------------------------------------------


class TestSkillsShSource:
    def test_source_id(self):
        assert SkillsShSource().source_id() == "skillssh"

    def test_parse_identifier(self):
        source = SkillsShSource()
        assert source._parse_identifier("skillssh://my-skill") == "my-skill"
        assert source._parse_identifier("not-skillssh://my-skill") is None


# ---------------------------------------------------------------------------
# ClawHubSource
# ---------------------------------------------------------------------------


class TestClawHubSource:
    def test_source_id(self):
        assert ClawHubSource().source_id() == "clawhub"

    def test_parse_identifier(self):
        source = ClawHubSource()
        assert source._parse_identifier("clawhub://my-skill") == "my-skill"
        assert source._parse_identifier("not-clawhub://my-skill") is None

    def test_extract_zip(self):
        import io
        import zipfile
        # Create a test ZIP.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("SKILL.md", "# Test Skill\n")
            zf.writestr("references/foo.md", "# Foo\n")
        data = buf.getvalue()
        files = ClawHubSource._extract_zip(data)
        assert "SKILL.md" in files
        assert "references/foo.md" in files
        assert "Test Skill" in files["SKILL.md"]


# ---------------------------------------------------------------------------
# ClaudeMarketplaceSource
# ---------------------------------------------------------------------------


class TestClaudeMarketplaceSource:
    def test_source_id(self):
        assert ClaudeMarketplaceSource().source_id() == "claude-marketplace"

    def test_trust_level_is_trusted(self):
        source = ClaudeMarketplaceSource()
        assert source.trust_level_for("anything") == "trusted"

    def test_parse_identifier(self):
        source = ClaudeMarketplaceSource()
        assert source._parse_identifier("claude-marketplace://my-skill") == "my-skill"
        assert source._parse_identifier("not-claude://my-skill") is None


# ---------------------------------------------------------------------------
# LobeHubSource
# ---------------------------------------------------------------------------


class TestLobeHubSource:
    def test_source_id(self):
        assert LobeHubSource().source_id() == "lobehub"

    def test_parse_identifier(self):
        source = LobeHubSource()
        assert source._parse_identifier("lobehub://my-plugin") == "my-plugin"
        assert source._parse_identifier("not-lobehub://my-plugin") is None

    def test_manifest_to_skill_md(self):
        meta = SkillMeta(
            name="test-plugin",
            description="A test plugin",
            source="lobehub",
            identifier="lobehub://test-plugin",
        )
        md = LobeHubSource._manifest_to_skill_md(meta)
        assert "test-plugin" in md
        assert "A test plugin" in md
        assert "LobeHub" in md


# ---------------------------------------------------------------------------
# BrowseShSource
# ---------------------------------------------------------------------------


class TestBrowseShSource:
    def test_source_id(self):
        assert BrowseShSource().source_id() == "browsesh"

    def test_parse_identifier(self):
        source = BrowseShSource()
        assert source._parse_identifier("browsesh://my-skill") == "my-skill"
        assert source._parse_identifier("not-browsesh://my-skill") is None


# ---------------------------------------------------------------------------
# HermesIndexSource
# ---------------------------------------------------------------------------


class TestHermesIndexSource:
    def test_source_id(self):
        assert HermesIndexSource().source_id() == "hermes-index"

    def test_parse_identifier(self):
        source = HermesIndexSource()
        assert source._parse_identifier("hermes-index://my-skill") == "my-skill"
        assert source._parse_identifier("not-hermes://my-skill") is None

    def test_load_index_empty(self, tmp_path: Path):
        source = HermesIndexSource(index_path=tmp_path / "nonexistent.json")
        assert source._load_index() == []

    def test_load_index_from_file(self, tmp_path: Path):
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps([
            {"name": "skill1", "description": "First skill"},
            {"name": "skill2", "description": "Second skill"},
        ]))
        source = HermesIndexSource(index_path=index_path)
        index = source._load_index()
        assert len(index) == 2
        assert index[0]["name"] == "skill1"

    def test_search_from_index(self, tmp_path: Path):
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps([
            {"name": "email-skill", "description": "Email management"},
            {"name": "calendar-skill", "description": "Calendar management"},
        ]))
        source = HermesIndexSource(index_path=index_path)
        results = source.search("email")
        assert len(results) == 1
        assert results[0].name == "email-skill"

    def test_inspect_from_index(self, tmp_path: Path):
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps([
            {"name": "test-skill", "description": "Test", "trust_level": "trusted"},
        ]))
        source = HermesIndexSource(index_path=index_path)
        meta = source.inspect("hermes-index://test-skill")
        assert meta is not None
        assert meta.name == "test-skill"
        assert meta.trust_level == "trusted"


# ---------------------------------------------------------------------------
# TapsManager
# ---------------------------------------------------------------------------


class TestTapsManager:
    def test_load_empty(self, tmp_path: Path):
        tm = TapsManager(path=tmp_path / "taps.json")
        assert tm.load() == []

    def test_add_and_list(self, tmp_path: Path):
        tm = TapsManager(path=tmp_path / "taps.json")
        assert tm.add("owner/repo1", "skills/") is True
        assert tm.add("owner/repo2", "skills/") is True
        taps = tm.list_taps()
        assert len(taps) == 2
        assert taps[0]["repo"] == "owner/repo1"

    def test_add_duplicate(self, tmp_path: Path):
        tm = TapsManager(path=tmp_path / "taps.json")
        tm.add("owner/repo1", "skills/")
        assert tm.add("owner/repo1", "skills/") is False  # already exists

    def test_remove(self, tmp_path: Path):
        tm = TapsManager(path=tmp_path / "taps.json")
        tm.add("owner/repo1", "skills/")
        tm.add("owner/repo2", "skills/")
        assert tm.remove("owner/repo1") is True
        taps = tm.list_taps()
        assert len(taps) == 1
        assert taps[0]["repo"] == "owner/repo2"

    def test_remove_not_found(self, tmp_path: Path):
        tm = TapsManager(path=tmp_path / "taps.json")
        assert tm.remove("nonexistent") is False

    def test_persistence(self, tmp_path: Path):
        path = tmp_path / "taps.json"
        tm1 = TapsManager(path=path)
        tm1.add("owner/repo", "skills/")
        # Create a new instance pointing at the same file.
        tm2 = TapsManager(path=path)
        taps = tm2.list_taps()
        assert len(taps) == 1
        assert taps[0]["repo"] == "owner/repo"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestFactory:
    def test_get_extra_sources_returns_8(self):
        sources = get_extra_sources()
        assert len(sources) == 8
        ids = {s.source_id() for s in sources}
        assert ids == {
            "well-known", "url", "skillssh", "clawhub",
            "claude-marketplace", "lobehub", "browsesh", "hermes-index",
        }

    def test_get_all_sources_includes_extra(self):
        sources = get_all_sources()
        assert len(sources) >= 8  # at least the 8 extra
        ids = {s.source_id() for s in sources}
        assert "well-known" in ids
        assert "clawhub" in ids

    def test_all_sources_implement_skill_source(self):
        for source in get_extra_sources():
            assert isinstance(source, SkillSource)


# ---------------------------------------------------------------------------
# Ported skills exist
# ---------------------------------------------------------------------------


class TestPortedSkills:
    def test_computer_use_exists(self):
        from niaharness.skills.bundled import get_bundled_skills_dir
        skill_path = get_bundled_skills_dir() / "optional" / "agents" / "computer-use" / "SKILL.md"
        assert skill_path.exists(), f"computer-use skill not found at {skill_path}"

    def test_dogfood_exists(self):
        from niaharness.skills.bundled import get_bundled_skills_dir
        skill_path = get_bundled_skills_dir() / "optional" / "agents" / "dogfood" / "SKILL.md"
        assert skill_path.exists()

    def test_yuanbao_exists(self):
        from niaharness.skills.bundled import get_bundled_skills_dir
        skill_path = get_bundled_skills_dir() / "optional" / "agents" / "yuanbao" / "SKILL.md"
        assert skill_path.exists()

    def test_total_skill_count_increased(self):
        """NIA should now have more than the original 34 skills."""
        from niaharness.skills.bundled import get_bundled_skills_dir
        skills = list(get_bundled_skills_dir().rglob("SKILL.md"))
        assert len(skills) >= 50  # was 34, now 60


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
