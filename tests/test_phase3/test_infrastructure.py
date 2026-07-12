"""Tests for Phase 3 infrastructure modules."""

import json
import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# projects_db
# ---------------------------------------------------------------------------


class TestProjectsDB:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "projects.db"

    def test_create_and_get(self):
        from niaharness.cli.projects_db import connect_closing, create_project, get_project
        with connect_closing(self.db_path) as conn:
            pid = create_project(conn, name="Test Project", folders=["/tmp/test1"])
            proj = get_project(conn, pid)
            assert proj is not None
            assert proj.name == "Test Project"
            assert proj.slug == "test-project"
            assert len(proj.folders) == 1
            assert proj.folders[0].path == "/tmp/test1"
            assert proj.folders[0].is_primary is True

    def test_list_projects(self):
        from niaharness.cli.projects_db import connect_closing, create_project, list_projects
        with connect_closing(self.db_path) as conn:
            create_project(conn, name="Project A")
            create_project(conn, name="Project B")
            projects = list_projects(conn)
            assert len(projects) == 2

    def test_get_by_slug(self):
        from niaharness.cli.projects_db import connect_closing, create_project, get_project
        with connect_closing(self.db_path) as conn:
            create_project(conn, name="My Cool Project")
            proj = get_project(conn, "my-cool-project")
            assert proj is not None
            assert proj.name == "My Cool Project"

    def test_update_project(self):
        from niaharness.cli.projects_db import connect_closing, create_project, update_project, get_project
        with connect_closing(self.db_path) as conn:
            pid = create_project(conn, name="Original")
            update_project(conn, pid, name="Updated", icon="🚀")
            proj = get_project(conn, pid)
            assert proj.name == "Updated"
            assert proj.icon == "🚀"

    def test_add_remove_folder(self):
        from niaharness.cli.projects_db import connect_closing, create_project, add_folder, remove_folder, get_project
        with connect_closing(self.db_path) as conn:
            pid = create_project(conn, name="Test", folders=["/tmp/a"])
            add_folder(conn, pid, "/tmp/b", is_primary=True)
            proj = get_project(conn, pid)
            assert len(proj.folders) == 2
            assert proj.primary_path == "/tmp/b"
            remove_folder(conn, pid, "/tmp/b")
            proj = get_project(conn, pid)
            assert len(proj.folders) == 1

    def test_archive_and_delete(self):
        from niaharness.cli.projects_db import connect_closing, create_project, archive_project, delete_project, get_project
        with connect_closing(self.db_path) as conn:
            pid = create_project(conn, name="Test")
            archive_project(conn, pid)
            proj = get_project(conn, pid)
            assert proj.archived is True
            delete_project(conn, pid)
            assert get_project(conn, pid) is None

    def test_find_for_cwd(self):
        from niaharness.cli.projects_db import connect_closing, create_project, find_for_cwd
        with connect_closing(self.db_path) as conn:
            create_project(conn, name="Test", folders=["/tmp/myproject"])
            proj = find_for_cwd(conn, "/tmp/myproject/subdir")
            assert proj is not None
            assert proj.name == "Test"

    def test_set_active(self):
        from niaharness.cli.projects_db import connect_closing, create_project, set_active, get_active_id
        with connect_closing(self.db_path) as conn:
            pid = create_project(conn, name="Test")
            set_active(conn, pid)
            assert get_active_id(conn) == pid

    def test_normalize_slug(self):
        from niaharness.cli.projects_db import normalize_slug
        assert normalize_slug("My-Project") == "my-project"
        assert normalize_slug(None) is None
        assert normalize_slug("") is None
        with pytest.raises(ValueError):
            normalize_slug("-invalid")


# ---------------------------------------------------------------------------
# checkpoint_manager
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    def test_disabled_by_default(self):
        from niaharness.engine.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager()
        assert mgr.enabled is False
        assert mgr.ensure_checkpoint("/tmp") is False

    def test_new_turn_resets(self):
        from niaharness.engine.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(enabled=True)
        mgr._checkpointed_dirs.add("/tmp/test")
        mgr.new_turn()
        assert len(mgr._checkpointed_dirs) == 0

    def test_list_checkpoints_empty(self):
        from niaharness.engine.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(enabled=True)
        result = mgr.list_checkpoints("/tmp")
        assert result == []

    def test_resolve_hash_fallback(self):
        from niaharness.engine.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(enabled=True)
        # No store exists — returns the input ref.
        assert mgr.resolve_hash("/tmp", "abc123") == "abc123"


# ---------------------------------------------------------------------------
# redact
# ---------------------------------------------------------------------------


class TestRedact:
    def test_mask_secret_short(self):
        from niaharness.engine.redact import mask_secret
        assert mask_secret("short") == "***"

    def test_mask_secret_long(self):
        from niaharness.engine.redact import mask_secret
        result = mask_secret("sk-proj-abcdef1234567890")
        assert result.startswith("sk-p")
        assert result.endswith("7890")
        assert "..." in result

    def test_mask_secret_empty(self):
        from niaharness.engine.redact import mask_secret
        assert mask_secret("") == ""

    def test_redact_github_token(self):
        from niaharness.engine.redact import redact_sensitive_text
        text = "My token is ghp_1234567890abcdefghijklmnop"
        result = redact_sensitive_text(text)
        assert "ghp_1234567890abcdefghijklmnop" not in result

    def test_redact_env_assignment(self):
        from niaharness.engine.redact import redact_sensitive_text
        text = "OPENAI_API_KEY=sk-proj-abcdef1234567890"
        result = redact_sensitive_text(text)
        assert "sk-proj-abcdef1234567890" not in result

    def test_redact_json_field(self):
        from niaharness.engine.redact import redact_sensitive_text
        text = '{"api_key": "sk-test-1234567890abcdef"}'
        result = redact_sensitive_text(text)
        assert "sk-test-1234567890abcdef" not in result

    def test_redact_auth_header(self):
        from niaharness.engine.redact import redact_sensitive_text
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature"
        result = redact_sensitive_text(text)
        assert "Bearer eyJhbGciOiJIUzI1NiJ9" not in result

    def test_redact_db_connstr(self):
        from niaharness.engine.redact import redact_sensitive_text
        text = "postgresql://user:secretpass@host:5432/db"
        result = redact_sensitive_text(text)
        assert "secretpass" not in result

    def test_redact_private_key(self):
        from niaharness.engine.redact import redact_sensitive_text
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
        result = redact_sensitive_text(text)
        assert "[REDACTED PRIVATE KEY]" in result

    def test_redact_normal_text_passthrough(self):
        from niaharness.engine.redact import redact_sensitive_text
        text = "Hello world, this is normal text."
        result = redact_sensitive_text(text)
        assert result == text

    def test_redact_form_body(self):
        from niaharness.engine.redact import redact_sensitive_text
        text = "token=secret123&name=bob&api_key=xyz"
        result = redact_sensitive_text(text)
        assert "secret123" not in result
        assert "xyz" not in result
        assert "bob" in result


# ---------------------------------------------------------------------------
# replay_cleanup
# ---------------------------------------------------------------------------


class TestReplayCleanup:
    def test_strip_dangling_tool_call_tail(self):
        from niaharness.engine.replay_cleanup import strip_dangling_tool_call_tail
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "tool_calls": [{"id": "1", "function": {"name": "test"}}]},
        ]
        result = strip_dangling_tool_call_tail(history)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_keep_completed_tool_call(self):
        from niaharness.engine.replay_cleanup import strip_dangling_tool_call_tail
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "result"},
        ]
        result = strip_dangling_tool_call_tail(history)
        assert len(result) == 3

    def test_strip_interrupted_tool_tails(self):
        from niaharness.engine.replay_cleanup import strip_interrupted_tool_tails
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "[command interrupted]"},
        ]
        result = strip_interrupted_tool_tails(history)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_sanitize_replay_history_empty(self):
        from niaharness.engine.replay_cleanup import sanitize_replay_history
        assert sanitize_replay_history([]) == []


# ---------------------------------------------------------------------------
# learn_prompt
# ---------------------------------------------------------------------------


class TestLearnPrompt:
    def test_build_learn_prompt_with_text(self):
        from niaharness.engine.learn_prompt import build_learn_prompt
        result = build_learn_prompt("Create a skill for deploying to Vercel")
        assert "[/learn]" in result
        assert "Create a skill for deploying to Vercel" in result
        assert "SKILL.md" in result

    def test_build_learn_prompt_empty(self):
        from niaharness.engine.learn_prompt import build_learn_prompt
        result = build_learn_prompt("")
        assert "[/learn]" in result
        assert "the workflow we just went through" in result


# ---------------------------------------------------------------------------
# goals
# ---------------------------------------------------------------------------


class TestGoals:
    def test_parse_contract(self):
        from niaharness.goals import parse_contract
        headline, contract = parse_contract("Migrate auth to JWT\nverify: tests pass\nconstraints: keep /login shape")
        assert "Migrate auth to JWT" in headline
        assert contract.verification == "tests pass"
        assert contract.constraints == "keep /login shape"

    def test_goal_contract_is_empty(self):
        from niaharness.goals import GoalContract
        assert GoalContract().is_empty() is True
        assert GoalContract(outcome="Do thing").is_empty() is False

    def test_goal_contract_render_block(self):
        from niaharness.goals import GoalContract
        c = GoalContract(outcome="Do the thing", verification="tests pass")
        block = c.render_block()
        assert "Outcome: Do the thing" in block
        assert "Verification: tests pass" in block

    def test_goal_state_to_json_roundtrip(self):
        from niaharness.goals import GoalState
        state = GoalState(goal="Test goal", max_turns=10)
        raw = state.to_json()
        restored = GoalState.from_json(raw)
        assert restored.goal == "Test goal"
        assert restored.max_turns == 10

    def test_goal_manager_status_line_no_goal(self):
        from niaharness.goals import GoalManager
        mgr = GoalManager("test-session")
        assert "No active goal" in mgr.status_line()


# ---------------------------------------------------------------------------
# verification_evidence
# ---------------------------------------------------------------------------


class TestVerificationEvidence:
    def test_status_not_applicable(self):
        from niaharness.engine.verification_evidence import verification_status
        # No project facts for a non-existent dir.
        status = verification_status(session_id="test", cwd="/nonexistent")
        assert status["status"] in ("not_applicable", "unverified")

    def test_classify_non_verification_command(self):
        from niaharness.engine.verification_evidence import classify_verification_command
        result = classify_verification_command("echo hello", cwd="/tmp", session_id="test")
        assert result is None


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------


class TestAuth:
    def test_has_usable_secret_valid(self):
        from niaharness.cli.auth import has_usable_secret
        assert has_usable_secret("sk-test-12345") is True

    def test_has_usable_secret_placeholder(self):
        from niaharness.cli.auth import has_usable_secret
        assert has_usable_secret("***") is False
        assert has_usable_secret("changeme") is False
        assert has_usable_secret("your_api_key") is False

    def test_has_usable_secret_short(self):
        from niaharness.cli.auth import has_usable_secret
        assert has_usable_secret("ab") is False

    def test_provider_registry_has_providers(self):
        from niaharness.cli.auth import PROVIDER_REGISTRY
        assert "openai-api" in PROVIDER_REGISTRY
        assert "anthropic" in PROVIDER_REGISTRY
        assert "deepseek" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["openai-api"].auth_type == "api_key"
        assert "OPENAI_API_KEY" in PROVIDER_REGISTRY["openai-api"].api_key_env_vars

    def test_clear_provider_auth_no_store(self):
        from niaharness.cli.auth import clear_provider_auth
        # No auth.json exists — should return False.
        assert clear_provider_auth("nonexistent-provider") is False


# ---------------------------------------------------------------------------
# runtime_provider
# ---------------------------------------------------------------------------


class TestRuntimeProvider:
    def test_resolve_runtime_provider_auto(self):
        from niaharness.cli.runtime_provider import resolve_runtime_provider
        rt = resolve_runtime_provider()
        assert "provider" in rt
        assert "api_key" in rt
        assert "base_url" in rt

    def test_resolve_requested_provider_auto(self):
        from niaharness.cli.runtime_provider import resolve_requested_provider
        result = resolve_requested_provider(None)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# clipboard
# ---------------------------------------------------------------------------


class TestClipboard:
    def test_has_clipboard_image_headless(self):
        from niaharness.cli.clipboard import has_clipboard_image
        # In a headless env, this should return False without crashing.
        result = has_clipboard_image()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# skill_commands
# ---------------------------------------------------------------------------


class TestSkillCommands:
    def test_scan_returns_dict(self):
        from niaharness.skills.skill_commands import scan_skill_commands
        result = scan_skill_commands()
        assert isinstance(result, dict)

    def test_resolve_skill_command_key_none(self):
        from niaharness.skills.skill_commands import resolve_skill_command_key
        assert resolve_skill_command_key("") is None
        assert resolve_skill_command_key(None) is None

    def test_extract_user_instruction_non_skill(self):
        from niaharness.skills.skill_commands import extract_user_instruction_from_skill_message
        # Non-skill content passes through unchanged.
        assert extract_user_instruction_from_skill_message("hello world") == "hello world"


# ---------------------------------------------------------------------------
# plugins
# ---------------------------------------------------------------------------


class TestPlugins:
    def test_get_plugin_commands_empty(self):
        from niaharness.plugins import get_plugin_commands
        cmds = get_plugin_commands()
        assert isinstance(cmds, dict)

    def test_get_plugin_command_handler_none(self):
        from niaharness.plugins import get_plugin_command_handler
        assert get_plugin_command_handler("nonexistent") is None

    def test_resolve_plugin_command_result_sync(self):
        from niaharness.plugins import resolve_plugin_command_result
        assert resolve_plugin_command_result("hello") == "hello"
        assert resolve_plugin_command_result(42) == 42


# ---------------------------------------------------------------------------
# attachment_paths
# ---------------------------------------------------------------------------


class TestAttachmentPaths:
    def test_image_extensions(self):
        from niaharness.cli.attachment_paths import _IMAGE_EXTENSIONS
        assert ".png" in _IMAGE_EXTENSIONS
        assert ".jpg" in _IMAGE_EXTENSIONS
        assert ".txt" not in _IMAGE_EXTENSIONS

    def test_split_path_input(self):
        from niaharness.cli.attachment_paths import _split_path_input
        token, remainder = _split_path_input("/tmp/test.png describe this")
        assert token == "/tmp/test.png"
        assert remainder == "describe this"

    def test_split_path_input_quoted(self):
        from niaharness.cli.attachment_paths import _split_path_input
        token, remainder = _split_path_input('"/tmp/my file.png" describe')
        assert token == "/tmp/my file.png"
        assert remainder == "describe"

    def test_split_path_input_empty(self):
        from niaharness.cli.attachment_paths import _split_path_input
        assert _split_path_input("") == ("", "")

    def test_resolve_attachment_path_nonexistent(self):
        from niaharness.cli.attachment_paths import _resolve_attachment_path
        assert _resolve_attachment_path("/nonexistent/file.png") is None

    def test_resolve_attachment_path_empty(self):
        from niaharness.cli.attachment_paths import _resolve_attachment_path
        assert _resolve_attachment_path("") is None

    def test_detect_file_drop_non_path(self):
        from niaharness.cli.attachment_paths import _detect_file_drop
        assert _detect_file_drop("hello world") is None
        assert _detect_file_drop("/goal do something") is None  # not a real file

    def test_detect_file_drop_empty(self):
        from niaharness.cli.attachment_paths import _detect_file_drop
        assert _detect_file_drop("") is None
        assert _detect_file_drop(None) is None


# ---------------------------------------------------------------------------
# voice_mode
# ---------------------------------------------------------------------------


class TestVoiceMode:
    def test_check_voice_requirements(self):
        from niaharness.voice_mode import check_voice_requirements
        result = check_voice_requirements()
        assert "available" in result
        assert "audio_available" in result
        assert "stt_available" in result
        assert "details" in result

    def test_detect_audio_environment(self):
        from niaharness.voice_mode import detect_audio_environment
        result = detect_audio_environment()
        assert "available" in result
        assert "warnings" in result


# ---------------------------------------------------------------------------
# preview_restart
# ---------------------------------------------------------------------------


class TestPreviewRestart:
    def test_preview_restart_history(self):
        from niaharness.engine.preview_restart import preview_restart_history
        session = {"history": [{"role": "user", "text": "hello"}, {"role": "assistant", "text": "hi"}]}
        result = preview_restart_history(session)
        assert len(result) == 2
        assert result[0]["role"] == "user"

    def test_preview_restart_history_empty(self):
        from niaharness.engine.preview_restart import preview_restart_history
        assert preview_restart_history({}) == []

    def test_ephemeral_preview_agent_kwargs(self):
        from niaharness.engine.preview_restart import ephemeral_preview_agent_kwargs
        kwargs = ephemeral_preview_agent_kwargs(None, "preview_abc")
        assert kwargs["task_id"] == "preview_abc"
        assert kwargs["disable_memory"] is True
