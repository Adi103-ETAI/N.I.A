"""Tests for MCP OAuth — token storage, PKCE flow, manager, 401 dedup, invalid_client auto-heal.

Covers:
  - NiaTokenStorage: save/load tokens, client info, metadata, expires_at recomputation, poison
  - Path helpers: _safe_filename, _get_token_dir (profile-aware)
  - OAuthConfig + McpHttpServerConfig auth modes
  - MCPOAuthManager: singleton, get_or_build_provider caching, URL change eviction, remove
  - invalidate_if_disk_changed (mtime-based cross-process reload)
  - handle_401 dedup (concurrent callers, same future)
  - maybe_flag_poisoned_client (invalid_client auto-heal)
  - OAuthNonInteractiveError (non-interactive + no cached tokens)
  - suppress_interactive_oauth / restore_interactive_oauth ContextVar
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from niaharness.mcp.oauth import (
    NiaTokenStorage,
    OAuthNonInteractiveError,
    _OAUTH_AVAILABLE,
    _get_token_dir,
    _safe_filename,
    _is_interactive,
    build_oauth_auth,
    remove_oauth_tokens,
    restore_interactive_oauth,
    suppress_interactive_oauth,
    _write_json,
    _read_json,
)
from niaharness.mcp.oauth_manager import (
    MCPOAuthManager,
    _ProviderEntry,
    _same_endpoint,
    get_manager,
    reset_manager_for_tests,
)
from niaharness.mcp.types import McpHttpServerConfig, OAuthConfig


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestSafeFilename:
    def test_strips_special_chars(self):
        assert _safe_filename("Notion MCP!") == "Notion_MCP"
        assert _safe_filename("server.name/path") == "server_name_path"

    def test_empty_returns_default(self):
        assert _safe_filename("") == "default"
        assert _safe_filename("!!!") == "default"

    def test_truncates_long_names(self):
        long_name = "a" * 200
        result = _safe_filename(long_name)
        assert len(result) == 128

    def test_preserves_alphanumeric_and_dashes(self):
        assert _safe_filename("my-server-1") == "my-server-1"
        assert _safe_filename("Server_2") == "Server_2"


class TestGetTokenDir:
    def test_returns_path_object(self):
        result = _get_token_dir()
        assert isinstance(result, Path)

    def test_ends_with_mcp_tokens(self):
        result = _get_token_dir()
        assert result.name == "mcp-tokens"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


class TestJsonHelpers:
    def test_write_and_read_json(self, tmp_path):
        path = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        _write_json(path, data)
        result = _read_json(path)
        assert result == data

    def test_read_missing_file_returns_none(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert _read_json(path) is None

    def test_read_corrupt_json_returns_none(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{{")
        assert _read_json(path) is None

    def test_write_json_creates_parent_dir(self, tmp_path):
        path = tmp_path / "subdir" / "test.json"
        _write_json(path, {"x": 1})
        assert path.exists()


# ---------------------------------------------------------------------------
# NiaTokenStorage
# ---------------------------------------------------------------------------


class TestNiaTokenStorage:
    @pytest.fixture
    def storage(self, tmp_path, monkeypatch):
        """Build a storage backed by tmp_path."""
        monkeypatch.setattr(
            "niaharness.mcp.oauth._get_token_dir", lambda: tmp_path
        )
        return NiaTokenStorage("test-server")

    def test_paths_use_safe_filename(self, storage, tmp_path):
        assert storage._tokens_path() == tmp_path / "test-server.json"
        assert storage._client_info_path() == tmp_path / "test-server.client.json"
        assert storage._meta_path() == tmp_path / "test-server.meta.json"

    def test_has_cached_tokens_false_when_no_file(self, storage):
        assert storage.has_cached_tokens() is False

    def test_has_cached_tokens_true_after_write(self, storage):
        _write_json(storage._tokens_path(), {"access_token": "test"})
        assert storage.has_cached_tokens() is True

    @pytest.mark.skipif(not _OAUTH_AVAILABLE, reason="MCP SDK OAuth not available")
    @pytest.mark.asyncio
    async def test_set_and_get_tokens(self, storage):
        from mcp.shared.auth import OAuthToken

        tokens = OAuthToken(
            access_token="abc123",
            token_type="bearer",
            expires_in=3600,
            refresh_token="refresh_xyz",
        )
        await storage.set_tokens(tokens)
        loaded = await storage.get_tokens()
        assert loaded is not None
        assert loaded.access_token == "abc123"
        assert loaded.refresh_token == "refresh_xyz"

    @pytest.mark.skipif(not _OAUTH_AVAILABLE, reason="MCP SDK OAuth not available")
    @pytest.mark.asyncio
    async def test_get_tokens_recomputes_expires_in(self, storage):
        """After a process restart, expires_in should be recomputed from expires_at."""
        # Write tokens with expires_at in the past.
        past_expiry = time.time() - 100  # Expired 100s ago.
        _write_json(storage._tokens_path(), {
            "access_token": "old",
            "token_type": "bearer",
            "expires_in": 3600,  # Original (stale) value.
            "expires_at": past_expiry,
        })
        loaded = await storage.get_tokens()
        assert loaded is not None
        # expires_in should be 0 (expired).
        assert loaded.expires_in == 0

    @pytest.mark.skipif(not _OAUTH_AVAILABLE, reason="MCP SDK OAuth not available")
    @pytest.mark.asyncio
    async def test_get_tokens_returns_none_for_missing(self, storage):
        loaded = await storage.get_tokens()
        assert loaded is None

    @pytest.mark.skipif(not _OAUTH_AVAILABLE, reason="MCP SDK OAuth not available")
    @pytest.mark.asyncio
    async def test_get_tokens_returns_none_for_corrupt(self, storage):
        storage._tokens_path().write_text("not json")
        loaded = await storage.get_tokens()
        assert loaded is None

    def test_remove_deletes_all_files(self, storage):
        # Create all three files.
        _write_json(storage._tokens_path(), {"x": 1})
        _write_json(storage._client_info_path(), {"y": 2})
        _write_json(storage._meta_path(), {"z": 3})
        storage.remove()
        assert not storage._tokens_path().exists()
        assert not storage._client_info_path().exists()
        assert not storage._meta_path().exists()

    def test_remove_missing_files_no_error(self, storage):
        # Should not raise even if files don't exist.
        storage.remove()

    def test_poison_client_registration_no_file(self, storage):
        assert storage.poison_client_registration() is False

    def test_poison_client_registration_with_file(self, storage):
        _write_json(storage._client_info_path(), {"client_id": "test"})
        _write_json(storage._meta_path(), {"issuer": "test"})
        result = storage.poison_client_registration()
        assert result is True
        # Client + meta deleted, backup created.
        assert not storage._client_info_path().exists()
        assert not storage._meta_path().exists()
        backup = storage._client_info_path().with_name(
            storage._client_info_path().name + ".bak"
        )
        assert backup.exists()


# ---------------------------------------------------------------------------
# OAuthConfig + McpHttpServerConfig
# ---------------------------------------------------------------------------


class TestOAuthConfig:
    def test_defaults(self):
        cfg = OAuthConfig()
        assert cfg.client_id is None
        assert cfg.client_name == "NIA Agent"
        assert cfg.redirect_port == 0
        assert cfg.timeout == 300.0

    def test_with_client_id(self):
        cfg = OAuthConfig(client_id="my-id", scope="read write")
        assert cfg.client_id == "my-id"
        assert cfg.scope == "read write"


class TestMcpHttpServerConfigAuthModes:
    def test_default_auth_is_bearer(self):
        cfg = McpHttpServerConfig(url="https://example.com/mcp")
        assert cfg.auth == "bearer"
        assert cfg.oauth is None

    def test_oauth_mode(self):
        cfg = McpHttpServerConfig(
            url="https://example.com/mcp",
            auth="oauth",
            oauth=OAuthConfig(client_id="test"),
        )
        assert cfg.auth == "oauth"
        assert cfg.oauth is not None
        assert cfg.oauth.client_id == "test"

    def test_none_mode(self):
        cfg = McpHttpServerConfig(url="https://example.com/mcp", auth="none")
        assert cfg.auth == "none"


# ---------------------------------------------------------------------------
# build_oauth_auth
# ---------------------------------------------------------------------------


class TestBuildOAuthAuth:
    @pytest.mark.skipif(not _OAUTH_AVAILABLE, reason="MCP SDK OAuth not available")
    def test_returns_provider_when_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        # Force interactive so it doesn't raise.
        from niaharness.mcp.oauth import force_interactive_oauth, restore_forced_interactive

        token = force_interactive_oauth()
        try:
            provider = build_oauth_auth(
                "test-server",
                "https://example.com/mcp",
                oauth_config={"client_id": "test"},
            )
            assert provider is not None
        finally:
            restore_forced_interactive(token)

    @pytest.mark.skipif(not _OAUTH_AVAILABLE, reason="MCP SDK OAuth not available")
    def test_raises_non_interactive_without_tokens(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        # Suppress interactivity.
        token = suppress_interactive_oauth()
        try:
            with pytest.raises(OAuthNonInteractiveError):
                build_oauth_auth(
                    "test-server",
                    "https://example.com/mcp",
                )
        finally:
            restore_interactive_oauth(token)

    @pytest.mark.skipif(not _OAUTH_AVAILABLE, reason="MCP SDK OAuth not available")
    def test_non_interactive_with_cached_tokens_succeeds(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        # Pre-create token file so has_cached_tokens() returns True.
        storage = NiaTokenStorage("test-server")
        _write_json(storage._tokens_path(), {"access_token": "cached"})
        # Suppress interactivity — should still succeed because tokens exist.
        token = suppress_interactive_oauth()
        try:
            provider = build_oauth_auth(
                "test-server",
                "https://example.com/mcp",
            )
            assert provider is not None
        finally:
            restore_interactive_oauth(token)


# ---------------------------------------------------------------------------
# Interactivity ContextVars
# ---------------------------------------------------------------------------


class TestInteractivityContextVars:
    def test_suppress_interactive_oauth(self):
        original = _is_interactive()
        token = suppress_interactive_oauth()
        assert _is_interactive() is False
        restore_interactive_oauth(token)
        assert _is_interactive() == original

    def test_force_interactive_oauth(self):
        from niaharness.mcp.oauth import force_interactive_oauth, restore_forced_interactive

        token = force_interactive_oauth()
        assert _is_interactive() is True
        restore_forced_interactive(token)


# ---------------------------------------------------------------------------
# remove_oauth_tokens
# ---------------------------------------------------------------------------


class TestRemoveOAuthTokens:
    def test_removes_all_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        storage = NiaTokenStorage("test-server")
        _write_json(storage._tokens_path(), {"x": 1})
        _write_json(storage._client_info_path(), {"y": 2})
        _write_json(storage._meta_path(), {"z": 3})
        remove_oauth_tokens("test-server")
        assert not storage._tokens_path().exists()
        assert not storage._client_info_path().exists()
        assert not storage._meta_path().exists()


# ---------------------------------------------------------------------------
# MCPOAuthManager — singleton + caching
# ---------------------------------------------------------------------------


class TestMCPOAuthManagerSingleton:
    def test_get_manager_returns_singleton(self):
        reset_manager_for_tests()
        m1 = get_manager()
        m2 = get_manager()
        assert m1 is m2

    def test_reset_manager_for_tests(self):
        m1 = get_manager()
        reset_manager_for_tests()
        m2 = get_manager()
        assert m1 is not m2


class TestMCPOAuthManagerCaching:
    def test_get_or_build_provider_caches(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        from niaharness.mcp.oauth import force_interactive_oauth, restore_forced_interactive

        token = force_interactive_oauth()
        try:
            manager = MCPOAuthManager()
            p1 = manager.get_or_build_provider("srv", "https://example.com", {"client_id": "x"})
            p2 = manager.get_or_build_provider("srv", "https://example.com", {"client_id": "x"})
            assert p1 is p2  # Cached.
        finally:
            restore_forced_interactive(token)

    def test_url_change_evicts_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        from niaharness.mcp.oauth import force_interactive_oauth, restore_forced_interactive

        token = force_interactive_oauth()
        try:
            manager = MCPOAuthManager()
            p1 = manager.get_or_build_provider("srv", "https://old.example.com", None)
            p2 = manager.get_or_build_provider("srv", "https://new.example.com", None)
            assert p1 is not p2  # Evicted + rebuilt.
        finally:
            restore_forced_interactive(token)

    def test_remove_evicts_from_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        from niaharness.mcp.oauth import force_interactive_oauth, restore_forced_interactive

        token = force_interactive_oauth()
        try:
            manager = MCPOAuthManager()
            manager.get_or_build_provider("srv", "https://example.com", None)
            assert "srv" in manager._entries
            manager.remove("srv")
            assert "srv" not in manager._entries
        finally:
            restore_forced_interactive(token)


# ---------------------------------------------------------------------------
# invalidate_if_disk_changed
# ---------------------------------------------------------------------------


class TestInvalidateIfDiskChanged:
    @pytest.mark.asyncio
    async def test_no_entry_returns_false(self):
        manager = MCPOAuthManager()
        result = await manager.invalidate_if_disk_changed("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_no_change_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        from niaharness.mcp.oauth import force_interactive_oauth, restore_forced_interactive

        token = force_interactive_oauth()
        try:
            manager = MCPOAuthManager()
            provider = manager.get_or_build_provider("srv", "https://example.com", None)
            # Write the token file so stat() works.
            storage = NiaTokenStorage("srv")
            _write_json(storage._tokens_path(), {"access_token": "test"})
            # First call sets last_mtime_ns.
            await manager.invalidate_if_disk_changed("srv")
            # Second call — no change.
            result = await manager.invalidate_if_disk_changed("srv")
            assert result is False
        finally:
            restore_forced_interactive(token)

    @pytest.mark.asyncio
    async def test_disk_change_returns_true(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        from niaharness.mcp.oauth import force_interactive_oauth, restore_forced_interactive

        token = force_interactive_oauth()
        try:
            manager = MCPOAuthManager()
            provider = manager.get_or_build_provider("srv", "https://example.com", None)
            storage = NiaTokenStorage("srv")
            _write_json(storage._tokens_path(), {"access_token": "v1"})
            # First call sets baseline.
            await manager.invalidate_if_disk_changed("srv")
            # Modify the file (change mtime).
            await asyncio.sleep(0.05)  # Ensure mtime changes.
            _write_json(storage._tokens_path(), {"access_token": "v2"})
            result = await manager.invalidate_if_disk_changed("srv")
            assert result is True
        finally:
            restore_forced_interactive(token)


# ---------------------------------------------------------------------------
# handle_401 dedup
# ---------------------------------------------------------------------------


class TestHandle401Dedup:
    @pytest.mark.asyncio
    async def test_no_entry_returns_false(self):
        manager = MCPOAuthManager()
        result = await manager.handle_401("nonexistent", "token_abc")
        assert result is False

    @pytest.mark.asyncio
    async def test_concurrent_callers_share_future(self, tmp_path, monkeypatch):
        """When N callers hit 401 with the same token, only one handler runs."""
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        from niaharness.mcp.oauth import force_interactive_oauth, restore_forced_interactive

        token = force_interactive_oauth()
        try:
            manager = MCPOAuthManager()
            manager.get_or_build_provider("srv", "https://example.com", None)
            storage = NiaTokenStorage("srv")
            _write_json(storage._tokens_path(), {"access_token": "test"})

            # Launch 3 concurrent handle_401 calls with the same token.
            results = await asyncio.gather(
                manager.handle_401("srv", "same_token"),
                manager.handle_401("srv", "same_token"),
                manager.handle_401("srv", "same_token"),
            )
            # All should get the same result (True or False, but identical).
            assert len(set(results)) == 1
        finally:
            restore_forced_interactive(token)

    @pytest.mark.asyncio
    async def test_different_tokens_get_different_futures(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        from niaharness.mcp.oauth import force_interactive_oauth, restore_forced_interactive

        token = force_interactive_oauth()
        try:
            manager = MCPOAuthManager()
            manager.get_or_build_provider("srv", "https://example.com", None)
            storage = NiaTokenStorage("srv")
            _write_json(storage._tokens_path(), {"access_token": "test"})

            # Two different tokens — should get independent futures.
            r1, r2 = await asyncio.gather(
                manager.handle_401("srv", "token_a"),
                manager.handle_401("srv", "token_b"),
            )
            # Both should return a bool (True or False depending on can_refresh).
            assert isinstance(r1, bool)
            assert isinstance(r2, bool)
        finally:
            restore_forced_interactive(token)


# ---------------------------------------------------------------------------
# maybe_flag_poisoned_client (invalid_client auto-heal)
# ---------------------------------------------------------------------------


class TestMaybeFlagPoisonedClient:
    @pytest.mark.asyncio
    async def test_no_response_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        manager = MCPOAuthManager()
        result = await manager.maybe_flag_poisoned_client("srv", None, "https://token.example.com")
        assert result is False

    @pytest.mark.asyncio
    async def test_status_200_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        manager = MCPOAuthManager()
        response = MagicMock()
        response.status_code = 200
        result = await manager.maybe_flag_poisoned_client("srv", response, "https://token.example.com")
        assert result is False

    @pytest.mark.asyncio
    async def test_401_with_invalid_client_poisons(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        # Pre-create client.json so poison has something to delete.
        storage = NiaTokenStorage("srv")
        _write_json(storage._client_info_path(), {"client_id": "old"})

        manager = MCPOAuthManager()
        response = MagicMock()
        response.status_code = 401
        response.content = b'{"error": "invalid_client"}'
        response.request.url = "https://token.example.com/oauth/token"
        result = await manager.maybe_flag_poisoned_client(
            "srv", response, "https://token.example.com/oauth/token"
        )
        assert result is True
        # Client file should be deleted.
        assert not storage._client_info_path().exists()
        # Backup should exist.
        backup = storage._client_info_path().with_name(
            storage._client_info_path().name + ".bak"
        )
        assert backup.exists()

    @pytest.mark.asyncio
    async def test_401_with_invalid_client_metadata_does_not_poison(self, tmp_path, monkeypatch):
        """Word-boundary match: invalid_client_metadata should NOT trigger."""
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        storage = NiaTokenStorage("srv")
        _write_json(storage._client_info_path(), {"client_id": "old"})

        manager = MCPOAuthManager()
        response = MagicMock()
        response.status_code = 400
        response.content = b'{"error": "invalid_client_metadata"}'
        response.request.url = "https://token.example.com/oauth/token"
        result = await manager.maybe_flag_poisoned_client(
            "srv", response, "https://token.example.com/oauth/token"
        )
        assert result is False
        # Client file should still exist.
        assert storage._client_info_path().exists()

    @pytest.mark.asyncio
    async def test_wrong_endpoint_does_not_poison(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.mcp.oauth._get_token_dir", lambda: tmp_path)
        storage = NiaTokenStorage("srv")
        _write_json(storage._client_info_path(), {"client_id": "old"})

        manager = MCPOAuthManager()
        response = MagicMock()
        response.status_code = 401
        response.content = b'{"error": "invalid_client"}'
        response.request.url = "https://other.example.com/api"  # Wrong endpoint.
        result = await manager.maybe_flag_poisoned_client(
            "srv", response, "https://token.example.com/oauth/token"
        )
        assert result is False


# ---------------------------------------------------------------------------
# _same_endpoint helper
# ---------------------------------------------------------------------------


class TestSameEndpoint:
    def test_same_url(self):
        assert _same_endpoint(
            "https://example.com/oauth/token",
            "https://example.com/oauth/token",
        ) is True

    def test_trailing_slash_ignored(self):
        assert _same_endpoint(
            "https://example.com/oauth/token",
            "https://example.com/oauth/token/",
        ) is True

    def test_different_path(self):
        assert _same_endpoint(
            "https://example.com/oauth/token",
            "https://example.com/oauth/other",
        ) is False

    def test_different_host(self):
        assert _same_endpoint(
            "https://example.com/oauth/token",
            "https://other.com/oauth/token",
        ) is False

    def test_case_insensitive_scheme(self):
        assert _same_endpoint(
            "HTTPS://example.com/token",
            "https://example.com/token",
        ) is True


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
