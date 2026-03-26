"""Unit tests for namespace memory and sandbox result models.

Tests NamespaceManager creation (with mocked ChromaDB) and SandboxResult
Pydantic model field access.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.infrastructure.container_engine.idempotency import SandboxResult


# ---------------------------------------------------------------------------
# NamespaceManager creation (mocked deps)
# ---------------------------------------------------------------------------


class TestNamespaceManager:
    """Tests for NamespaceManager instantiation with mocked dependencies."""

    def test_namespace_manager_creation(self) -> None:
        """NamespaceManager must instantiate without error when deps are mocked."""
        # We mock the heavy dependencies (ChromaDB client, embedding function,
        # MemoryManager) so the test does not start real services.
        mock_memory_manager = MagicMock()
        mock_memory_manager._chroma_client = MagicMock()
        # get_or_create_collection returns a mock collection
        mock_memory_manager._chroma_client.get_or_create_collection = MagicMock(
            return_value=MagicMock()
        )

        with patch(
            "src.core.memory.namespaces.get_embedding_function",
            return_value=None,
        ):
            from src.core.memory.namespaces import NamespaceManager

            ns = NamespaceManager(memory_manager=mock_memory_manager)

            assert ns is not None
            assert ns._client is mock_memory_manager._chroma_client


# ---------------------------------------------------------------------------
# SandboxResult model
# ---------------------------------------------------------------------------


class TestSandboxResult:
    """Tests for the SandboxResult Pydantic model."""

    def test_sandbox_result_model(self) -> None:
        """All fields must be accessible after creation."""
        result = SandboxResult(
            exit_code=0,
            output="Hello, world!",
            idempotency_key="key-001",
            manifest_id="mission-007",
            cached=False,
            executed_at="2026-03-15T12:00:00+00:00",
        )

        assert result.exit_code == 0
        assert result.output == "Hello, world!"
        assert result.idempotency_key == "key-001"
        assert result.manifest_id == "mission-007"
        assert result.cached is False
        assert result.executed_at == "2026-03-15T12:00:00+00:00"

    def test_sandbox_result_cached_flag(self) -> None:
        """cached=True must indicate a checkpoint-served result."""
        result = SandboxResult(
            exit_code=0,
            output="cached output",
            idempotency_key="key-002",
            manifest_id="mission-008",
            cached=True,
            executed_at="2026-03-15T11:00:00+00:00",
        )

        assert result.cached is True

    def test_sandbox_result_nonzero_exit_code(self) -> None:
        """Non-zero exit codes must be stored faithfully."""
        result = SandboxResult(
            exit_code=1,
            output="Error: command not found",
            idempotency_key="key-003",
            manifest_id="mission-009",
            cached=False,
            executed_at="2026-03-15T13:00:00+00:00",
        )

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_sandbox_result_model_dump(self) -> None:
        """model_dump must produce a serialisable dict."""
        result = SandboxResult(
            exit_code=0,
            output="ok",
            idempotency_key="key-004",
            manifest_id="mission-010",
            cached=False,
            executed_at="2026-03-15T14:00:00+00:00",
        )

        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["exit_code"] == 0
        assert data["cached"] is False
