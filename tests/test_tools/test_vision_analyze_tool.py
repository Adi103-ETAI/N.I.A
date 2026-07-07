"""Tests for the vision_analyze tool."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from niaharness.tools.base import ToolExecutionContext
from niaharness.tools.vision_analyze_tool import (
    VisionAnalyzeTool,
    VisionAnalyzeToolInput,
    _detect_mime_type,
    _load_image_from_file,
    _resolve_vision_config,
)


@pytest.fixture
def context(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path)


# ---------------------------------------------------------------------------
# MIME type detection
# ---------------------------------------------------------------------------


class TestDetectMimeType:
    def test_png_magic_bytes(self):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        assert _detect_mime_type("file.bin", png_bytes) == "image/png"

    def test_jpeg_magic_bytes(self):
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        assert _detect_mime_type("file.bin", jpeg_bytes) == "image/jpeg"

    def test_gif_magic_bytes(self):
        gif_bytes = b"GIF89a" + b"\x00" * 100
        assert _detect_mime_type("file.bin", gif_bytes) == "image/gif"

    def test_webp_magic_bytes(self):
        webp_bytes = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100
        assert _detect_mime_type("file.bin", webp_bytes) == "image/webp"

    def test_by_extension(self):
        assert _detect_mime_type("photo.jpg", None) == "image/jpeg"
        assert _detect_mime_type("photo.png", None) == "image/png"
        assert _detect_mime_type("photo.gif", None) == "image/gif"
        assert _detect_mime_type("photo.webp", None) == "image/webp"

    def test_unsupported_returns_none(self):
        assert _detect_mime_type("file.txt", b"hello world") is None

    def test_empty_content(self):
        assert _detect_mime_type("file.bin", b"") is None


# ---------------------------------------------------------------------------
# Local file loading
# ---------------------------------------------------------------------------


class TestLoadImageFromFile:
    def test_load_png(self, tmp_path: Path):
        path = tmp_path / "test.png"
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        path.write_bytes(png_bytes)

        content, mime = _load_image_from_file(path)
        assert content == png_bytes
        assert mime == "image/png"

    def test_load_jpeg_by_extension(self, tmp_path: Path):
        path = tmp_path / "photo.jpg"
        path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

        content, mime = _load_image_from_file(path)
        assert mime == "image/jpeg"

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            _load_image_from_file(tmp_path / "missing.png")

    def test_directory_not_file(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Not a file"):
            _load_image_from_file(tmp_path)

    def test_unsupported_type(self, tmp_path: Path):
        path = tmp_path / "data.txt"
        path.write_bytes(b"just text, not an image")
        with pytest.raises(ValueError, match="Could not determine image type"):
            _load_image_from_file(path)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


class TestResolveVisionConfig:
    def test_dedicated_vision_env_vars(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NIA_VISION_API_KEY", "vision-key-123")
        monkeypatch.setenv("NIA_VISION_BASE_URL", "https://vision.example.com/v1")
        monkeypatch.setenv("NIA_VISION_MODEL", "gpt-4o-mini")
        config = _resolve_vision_config()
        assert config["api_key"] == "vision-key-123"
        assert config["base_url"] == "https://vision.example.com/v1"
        assert config["model"] == "gpt-4o-mini"

    def test_fallback_to_openai_key(self, monkeypatch: pytest.MonkeyPatch):
        """When no NIA_VISION_* and no settings api_key, falls back to OPENAI_API_KEY."""
        monkeypatch.delenv("NIA_VISION_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key-456")
        # Mock settings to not find a key (forces fallback to OPENAI_API_KEY).
        with patch("niaharness.config.settings.load_settings") as mock_load:
            mock_settings = MagicMock()
            mock_settings.resolve_api_key.side_effect = ValueError("no key")
            mock_settings.base_url = None
            mock_settings.model = "ignored"
            mock_load.return_value = mock_settings
            config = _resolve_vision_config()
            assert config["api_key"] == "openai-key-456"
            assert config["model"] == "gpt-4o"  # default vision model

    def test_no_key_returns_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NIA_VISION_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Also need to mock load_settings to not find a key
        with patch("niaharness.config.settings.load_settings") as mock_load:
            mock_settings = MagicMock()
            mock_settings.resolve_api_key.side_effect = ValueError("no key")
            mock_settings.base_url = None
            mock_settings.model = "test"
            mock_load.return_value = mock_settings
            config = _resolve_vision_config()
            assert config["api_key"] == ""


# ---------------------------------------------------------------------------
# Tool — error paths (no real API calls)
# ---------------------------------------------------------------------------


class TestToolErrorPaths:
    @pytest.mark.asyncio
    async def test_empty_source(self, context: ToolExecutionContext):
        result = await VisionAnalyzeTool().execute(
            VisionAnalyzeToolInput(image_source=""),
            context,
        )
        assert result.is_error is True
        assert "required" in result.output.lower()

    @pytest.mark.asyncio
    async def test_file_not_found(self, context: ToolExecutionContext):
        result = await VisionAnalyzeTool().execute(
            VisionAnalyzeToolInput(image_source="/nonexistent/image.png"),
            context,
        )
        assert result.is_error is True
        assert "not found" in result.output.lower()

    @pytest.mark.asyncio
    async def test_unsupported_file_type(self, tmp_path: Path, context: ToolExecutionContext):
        path = tmp_path / "data.txt"
        path.write_bytes(b"not an image")
        result = await VisionAnalyzeTool().execute(
            VisionAnalyzeToolInput(image_source=str(path)),
            context,
        )
        assert result.is_error is True
        assert "image" in result.output.lower() or "type" in result.output.lower()

    @pytest.mark.asyncio
    async def test_no_api_key(
        self, tmp_path: Path, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch
    ):
        # Create a valid PNG.
        path = tmp_path / "test.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        # Ensure no API keys are available.
        monkeypatch.delenv("NIA_VISION_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

        with patch("niaharness.config.settings.load_settings") as mock_load:
            mock_settings = MagicMock()
            mock_settings.resolve_api_key.side_effect = ValueError("no key")
            mock_settings.base_url = None
            mock_settings.model = "test"
            mock_load.return_value = mock_settings

            result = await VisionAnalyzeTool().execute(
                VisionAnalyzeToolInput(image_source=str(path)),
                context,
            )
            assert result.is_error is True
            assert "API key" in result.output

    @pytest.mark.asyncio
    async def test_relative_path_resolved(
        self, tmp_path: Path, context: ToolExecutionContext
    ):
        # Create image in the cwd.
        path = tmp_path / "relative.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        # Use a relative path — should resolve against context.cwd.
        with patch("niaharness.tools.vision_analyze_tool._resolve_vision_config") as mock_cfg:
            mock_cfg.return_value = {"api_key": "", "base_url": None, "model": "test"}
            result = await VisionAnalyzeTool().execute(
                VisionAnalyzeToolInput(image_source="relative.png"),
                context,
            )
            # Should get past file loading and fail at the API key check,
            # not at file-not-found.
            assert "API key" in result.output or "no key" in result.output.lower()


# ---------------------------------------------------------------------------
# Tool — mocked LLM call (happy path)
# ---------------------------------------------------------------------------


class TestToolMockedVisionCall:
    @pytest.mark.asyncio
    async def test_successful_analysis(
        self, tmp_path: Path, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch
    ):
        # Create a valid PNG.
        path = tmp_path / "photo.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        monkeypatch.setenv("NIA_VISION_API_KEY", "test-key")
        monkeypatch.setenv("NIA_VISION_MODEL", "gpt-4o")

        # Mock the OpenAI client's chat.completions.create.
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="A photo of a cat sitting on a windowsill.")
            )
        ]

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = await VisionAnalyzeTool().execute(
                VisionAnalyzeToolInput(
                    image_source=str(path),
                    prompt="What animal is in this image?",
                ),
                context,
            )

        assert result.is_error is False
        assert "A photo of a cat" in result.output
        assert "image/png" in result.output
        assert "gpt-4o" in result.output
        assert result.metadata["analysis_length"] > 0
        assert result.metadata["mime_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_llm_call_failure(
        self, tmp_path: Path, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch
    ):
        path = tmp_path / "photo.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        monkeypatch.setenv("NIA_VISION_API_KEY", "test-key")

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API rate limit exceeded")
            )
            mock_openai.return_value = mock_client

            result = await VisionAnalyzeTool().execute(
                VisionAnalyzeToolInput(image_source=str(path)),
                context,
            )

        assert result.is_error is True
        assert "Vision model call failed" in result.output
        assert "rate limit" in result.output.lower()

    @pytest.mark.asyncio
    async def test_empty_response(
        self, tmp_path: Path, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch
    ):
        path = tmp_path / "photo.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        monkeypatch.setenv("NIA_VISION_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.choices = []  # empty

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await VisionAnalyzeTool().execute(
                VisionAnalyzeToolInput(image_source=str(path)),
                context,
            )

        assert result.is_error is False
        assert "no response" in result.output.lower()


# ---------------------------------------------------------------------------
# Read-only flag
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_is_read_only(self):
        tool = VisionAnalyzeTool()
        assert tool.is_read_only(VisionAnalyzeToolInput(image_source="x.png")) is True
