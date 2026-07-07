"""Tests for the image_generate tool."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from niaharness.tools.base import ToolExecutionContext
from niaharness.tools.image_generate_tool import (
    ImageGenerateTool,
    ImageGenerateInput,
    _resolve_image_config,
)


@pytest.fixture
def context(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


class TestConfigResolution:
    def test_dedicated_env_vars(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NIA_IMAGE_API_KEY", "image-key")
        monkeypatch.setenv("NIA_IMAGE_BASE_URL", "https://images.example.com/v1")
        monkeypatch.setenv("NIA_IMAGE_MODEL", "flux-pro")
        config = _resolve_image_config()
        assert config["api_key"] == "image-key"
        assert config["base_url"] == "https://images.example.com/v1"
        assert config["model"] == "flux-pro"

    def test_fallback_to_openai_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NIA_IMAGE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        with patch("niaharness.config.settings.load_settings") as mock_load:
            mock_settings = MagicMock()
            mock_settings.resolve_api_key.side_effect = ValueError("no key")
            mock_settings.base_url = None
            mock_load.return_value = mock_settings
            config = _resolve_image_config()
            assert config["api_key"] == "openai-key"
            assert config["model"] == "dall-e-3"

    def test_no_key_returns_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NIA_IMAGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("niaharness.config.settings.load_settings") as mock_load:
            mock_settings = MagicMock()
            mock_settings.resolve_api_key.side_effect = ValueError("no key")
            mock_settings.base_url = None
            mock_load.return_value = mock_settings
            config = _resolve_image_config()
            assert config["api_key"] == ""


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_empty_prompt(self, context: ToolExecutionContext):
        result = await ImageGenerateTool().execute(
            ImageGenerateInput(prompt=""),
            context,
        )
        assert result.is_error is True
        assert "required" in result.output.lower()

    @pytest.mark.asyncio
    async def test_prompt_too_long(self, context: ToolExecutionContext):
        result = await ImageGenerateTool().execute(
            ImageGenerateInput(prompt="x" * 5000),
            context,
        )
        assert result.is_error is True
        assert "too long" in result.output.lower()

    @pytest.mark.asyncio
    async def test_no_api_key(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NIA_IMAGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("niaharness.config.settings.load_settings") as mock_load:
            mock_settings = MagicMock()
            mock_settings.resolve_api_key.side_effect = ValueError("no key")
            mock_settings.base_url = None
            mock_load.return_value = mock_settings

            result = await ImageGenerateTool().execute(
                ImageGenerateInput(prompt="a cat"),
                context,
            )
            assert result.is_error is True
            assert "API key" in result.output


# ---------------------------------------------------------------------------
# Mocked API call
# ---------------------------------------------------------------------------


class TestMockedGeneration:
    @pytest.mark.asyncio
    async def test_successful_generation_b64(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        """Test successful image generation with base64 response."""
        monkeypatch.setenv("NIA_IMAGE_API_KEY", "test-key")

        # Create a tiny PNG (1x1 red pixel).
        import struct
        import zlib

        def _make_png():
            width, height = 1, 1
            raw = b"\x00\xff\x00\x00"  # filter byte + RGB
            compressed = zlib.compress(raw)
            png = b"\x89PNG\r\n\x1a\n"
            # IHDR chunk
            ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
            png += struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
            # IDAT chunk
            idat_crc = zlib.crc32(b"IDAT" + compressed)
            png += struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
            # IEND chunk
            iend_crc = zlib.crc32(b"IEND")
            png += struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            return png

        png_bytes = _make_png()
        b64 = base64.b64encode(png_bytes).decode("ascii")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "data": [{"b64_json": b64, "format": "png"}]
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await ImageGenerateTool().execute(
                ImageGenerateInput(prompt="a red dot", size="1024x1024"),
                context,
            )

        assert result.is_error is False
        assert "Generated 1 image" in result.output
        assert "dall-e-3" in result.output
        assert "1024x1024" in result.output
        assert len(result.metadata["paths"]) == 1
        # Verify the file was written.
        saved_path = Path(result.metadata["paths"][0])
        assert saved_path.exists()
        assert saved_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_api_error(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        """Test that API errors are handled."""
        monkeypatch.setenv("NIA_IMAGE_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("429", request=MagicMock(), response=mock_response)
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await ImageGenerateTool().execute(
                ImageGenerateInput(prompt="test"),
                context,
            )

        assert result.is_error is True
        assert "429" in result.output
        assert "Rate limit" in result.output

    @pytest.mark.asyncio
    async def test_no_images_returned(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        """Test that empty response is handled."""
        monkeypatch.setenv("NIA_IMAGE_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"data": []})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await ImageGenerateTool().execute(
                ImageGenerateInput(prompt="test"),
                context,
            )

        assert result.is_error is True
        assert "no images" in result.output.lower()


# ---------------------------------------------------------------------------
# Read-only flag
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_is_read_only(self):
        tool = ImageGenerateTool()
        assert tool.is_read_only(ImageGenerateInput(prompt="test")) is True
