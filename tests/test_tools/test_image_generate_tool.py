"""Tests for the image_generate tool (post-audit-fix version)."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from niaharness.tools.base import ToolExecutionContext
from niaharness.tools.image_generate_tool import (
    IMAGE_MODELS,
    ImageGenerateTool,
    ImageGenerateInput,
    _get_model_meta,
    _get_output_dir,
    _resolve_image_config,
)


@pytest.fixture
def context(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------


class TestModelCatalog:
    def test_dall_e_3_in_catalog(self):
        assert "dall-e-3" in IMAGE_MODELS
        meta = IMAGE_MODELS["dall-e-3"]
        assert "quality" in meta["supports"]
        assert "style" in meta["supports"]
        assert "1024x1024" in meta["sizes"]

    def test_dall_e_2_in_catalog(self):
        assert "dall-e-2" in IMAGE_MODELS
        meta = IMAGE_MODELS["dall-e-2"]
        assert "quality" not in meta["supports"]  # DALL-E 2 doesn't support quality
        assert "512x512" in meta["sizes"]

    def test_gpt_image_1_in_catalog(self):
        assert "gpt-image-1" in IMAGE_MODELS

    def test_unknown_model_returns_generic(self):
        meta = _get_model_meta("some-unknown-model")
        assert "model" in meta["supports"]
        assert "prompt" in meta["supports"]
        assert meta["sizes"] == []

    def test_get_model_meta_known(self):
        meta = _get_model_meta("dall-e-3")
        assert meta["display"] == "DALL-E 3"


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


class TestConfigResolution:
    def test_dedicated_env_vars(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NIA_IMAGE_API_KEY", "image-key")
        monkeypatch.setenv("NIA_IMAGE_BASE_URL", "https://images.example.com/v1")
        monkeypatch.setenv("NIA_IMAGE_MODEL", "dall-e-2")
        config = _resolve_image_config()
        assert config["api_key"] == "image-key"
        assert config["base_url"] == "https://images.example.com/v1"
        assert config["model"] == "dall-e-2"

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
# Output directory
# ---------------------------------------------------------------------------


class TestOutputDir:
    def test_default_output_dir(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NIA_IMAGE_OUTPUT_DIR", raising=False)
        d = _get_output_dir()
        assert str(d) == "/home/z/my-project/download"

    def test_custom_output_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("NIA_IMAGE_OUTPUT_DIR", str(tmp_path))
        d = _get_output_dir()
        assert d == tmp_path


# ---------------------------------------------------------------------------
# Input validation + size validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_empty_prompt(self, context: ToolExecutionContext):
        result = await ImageGenerateTool().execute(ImageGenerateInput(prompt=""), context)
        assert result.is_error is True
        assert "required" in result.output.lower()

    @pytest.mark.asyncio
    async def test_prompt_too_long(self, context: ToolExecutionContext):
        result = await ImageGenerateTool().execute(ImageGenerateInput(prompt="x" * 5000), context)
        assert result.is_error is True
        assert "too long" in result.output.lower()

    @pytest.mark.asyncio
    async def test_unsupported_size_for_model(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        """Size validation against model catalog."""
        monkeypatch.setenv("NIA_IMAGE_API_KEY", "test-key")
        monkeypatch.setenv("NIA_IMAGE_MODEL", "dall-e-2")
        # dall-e-2 supports 512x512 but NOT 1024x1792
        result = await ImageGenerateTool().execute(
            ImageGenerateInput(prompt="test", size="1024x1792"),
            context,
        )
        assert result.is_error is True
        assert "does not support size" in result.output

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
                ImageGenerateInput(prompt="a cat"), context
            )
            assert result.is_error is True
            assert "API key" in result.output


# ---------------------------------------------------------------------------
# Mocked API call
# ---------------------------------------------------------------------------


class TestMockedGeneration:
    @pytest.mark.asyncio
    async def test_successful_generation_b64(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Test successful image generation with base64 response."""
        monkeypatch.setenv("NIA_IMAGE_API_KEY", "test-key")
        monkeypatch.setenv("NIA_IMAGE_MODEL", "dall-e-3")
        monkeypatch.setenv("NIA_IMAGE_OUTPUT_DIR", str(tmp_path))

        import struct
        import zlib

        def _make_png():
            width, height = 1, 1
            raw = b"\x00\xff\x00\x00"
            compressed = zlib.compress(raw)
            png = b"\x89PNG\r\n\x1a\n"
            ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
            png += struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
            idat_crc = zlib.crc32(b"IDAT" + compressed)
            png += struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
            iend_crc = zlib.crc32(b"IEND")
            png += struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            return png

        png_bytes = _make_png()
        b64 = base64.b64encode(png_bytes).decode("ascii")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"data": [{"b64_json": b64, "format": "png"}]})

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
        assert "DALL-E 3" in result.output
        assert result.metadata["returned_count"] == 1
        assert result.metadata["partial_success"] is False

    @pytest.mark.asyncio
    async def test_partial_success_reported(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """If n=3 but API returns 1, report partial success."""
        monkeypatch.setenv("NIA_IMAGE_API_KEY", "test-key")
        monkeypatch.setenv("NIA_IMAGE_MODEL", "dall-e-3")
        monkeypatch.setenv("NIA_IMAGE_OUTPUT_DIR", str(tmp_path))

        import struct
        import zlib

        def _make_png():
            raw = b"\x00\xff\x00\x00"
            compressed = zlib.compress(raw)
            png = b"\x89PNG\r\n\x1a\n"
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
            png += struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
            idat_crc = zlib.crc32(b"IDAT" + compressed)
            png += struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
            iend_crc = zlib.crc32(b"IEND")
            png += struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            return png

        b64 = base64.b64encode(_make_png()).decode("ascii")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        # API returns only 1 image even though n=3
        mock_response.json = MagicMock(return_value={"data": [{"b64_json": b64, "format": "png"}]})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await ImageGenerateTool().execute(
                ImageGenerateInput(prompt="test", n=3),
                context,
            )

        assert result.is_error is False
        assert "⚠" in result.output  # partial success warning
        assert result.metadata["partial_success"] is True
        assert result.metadata["requested_count"] == 3
        assert result.metadata["returned_count"] == 1

    @pytest.mark.asyncio
    async def test_api_error(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
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
                ImageGenerateInput(prompt="test"), context
            )
        assert result.is_error is True
        assert "429" in result.output


# ---------------------------------------------------------------------------
# Payload filtering (adapted from Hermes's _build_fal_payload pattern)
# ---------------------------------------------------------------------------


class TestPayloadFiltering:
    @pytest.mark.asyncio
    async def test_dall_e_2_payload_excludes_quality_style(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """DALL-E 2 doesn't support quality/style — they should be filtered out."""
        monkeypatch.setenv("NIA_IMAGE_API_KEY", "test-key")
        monkeypatch.setenv("NIA_IMAGE_MODEL", "dall-e-2")
        monkeypatch.setenv("NIA_IMAGE_OUTPUT_DIR", str(tmp_path))

        import struct
        import zlib

        def _make_png():
            raw = b"\x00\xff\x00\x00"
            compressed = zlib.compress(raw)
            png = b"\x89PNG\r\n\x1a\n"
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
            png += struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
            idat_crc = zlib.crc32(b"IDAT" + compressed)
            png += struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
            iend_crc = zlib.crc32(b"IEND")
            png += struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            return png

        b64 = base64.b64encode(_make_png()).decode("ascii")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"data": [{"b64_json": b64}]})

        captured_payload = {}
        def _capture_post(url, json=None, headers=None):
            captured_payload.update(json or {})
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json = MagicMock(return_value={"data": [{"b64_json": b64}]})
            return mock_resp

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=_capture_post)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            await ImageGenerateTool().execute(
                ImageGenerateInput(prompt="test", size="512x512"),
                context,
            )

        # DALL-E 2 supports: model, prompt, n, size, response_format
        # quality and style should NOT be in the payload.
        assert "quality" not in captured_payload
        assert "style" not in captured_payload
        assert "model" in captured_payload
        assert "prompt" in captured_payload


# ---------------------------------------------------------------------------
# Read-only flag
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_is_read_only(self):
        assert ImageGenerateTool().is_read_only(ImageGenerateInput(prompt="test")) is True
