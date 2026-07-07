"""Vision analysis tool — load images into the conversation for multimodal models.

The audit (P0 Task 4) flagged that NIA has no multimodal vision tool.
This tool fills that gap, letting the agent analyze images from URLs or
local file paths.

How it works
------------
Unlike most tools that return text to the agent's context, this tool makes
its **own** LLM call to a vision-capable model (separate from the main
conversation). This mirrors Hermes Agent's approach (auxiliary client for
vision) and avoids needing to add image-block support to NIA's core message
serialization.

The tool:
1. Resolves the image source (HTTP/HTTPS URL or local file path).
2. Loads the raw bytes and detects the MIME type.
3. Encodes to base64.
4. Makes a single non-streaming chat completion call with the image as an
   ``image_url`` content block (OpenAI vision API format).
5. Returns the model's analysis text to the agent.

Configuration
-------------
The vision model and API credentials are resolved in this order:

1. ``NIA_VISION_API_KEY`` + ``NIA_VISION_BASE_URL`` + ``NIA_VISION_MODEL``
   env vars (dedicated vision provider — recommended).
2. The main agent's settings (``api_key``, ``base_url``, ``model``) loaded
   via :func:`niaharness.config.settings.load_settings`.
3. ``OPENAI_API_KEY`` env var with a sensible default model.

Default vision model: ``gpt-4o`` (OpenAI's multimodal model). Override
with ``NIA_VISION_MODEL``.

Safety
------
- HTTP downloads capped at 50 MB (decompression-bomb defense).
- Local file paths must exist and be under 50 MB.
- Only image MIME types accepted (jpeg, png, gif, webp, bmp, svg).
- Timeout: 60s for downloads, 120s for the LLM call.
- URL scheme allowlist: ``http``, ``https``, ``file`` (file:// blocked by
  default to prevent reading arbitrary system files — use local paths
  instead, which are explicit).

Reference: Hermes Agent's ``tools/vision_tools.py::vision_analyze_tool``.
Hermes uses an auxiliary client + image source resolver; NIA's version is
simpler — direct httpx download + OpenAI client call — but follows the
same pattern of doing the vision call outside the main conversation.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50 MB
_DOWNLOAD_TIMEOUT = 60.0
_LLM_TIMEOUT = 120.0

_DEFAULT_VISION_MODEL = "gpt-4o"

_SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/svg+xml",
}

# MIME type → file extension mapping for temp files (if needed).
_MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class VisionAnalyzeToolInput(BaseModel):
    """Arguments for the vision_analyze tool."""

    image_source: str = Field(
        description=(
            "Image to analyze. Can be:\n"
            "- An HTTP/HTTPS URL (e.g. 'https://example.com/photo.jpg')\n"
            "- A local file path (e.g. '/tmp/screenshot.png' or './diagram.jpg')"
        ),
    )
    prompt: str = Field(
        default="Describe this image in detail. What do you see?",
        description=(
            "Question or instruction for the vision model. "
            "Examples: 'What text is in this image?', 'Describe the architecture "
            "in this diagram', 'Are there any people in this photo?', "
            "'What error is shown in this screenshot?'"
        ),
    )
    max_tokens: int = Field(
        default=1024,
        ge=100,
        le=4096,
        description="Maximum tokens for the vision model's response.",
    )


# ---------------------------------------------------------------------------
# Image resolution
# ---------------------------------------------------------------------------


def _detect_mime_type(source: str, content_bytes: bytes | None = None) -> str | None:
    """Detect the MIME type of an image."""
    # Try by URL/path extension first.
    guessed, _ = mimetypes.guess_type(source)
    if guessed and guessed in _SUPPORTED_MIME_TYPES:
        return guessed
    # Fall back to sniffing magic bytes.
    if content_bytes:
        if content_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if content_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if content_bytes.startswith(b"GIF87a") or content_bytes.startswith(b"GIF89a"):
            return "image/gif"
        if content_bytes.startswith(b"RIFF") and b"WEBP" in content_bytes[:16]:
            return "image/webp"
        if content_bytes.startswith(b"BM"):
            return "image/bmp"
        if b"<svg" in content_bytes[:500]:
            return "image/svg+xml"
    return None


async def _load_image_from_url(url: str) -> tuple[bytes, str]:
    """Download an image from a URL. Returns (bytes, mime_type)."""
    async with httpx.AsyncClient(
        timeout=_DOWNLOAD_TIMEOUT,
        follow_redirects=True,
        max_redirects=5,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.content

    if len(content) > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image too large: {len(content):,} bytes (max {_MAX_IMAGE_BYTES:,})"
        )

    # Detect MIME from content + URL.
    mime = _detect_mime_type(url, content)
    if mime is None:
        # Fall back to Content-Type header.
        mime = response.headers.get("content-type", "").split(";")[0].strip()
    if mime not in _SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported image type: {mime!r}. Supported: {', '.join(sorted(_SUPPORTED_MIME_TYPES))}"
        )
    return content, mime


def _load_image_from_file(path: Path) -> tuple[bytes, str]:
    """Load an image from a local file. Returns (bytes, mime_type)."""
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    content = path.read_bytes()
    if len(content) > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image too large: {len(content):,} bytes (max {_MAX_IMAGE_BYTES:,})"
        )

    mime = _detect_mime_type(str(path), content)
    if mime is None:
        raise ValueError(
            f"Could not determine image type for {path}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_MIME_TYPES))}"
        )
    return content, mime


# ---------------------------------------------------------------------------
# Vision LLM call
# ---------------------------------------------------------------------------


def _resolve_vision_config() -> dict[str, str | None]:
    """Resolve API key, base URL, and model for the vision call.

    Resolution order:
    1. NIA_VISION_* env vars (dedicated vision provider).
    2. Main agent settings (api_key, base_url, model).
    3. OPENAI_API_KEY env var with default model.
    """
    api_key = os.environ.get("NIA_VISION_API_KEY")
    base_url = os.environ.get("NIA_VISION_BASE_URL")
    model = os.environ.get("NIA_VISION_MODEL")

    if api_key:
        return {
            "api_key": api_key,
            "base_url": base_url,  # may be None — OpenAI default
            "model": model or _DEFAULT_VISION_MODEL,
        }

    # Fall back to main agent settings.
    try:
        from niaharness.config.settings import load_settings

        settings = load_settings()
        try:
            resolved_key = settings.resolve_api_key()
        except ValueError:
            resolved_key = ""
        if resolved_key:
            return {
                "api_key": resolved_key,
                "base_url": settings.base_url,
                "model": model or settings.model,
            }
    except Exception as exc:
        logger.debug("Could not load settings for vision config: %s", exc)

    # Last resort: OPENAI_API_KEY.
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        return {
            "api_key": openai_key,
            "base_url": base_url,
            "model": model or _DEFAULT_VISION_MODEL,
        }

    return {"api_key": "", "base_url": base_url, "model": model or _DEFAULT_VISION_MODEL}


async def _call_vision_model(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    *,
    api_key: str,
    base_url: str | None,
    model: str,
    max_tokens: int,
) -> str:
    """Make a vision LLM call and return the analysis text."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=_LLM_TIMEOUT)

    # Encode image as base64 data URL.
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{b64}"

    # Build the message with text + image content blocks (OpenAI vision format).
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=0.1,  # low temp for factual image description
    )

    # Extract the text from the response.
    if not response.choices:
        return "(no response from vision model)"
    choice = response.choices[0]
    if choice.message and choice.message.content:
        return choice.message.content
    return "(empty response from vision model)"


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class VisionAnalyzeTool(BaseTool):
    """Analyze an image using a vision-capable LLM."""

    name = "vision_analyze"
    description = (
        "Analyze an image from a URL or local file path using a vision-capable "
        "AI model. Use this to: describe what's in an image, read text from "
        "screenshots, identify objects/people/scenes, interpret diagrams or "
        "charts, or answer questions about visual content. The image is "
        "processed by a separate vision model — the analysis text is returned "
        "to your context."
    )
    input_model = VisionAnalyzeToolInput

    def is_read_only(self, arguments: VisionAnalyzeToolInput) -> bool:
        # Reads an image and makes an LLM call — no filesystem mutations.
        # Treat as read-only so the permission system doesn't gate it.
        del arguments
        return True

    async def execute(self, arguments: VisionAnalyzeToolInput, context: ToolExecutionContext) -> ToolResult:
        source = arguments.image_source.strip()
        if not source:
            return ToolResult(output="image_source is required", is_error=True)

        # Step 1: Load the image bytes.
        try:
            if source.startswith(("http://", "https://")):
                image_bytes, mime_type = await _load_image_from_url(source)
                source_label = source
            else:
                # Local file path — resolve relative to cwd.
                path = Path(source)
                if not path.is_absolute():
                    path = (context.cwd / path).resolve()
                image_bytes, mime_type = _load_image_from_file(path)
                source_label = str(path)
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                output=f"Failed to download image: HTTP {exc.response.status_code} {exc.response.reason_phrase}",
                is_error=True,
            )
        except httpx.RequestError as exc:
            return ToolResult(output=f"Failed to download image: {exc}", is_error=True)
        except FileNotFoundError as exc:
            return ToolResult(output=str(exc), is_error=True)
        except ValueError as exc:
            return ToolResult(output=str(exc), is_error=True)
        except Exception as exc:
            return ToolResult(output=f"Failed to load image: {exc}", is_error=True)

        size_kb = len(image_bytes) / 1024

        # Step 2: Resolve vision API config.
        config = _resolve_vision_config()
        if not config["api_key"]:
            return ToolResult(
                output=(
                    "No API key configured for vision analysis. Set one of:\n"
                    "  - NIA_VISION_API_KEY env var (dedicated vision provider)\n"
                    "  - ANTHROPIC_API_KEY / OPENAI_API_KEY env var\n"
                    "  - api_key in ~/.niaharness/settings.json\n"
                    "Optionally also set NIA_VISION_MODEL and NIA_VISION_BASE_URL."
                ),
                is_error=True,
            )

        # Step 3: Call the vision model.
        try:
            analysis = await _call_vision_model(
                image_bytes,
                mime_type,
                arguments.prompt,
                api_key=config["api_key"],
                base_url=config["base_url"],
                model=config["model"] or _DEFAULT_VISION_MODEL,
                max_tokens=arguments.max_tokens,
            )
        except Exception as exc:
            return ToolResult(
                output=f"Vision model call failed: {exc}",
                is_error=True,
            )

        # Step 4: Return the analysis.
        header = (
            f"Image: {source_label}\n"
            f"Type: {mime_type} · Size: {size_kb:.1f} KB\n"
            f"Model: {config['model']}\n"
            f"Prompt: {arguments.prompt[:100]}\n"
            f"---\n"
        )
        return ToolResult(
            output=header + analysis,
            metadata={
                "image_source": source_label,
                "mime_type": mime_type,
                "image_size_bytes": len(image_bytes),
                "model": config["model"],
                "prompt": arguments.prompt,
                "analysis_length": len(analysis),
            },
        )
