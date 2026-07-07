"""Image generation tool — text-to-image via OpenAI-compatible image APIs.

The audit (P1 #7) flagged that NIA has no text-to-image tool. This tool
fills that gap, letting the agent generate images from text prompts.

Supports any OpenAI-compatible image generation API:
- OpenAI DALL-E 3 / DALL-E 2 (default)
- Any provider that implements the ``/v1/images/generations`` endpoint

Configuration (3-tier, same pattern as vision_analyze):
1. ``NIA_IMAGE_API_KEY`` + ``NIA_IMAGE_BASE_URL`` + ``NIA_IMAGE_MODEL``
   env vars (dedicated image provider — recommended).
2. Main agent settings (``api_key``, ``base_url``).
3. ``OPENAI_API_KEY`` env var with DALL-E 3 default.

Generated images are saved to ``/home/z/my-project/download/`` and the
path is returned to the agent.

Reference: Hermes Agent's ``tools/image_generation_tool.py`` (uses FAL.ai).
NIA's version is simpler — uses the OpenAI images API format, which works
with DALL-E and any compatible provider.
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "dall-e-3"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_SIZE = "1024x1024"
_MAX_PROMPT_LENGTH = 4000


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ImageGenerateInput(BaseModel):
    """Arguments for the image_generate tool."""

    prompt: str = Field(
        description="Text description of the image to generate. Be specific and detailed.",
    )
    size: Literal["1024x1024", "1024x1792", "1792x1024", "512x512", "256x256"] = Field(
        default="1024x1024",
        description="Image dimensions. 1024x1024 (square), 1024x1792 (portrait), 1792x1024 (landscape).",
    )
    quality: Literal["standard", "hd"] = Field(
        default="standard",
        description="Image quality. 'hd' is higher quality but costs more.",
    )
    style: Literal["natural", "vivid"] = Field(
        default="natural",
        description="Image style. 'natural' = realistic, 'vivid' = hyperreal/dramatic.",
    )
    n: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of images to generate (1-4). Each is saved as a separate file.",
    )


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _resolve_image_config() -> dict[str, str | None]:
    """Resolve API key, base URL, and model for image generation.

    Resolution order:
    1. NIA_IMAGE_* env vars (dedicated image provider).
    2. Main agent settings (api_key, base_url).
    3. OPENAI_API_KEY env var with DALL-E 3 default.
    """
    api_key = os.environ.get("NIA_IMAGE_API_KEY")
    base_url = os.environ.get("NIA_IMAGE_BASE_URL")
    model = os.environ.get("NIA_IMAGE_MODEL")

    if api_key:
        return {
            "api_key": api_key,
            "base_url": base_url or _DEFAULT_BASE_URL,
            "model": model or _DEFAULT_MODEL,
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
                "base_url": settings.base_url or _DEFAULT_BASE_URL,
                "model": model or _DEFAULT_MODEL,
            }
    except Exception:
        pass

    # Last resort: OPENAI_API_KEY.
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        return {
            "api_key": openai_key,
            "base_url": base_url or _DEFAULT_BASE_URL,
            "model": model or _DEFAULT_MODEL,
        }

    return {"api_key": "", "base_url": base_url or _DEFAULT_BASE_URL, "model": model or _DEFAULT_MODEL}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class ImageGenerateTool(BaseTool):
    """Generate images from text prompts using a text-to-image AI model."""

    name = "image_generate"
    description = (
        "Generate an image from a text description. Specify the prompt, "
        "size, quality, and style. Images are saved to the download directory. "
        "Uses OpenAI DALL-E 3 by default (configurable via NIA_IMAGE_* env vars)."
    )
    input_model = ImageGenerateInput

    def is_read_only(self, arguments: ImageGenerateInput) -> bool:
        # Generates files but doesn't mutate user state — treat as read-only.
        del arguments
        return True

    async def execute(self, arguments: ImageGenerateInput, context: ToolExecutionContext) -> ToolResult:
        if not arguments.prompt.strip():
            return ToolResult(output="prompt is required", is_error=True)
        if len(arguments.prompt) > _MAX_PROMPT_LENGTH:
            return ToolResult(
                output=f"prompt is too long ({len(arguments.prompt)} chars; max {_MAX_PROMPT_LENGTH})",
                is_error=True,
            )

        # Resolve config.
        config = _resolve_image_config()
        if not config["api_key"]:
            return ToolResult(
                output=(
                    "No API key configured for image generation. Set one of:\n"
                    "  - NIA_IMAGE_API_KEY env var (dedicated image provider)\n"
                    "  - ANTHROPIC_API_KEY / OPENAI_API_KEY env var\n"
                    "  - api_key in ~/.niaharness/settings.json\n"
                    "Optionally also set NIA_IMAGE_MODEL and NIA_IMAGE_BASE_URL."
                ),
                is_error=True,
            )

        # Call the image generation API.
        try:
            images = await self._call_image_api(arguments, config)
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                output=f"Image generation API error: HTTP {exc.response.status_code}\n{exc.response.text[:500]}",
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(output=f"Image generation failed: {exc}", is_error=True)

        if not images:
            return ToolResult(output="Image generation returned no images.", is_error=True)

        # Save images.
        out_dir = Path("/home/z/my-project/download")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        saved_paths: list[str] = []
        for i, img_data in enumerate(images):
            ext = img_data.get("format", "png")
            filename = f"image-{ts}-{i+1}.{ext}" if len(images) > 1 else f"image-{ts}.{ext}"
            out_path = out_dir / filename

            if img_data.get("b64_json"):
                img_bytes = base64.b64decode(img_data["b64_json"])
                out_path.write_bytes(img_bytes)
            elif img_data.get("url"):
                # Download the image from the URL.
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.get(img_data["url"])
                    response.raise_for_status()
                    out_path.write_bytes(response.content)
            else:
                continue

            saved_paths.append(str(out_path))

        if not saved_paths:
            return ToolResult(output="No images could be saved.", is_error=True)

        # Build result.
        lines = [
            f"Generated {len(saved_paths)} image(s):",
            f"  Prompt: {arguments.prompt[:100]}",
            f"  Model: {config['model']}",
            f"  Size: {arguments.size} · Quality: {arguments.quality} · Style: {arguments.style}",
            "",
        ]
        for p in saved_paths:
            size_kb = Path(p).stat().st_size // 1024
            lines.append(f"  {p} ({size_kb} KB)")

        return ToolResult(
            output="\n".join(lines),
            metadata={
                "paths": saved_paths,
                "model": config["model"],
                "prompt": arguments.prompt,
                "size": arguments.size,
                "count": len(saved_paths),
            },
        )

    async def _call_image_api(
        self, arguments: ImageGenerateInput, config: dict[str, str | None]
    ) -> list[dict[str, str]]:
        """Call the /v1/images/generations endpoint. Returns list of image dicts."""
        api_key = config["api_key"]
        base_url = (config["base_url"] or _DEFAULT_BASE_URL).rstrip("/")
        model = config["model"] or _DEFAULT_MODEL

        payload: dict = {
            "model": model,
            "prompt": arguments.prompt,
            "n": arguments.n,
            "size": arguments.size,
            "response_format": "b64_json",
        }
        # DALL-E 3 specific params.
        if model.startswith("dall-e"):
            payload["quality"] = arguments.quality
            payload["style"] = arguments.style

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{base_url}/images/generations",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

        return data.get("data", [])
