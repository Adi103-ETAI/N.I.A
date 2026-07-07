"""Image generation tool — text-to-image via OpenAI-compatible image APIs.

Adapted from Hermes Agent's tools/image_generation_tool.py.

Uses the OpenAI images API format (/v1/images/generations), which works
with DALL-E and OpenAI-compatible providers. A model catalog (adapted from
Hermes's FAL_MODELS pattern) provides per-model metadata: supported
parameters, sizes, and defaults — so the payload is filtered to only keys
the model accepts.

Configuration (3-tier):
1. ``NIA_IMAGE_API_KEY`` + ``NIA_IMAGE_BASE_URL`` + ``NIA_IMAGE_MODEL``
   env vars (dedicated image provider — recommended).
2. Main agent settings (``api_key``, ``base_url``).
3. ``OPENAI_API_KEY`` env var with DALL-E 3 default.

Output directory: ``NIA_IMAGE_OUTPUT_DIR`` env var (default:
``/home/z/my-project/download``).
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog (adapted from Hermes's FAL_MODELS pattern)
# ---------------------------------------------------------------------------

# Each model entry has:
# - display: human-readable name
# - supports: set of API keys the model accepts (others are filtered out)
# - sizes: list of supported size strings
# - defaults: dict of default parameters merged into the payload

IMAGE_MODELS: dict[str, dict[str, Any]] = {
    "dall-e-3": {
        "display": "DALL-E 3",
        "supports": {"model", "prompt", "n", "size", "quality", "style", "response_format"},
        "sizes": ["1024x1024", "1024x1792", "1792x1024"],
        "defaults": {"quality": "standard", "style": "natural"},
    },
    "dall-e-2": {
        "display": "DALL-E 2",
        "supports": {"model", "prompt", "n", "size", "response_format"},
        "sizes": ["1024x1024", "512x512", "256x256"],
        "defaults": {},
    },
    "gpt-image-1": {
        "display": "GPT Image 1",
        "supports": {"model", "prompt", "n", "size", "quality"},
        "sizes": ["1024x1024", "1536x1024", "1024x1536"],
        "defaults": {"quality": "auto"},
    },
}

# Default model when none is configured.
_DEFAULT_MODEL = "dall-e-3"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_MAX_PROMPT_LENGTH = 4000


def _get_model_meta(model: str) -> dict[str, Any]:
    """Return model metadata from the catalog, or a generic entry for unknown models."""
    if model in IMAGE_MODELS:
        return IMAGE_MODELS[model]
    # Generic entry for unknown OpenAI-compatible models — accept everything.
    return {
        "display": model,
        "supports": {"model", "prompt", "n", "size", "response_format"},
        "sizes": [],
        "defaults": {},
    }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ImageGenerateInput(BaseModel):
    """Arguments for the image_generate tool."""

    prompt: str = Field(
        description="Text description of the image to generate. Be specific and detailed.",
    )
    size: str = Field(
        default="1024x1024",
        description="Image dimensions (e.g. '1024x1024', '1024x1792', '1792x1024').",
    )
    quality: Literal["standard", "hd", "auto"] = Field(
        default="standard",
        description="Image quality. 'hd' is higher quality but costs more. 'auto' for gpt-image-1.",
    )
    style: Literal["natural", "vivid"] = Field(
        default="natural",
        description="Image style (DALL-E 3 only). 'natural' = realistic, 'vivid' = hyperreal/dramatic.",
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

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        return {
            "api_key": openai_key,
            "base_url": base_url or _DEFAULT_BASE_URL,
            "model": model or _DEFAULT_MODEL,
        }

    return {"api_key": "", "base_url": base_url or _DEFAULT_BASE_URL, "model": model or _DEFAULT_MODEL}


def _get_output_dir() -> Path:
    """Return the output directory for generated images."""
    return Path(os.environ.get("NIA_IMAGE_OUTPUT_DIR", "/home/z/my-project/download"))


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class ImageGenerateTool(BaseTool):
    """Generate images from text prompts using a text-to-image AI model."""

    name = "image_generate"
    description = (
        "Generate an image from a text description. Specify the prompt, "
        "size, quality, and style. Images are saved to the output directory. "
        "Uses the OpenAI images API format — works with DALL-E 3 and "
        "OpenAI-compatible providers. Configurable via NIA_IMAGE_* env vars."
    )
    input_model = ImageGenerateInput

    def is_read_only(self, arguments: ImageGenerateInput) -> bool:
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

        model = config["model"] or _DEFAULT_MODEL
        model_meta = _get_model_meta(model)

        # Validate size against model catalog (if the model has known sizes).
        if model_meta["sizes"] and arguments.size not in model_meta["sizes"]:
            supported = ", ".join(model_meta["sizes"])
            return ToolResult(
                output=(
                    f"Model '{model}' does not support size '{arguments.size}'. "
                    f"Supported sizes: {supported}"
                ),
                is_error=True,
            )

        # Call the image generation API.
        try:
            images = await self._call_image_api(arguments, config, model_meta)
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
        out_dir = _get_output_dir()
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
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.get(img_data["url"])
                    response.raise_for_status()
                    out_path.write_bytes(response.content)
            else:
                logger.warning("Image %d has no b64_json or url — skipping", i)
                continue

            saved_paths.append(str(out_path))

        if not saved_paths:
            return ToolResult(output="No images could be saved.", is_error=True)

        # Audit fix: report partial success if fewer images than requested.
        partial_note = ""
        if len(saved_paths) < arguments.n:
            partial_note = (
                f"\n  ⚠ Requested {arguments.n} image(s) but only {len(saved_paths)} "
                f"were returned by the API."
            )

        lines = [
            f"Generated {len(saved_paths)} image(s):",
            f"  Prompt: {arguments.prompt[:100]}",
            f"  Model: {model} ({model_meta['display']})",
            f"  Size: {arguments.size} · Quality: {arguments.quality} · Style: {arguments.style}",
            "",
        ]
        for p in saved_paths:
            size_kb = Path(p).stat().st_size // 1024
            lines.append(f"  {p} ({size_kb} KB)")

        if partial_note:
            lines.append(partial_note)

        return ToolResult(
            output="\n".join(lines),
            metadata={
                "paths": saved_paths,
                "model": model,
                "prompt": arguments.prompt,
                "size": arguments.size,
                "requested_count": arguments.n,
                "returned_count": len(saved_paths),
                "partial_success": len(saved_paths) < arguments.n,
            },
        )

    async def _call_image_api(
        self,
        arguments: ImageGenerateInput,
        config: dict[str, str | None],
        model_meta: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Call the /v1/images/generations endpoint.

        Payload is filtered to the model's supported keys (adapted from
        Hermes's _build_fal_payload pattern).
        """
        api_key = config["api_key"]
        base_url = (config["base_url"] or _DEFAULT_BASE_URL).rstrip("/")
        model = config["model"] or _DEFAULT_MODEL
        supports = model_meta["supports"]
        defaults = model_meta.get("defaults", {})

        # Build payload — start with defaults, then apply caller values,
        # then filter to the model's supported keys.
        payload: dict[str, Any] = {
            "model": model,
            "prompt": arguments.prompt,
            "n": arguments.n,
            "size": arguments.size,
            "response_format": "b64_json",
            "quality": arguments.quality,
            "style": arguments.style,
        }

        # Merge defaults (model-specific defaults override caller defaults
        # only if the caller didn't explicitly set them — but since we use
        # Pydantic defaults, the caller always sets them; so we just use
        # defaults for keys the caller didn't provide).
        for k, v in defaults.items():
            payload.setdefault(k, v)

        # Filter to supported keys (adapted from Hermes's _build_fal_payload).
        # Always keep 'prompt' even if not in supports (defensive).
        filtered_payload = {}
        for k, v in payload.items():
            if k in supports or k == "prompt":
                filtered_payload[k] = v

        logger.info(
            "Image generate: model=%s, payload keys=%s",
            model,
            list(filtered_payload.keys()),
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{base_url}/images/generations",
                json=filtered_payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

        return data.get("data", [])
