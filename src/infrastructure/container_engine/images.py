"""Runtime Image Registry — Maps runtime names to Docker image configurations.

Each RuntimeImage defines how to launch a specific language environment
inside Docker, including the base image, entrypoint command, and
package install syntax.

Usage:
    from src.infrastructure.container_engine.images import RUNTIME_REGISTRY

    python_rt = RUNTIME_REGISTRY["python"]
    print(python_rt.image)  # "python:3.11-slim"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class RuntimeImage:
    """Metadata for a Docker runtime environment."""

    image: str              # Docker image tag (e.g., "python:3.11-slim")
    entrypoint: str         # Default command interpreter (e.g., "python")
    install_cmd: str        # Package install prefix (e.g., "pip install --no-cache-dir")
    healthcheck: str        # Command to verify runtime is working
    description: str        # Human-readable purpose


# =============================================================================
# Registry — Add new runtimes here
# =============================================================================

RUNTIME_REGISTRY: dict[str, RuntimeImage] = {
    "python": RuntimeImage(
        image="python:3.11-slim",
        entrypoint="python",
        install_cmd="pip install --no-cache-dir",
        healthcheck="python --version",
        description="Python 3.11 for data science, ML, scripting",
    ),
    "node": RuntimeImage(
        image="node:20-slim",
        entrypoint="node",
        install_cmd="npm install --no-save",
        healthcheck="node --version",
        description="Node.js 20 for web automation, TypeScript, APIs",
    ),
    "playwright": RuntimeImage(
        image="mcr.microsoft.com/playwright:v1.42.0-jammy",
        entrypoint="npx",
        install_cmd="npm install --no-save",
        healthcheck="npx playwright --version",
        description="Playwright + Chromium for browser automation",
    ),
    "bash": RuntimeImage(
        image="alpine:3.19",
        entrypoint="sh",
        install_cmd="apk add --no-cache",
        healthcheck="echo ok",
        description="Lightweight shell for simple commands",
    ),
}
