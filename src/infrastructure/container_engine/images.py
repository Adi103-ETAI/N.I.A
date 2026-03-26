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
        image="nia-sandbox-common:latest",
        entrypoint="python3",
        install_cmd="pip install --no-cache-dir",
        healthcheck="python3 --version",
        description="N.I.A. Sandbox — Python 3 + Pi-Mono + GUI tools",
    ),
    "node": RuntimeImage(
        image="nia-sandbox-common:latest",
        entrypoint="node",
        install_cmd="npm install --no-save",
        healthcheck="node --version",
        description="N.I.A. Sandbox — Node.js + Pi-Mono + GUI tools",
    ),
    "playwright": RuntimeImage(
        image="mcr.microsoft.com/playwright:v1.42.0-jammy",
        entrypoint="npx",
        install_cmd="npm install --no-save",
        healthcheck="npx playwright --version",
        description="Playwright + Chromium for browser automation",
    ),
    "bash": RuntimeImage(
        image="nia-sandbox-common:latest",
        entrypoint="bash",
        install_cmd="apt-get install -y --no-install-recommends",
        healthcheck="echo ok",
        description="N.I.A. Sandbox — Full shell with all tools",
    ),
    "custom": RuntimeImage(
        image="nia-sandbox-common:latest",
        entrypoint="python3",
        install_cmd="pip install --no-cache-dir",
        healthcheck="python3 --version",
        description="N.I.A. Sandbox — Custom runtime (defaults to Python)",
    ),
}
