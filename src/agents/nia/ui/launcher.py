"""N.I.A Launcher - Spawns the React frontend with N.I.A backend."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[4]


def get_frontend_dir() -> Path:
    """Return the React terminal frontend directory."""
    return _repo_root() / "frontend" / "terminal"


def build_backend_command(
    *,
    cwd: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[str]:
    """Return the command used by the React frontend to spawn the N.I.A backend."""
    command = [sys.executable, "-m", "agents.nia", "--backend-only"]
    if cwd:
        command.extend(["--cwd", cwd])
    if provider:
        command.extend(["--provider", provider])
    if model:
        command.extend(["--model", model])
    if api_key:
        command.extend(["--api-key", api_key])
    if base_url:
        command.extend(["--base-url", base_url])
    return command


def _ensure_frontend_deps(frontend_dir: Path) -> bool:
    """Ensure npm dependencies are installed. Returns True if successful."""
    if (frontend_dir / "node_modules").exists():
        return True

    logger.info("Installing frontend dependencies...")
    npm = shutil.which("npm")
    if not npm:
        logger.error("npm not found. Please install Node.js and npm.")
        return False

    try:
        result = subprocess.run(
            [npm, "install"],
            cwd=str(frontend_dir),
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"npm install failed: {result.stderr.decode()}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("npm install timed out")
        return False


async def launch_nia_tui(
    *,
    prompt: str | None = None,
    cwd: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> int:
    """Launch the N.I.A React terminal frontend.

    Returns the exit code from the frontend process.
    """
    import asyncio

    frontend_dir = get_frontend_dir()

    if not (frontend_dir / "package.json").exists():
        logger.error(f"Frontend not found at {frontend_dir}")
        return 1

    # Ensure node_modules exists
    if not _ensure_frontend_deps(frontend_dir):
        return 1

    # Build the backend command
    backend_command = build_backend_command(
        cwd=cwd,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    # Set frontend config
    config = {
        "backend_command": backend_command,
        "initial_prompt": prompt,
    }

    env = os.environ.copy()
    env["NIA_FRONTEND_CONFIG"] = json.dumps(config)

    # Find npx or npm for launching tsx
    npx = shutil.which("npx")
    npm = shutil.which("npm")

    if npx:
        cmd = [npx, "tsx", "src/index.tsx"]
    elif npm:
        cmd = [npm, "exec", "tsx", "--", "src/index.tsx"]
    else:
        logger.error("Neither npx nor npm found. Please install Node.js.")
        return 1

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(frontend_dir),
        stdin=None,
        stdout=None,
        stderr=None,
        env=env,
    )

    return await proc.wait()
