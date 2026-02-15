"""DockerBridge — The Universal Python ↔ Docker ↔ Any Language Adapter.

Handles execution of MissionManifests in Docker containers across multiple
language runtimes (Python, Node.js, Bash). Uses file-based JSON IPC via
shared volume mounts — zero network dependencies.

v5.0 Phase 1: Core bridge for the Polyglot Swarm.

Usage:
    from src.infrastructure.container_engine.bridge import DockerBridge
    from src.infrastructure.container_engine.manager import DockerEngine
    from src.agents.soldiers.schemas import MissionManifest

    bridge = DockerBridge(DockerEngine())
    result = bridge.execute_mission(manifest)
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from src.agents.soldiers.schemas import (
    MissionManifest,
    MissionResult,
    MissionStatus,
    RuntimeType,
)
from src.infrastructure.container_engine.images import RUNTIME_REGISTRY

logger = logging.getLogger("NIA.Infrastructure.DockerBridge")


# =============================================================================
# Entrypoint Templates
# =============================================================================

PYTHON_ENTRYPOINT = '''\
"""Auto-generated entrypoint — wraps soldier code with mission.json / result.json protocol."""
import json
import sys
import time
import traceback

def main():
    start = time.time()

    with open("/workspace/mission.json", "r") as f:
        mission = json.load(f)

    result = {
        "task_id": mission.get("task_id", "unknown"),
        "status": "success",
        "exit_code": 0,
        "output": "",
        "output_format": "text",
        "artifacts": [],
        "error": None,
        "execution_time_seconds": 0.0,
        "retries_used": 0,
        "timestamp": "",
    }

    try:
        # Execute the soldier code in a controlled namespace
        code_globals = {"__name__": "__main__", "mission": mission, "result": result}
        exec(open("/workspace/soldier_code.py").read(), code_globals)

        # Allow soldier code to update result dict directly
        if "result" in code_globals:
            result = code_globals["result"]
    except SystemExit as e:
        result["exit_code"] = e.code if isinstance(e.code, int) else 1
        if e.code != 0:
            result["status"] = "failure"
            result["error"] = f"SystemExit({e.code})"
    except Exception:
        result["status"] = "failure"
        result["error"] = traceback.format_exc()
        result["exit_code"] = 1

    result["execution_time_seconds"] = round(time.time() - start, 3)

    with open("/workspace/result.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
'''

NODE_ENTRYPOINT = '''\
/**
 * Auto-generated entrypoint — wraps soldier code with mission.json / result.json protocol.
 */
const fs = require('fs');
const path = require('path');

async function main() {
    const start = Date.now();
    const mission = JSON.parse(fs.readFileSync('/workspace/mission.json', 'utf8'));

    const result = {
        task_id: mission.task_id || 'unknown',
        status: 'success',
        exit_code: 0,
        output: '',
        output_format: 'text',
        artifacts: [],
        error: null,
        execution_time_seconds: 0.0,
        retries_used: 0,
        timestamp: new Date().toISOString(),
    };

    try {
        const soldierModule = require('/workspace/soldier_code.js');

        if (typeof soldierModule === 'function') {
            const output = await soldierModule(mission);
            result.output = typeof output === 'string' ? output : JSON.stringify(output);
        } else if (typeof soldierModule.run === 'function') {
            const output = await soldierModule.run(mission);
            result.output = typeof output === 'string' ? output : JSON.stringify(output);
        }
    } catch (e) {
        result.status = 'failure';
        result.error = e.stack || String(e);
        result.exit_code = 1;
    }

    result.execution_time_seconds = (Date.now() - start) / 1000;
    fs.writeFileSync('/workspace/result.json', JSON.stringify(result, null, 2));
}

main().catch(err => {
    fs.writeFileSync('/workspace/result.json', JSON.stringify({
        task_id: 'unknown',
        status: 'failure',
        exit_code: 1,
        error: err.stack || String(err),
    }, null, 2));
    process.exit(1);
});
'''


# =============================================================================
# DockerBridge
# =============================================================================

class DockerBridge:
    """Universal execution bridge: Python (Host) → Docker (Container) → Any Language.

    Protocol:
        1. Write mission.json + entrypoint + soldier_code to workspace volume
        2. docker run {image} {entrypoint} /workspace/_entrypoint.{ext}
        3. Container reads mission.json, executes code, writes result.json
        4. Bridge reads result.json and returns MissionResult
    """

    def __init__(self, engine):
        """Initialize the bridge.

        Args:
            engine: A DockerEngine instance (from src.infrastructure.container_engine.manager).
                    Accepts any object with `run_command()` and optionally `pull_image()`.
        """
        self.engine = engine
        self._base_dir = Path(__file__).resolve().parents[3]  # N.I.A project root
        self._workspace_root = self._base_dir / "data" / "sandbox_mounts"

    # =========================================================================
    # Public API
    # =========================================================================

    def execute_mission(self, manifest: MissionManifest) -> MissionResult:
        """Execute a mission inside a Docker container.

        Steps:
            1. Prepare workspace directory with mission files
            2. Write the entrypoint wrapper (Python or Node.js)
            3. Copy/write the soldier code
            4. Call DockerEngine.run_command with the correct image
            5. Read result.json and return MissionResult

        Args:
            manifest: The mission briefing (task_id, runtime, code, etc.)

        Returns:
            MissionResult from the container, or a failure result if execution failed.
        """
        start_time = time.time()
        workspace = self._prepare_workspace(manifest)

        try:
            # --- Write files to workspace ---
            self._write_mission_json(workspace, manifest)
            self._write_entrypoint(workspace, manifest)
            self._write_soldier_code(workspace, manifest)
            self._stage_input_files(workspace, manifest)

            # --- Resolve runtime ---
            runtime_key = manifest.runtime.value
            runtime = RUNTIME_REGISTRY.get(runtime_key)
            if runtime is None:
                return self._make_failure(
                    manifest, f"Unknown runtime: {runtime_key}", start_time,
                )

            # --- Ensure image is available ---
            try:
                self.engine.pull_image(runtime.image)
            except Exception as e:
                logger.warning(f"Image pull skipped (may already exist): {e}")

            # --- Build the command ---
            command = self._build_command(manifest, runtime)

            # --- Mount the workspace ---
            mounts = {
                str(workspace.absolute()): {
                    "bind": "/workspace",
                    "mode": "rw",
                }
            }

            # Phase 2: Host Mounts (The Wormhole)
            if manifest.host_workdir:
                # Security: In production, we'd validate this path against an allowlist
                # For now, we trust the General/User to provide a valid path
                mounts[manifest.host_workdir] = {
                    "bind": "/workspace/project",
                    "mode": "rw"
                }
                
                # Auto-switch execution context to the project folder
                # This ensures relative paths in commands (e.g., "ls") verify the host files
                if manifest.workdir == "/workspace":
                    manifest.workdir = "/workspace/project"

            # --- Execute (PTY or standard) ---
            logger.info(
                f"🚀 Executing mission {manifest.task_id} "
                f"[{runtime_key}] image={runtime.image}"
                f"{' (PTY)' if manifest.pty else ''}"
            )

            if manifest.pty:
                # Interactive CLI mode (Codex, Aider, etc.)
                exit_code, stdout, stderr = self.engine.run_command_pty(
                    image=runtime.image,
                    command=command,
                    session_id=manifest.task_id,
                    mounts=mounts,
                    workdir=manifest.workdir,
                    timeout=manifest.timeout_seconds,
                )
            else:
                # Standard non-interactive execution
                exit_code, stdout, stderr = self.engine.run_command(
                    image=runtime.image,
                    command=command,
                    session_id=manifest.task_id,
                    mounts=mounts,
                )

            logger.info(
                f"📦 Mission {manifest.task_id} finished: "
                f"exit_code={exit_code}"
            )

            # --- Collect result ---
            result_path = workspace / "result.json"
            if result_path.exists():
                try:
                    result = MissionResult.from_json_file(result_path)
                    return result
                except Exception as e:
                    logger.warning(f"Failed to parse result.json: {e}")

            # Fallback: build result from stdout/stderr
            return MissionResult(
                task_id=manifest.task_id,
                status=MissionStatus.SUCCESS if exit_code == 0 else MissionStatus.FAILURE,
                exit_code=exit_code,
                output=stdout.strip(),
                error=stderr.strip() if stderr.strip() else None,
                execution_time_seconds=round(time.time() - start_time, 3),
            )

        except Exception as e:
            logger.error(f"Bridge execution failed for {manifest.task_id}: {e}")
            return self._make_failure(manifest, str(e), start_time)

    # =========================================================================
    # Workspace Preparation
    # =========================================================================

    def _prepare_workspace(self, manifest: MissionManifest) -> Path:
        """Create a clean workspace directory for this mission."""
        workspace = self._workspace_root / manifest.task_id
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _write_mission_json(self, workspace: Path, manifest: MissionManifest):
        """Write the mission manifest as JSON to the workspace."""
        manifest.to_json_file(workspace / "mission.json")

    def _write_entrypoint(self, workspace: Path, manifest: MissionManifest):
        """Write the language-specific entrypoint wrapper."""
        if manifest.runtime in (RuntimeType.PYTHON, RuntimeType.CUSTOM):
            (workspace / "_entrypoint.py").write_text(
                PYTHON_ENTRYPOINT, encoding="utf-8"
            )
        elif manifest.runtime in (RuntimeType.NODE, RuntimeType.PLAYWRIGHT):
            (workspace / "_entrypoint.js").write_text(
                NODE_ENTRYPOINT, encoding="utf-8"
            )
        else:
            # Bash: soldier_code IS the entrypoint
            pass

    def _write_soldier_code(self, workspace: Path, manifest: MissionManifest):
        """Write the soldier source code to the workspace."""
        if not manifest.code:
            return  # Builder Soldier writes its own code

        if manifest.runtime in (RuntimeType.PYTHON, RuntimeType.CUSTOM):
            (workspace / "soldier_code.py").write_text(
                manifest.code, encoding="utf-8"
            )
        elif manifest.runtime in (RuntimeType.NODE, RuntimeType.PLAYWRIGHT):
            (workspace / "soldier_code.js").write_text(
                manifest.code, encoding="utf-8"
            )
        else:
            (workspace / "soldier_code.sh").write_text(
                manifest.code, encoding="utf-8"
            )

    def _stage_input_files(self, workspace: Path, manifest: MissionManifest):
        """Copy input files to the workspace if they exist."""
        for file_path in manifest.input_files:
            src = Path(file_path)
            if src.exists():
                dst = workspace / src.name
                shutil.copy2(str(src), str(dst))
                logger.debug(f"Staged input file: {src.name}")

    # =========================================================================
    # Command Building
    # =========================================================================

    def _build_command(self, manifest: MissionManifest, runtime) -> str:
        """Build the shell command to execute inside the container."""
        install_step = ""
        if manifest.dependencies:
            deps = " ".join(manifest.dependencies)
            install_step = f"{runtime.install_cmd} {deps} && "

        if manifest.runtime in (RuntimeType.PYTHON, RuntimeType.CUSTOM):
            return f"bash -c '{install_step}python /workspace/_entrypoint.py'"
        elif manifest.runtime in (RuntimeType.NODE, RuntimeType.PLAYWRIGHT):
            return f"bash -c '{install_step}node /workspace/_entrypoint.js'"
        else:
            # Bash: direct execution
            return f"bash -c '{install_step}sh /workspace/soldier_code.sh'"

    # =========================================================================
    # Helpers
    # =========================================================================

    def _make_failure(
        self,
        manifest: MissionManifest,
        error: str,
        start_time: float,
    ) -> MissionResult:
        """Create a failure MissionResult."""
        return MissionResult(
            task_id=manifest.task_id,
            status=MissionStatus.FAILURE,
            exit_code=-1,
            output="",
            error=error,
            execution_time_seconds=round(time.time() - start_time, 3),
        )

    def cleanup_workspace(self, task_id: str):
        """Remove the workspace directory for a completed mission."""
        workspace = self._workspace_root / task_id
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
            logger.debug(f"Cleaned up workspace: {task_id}")
