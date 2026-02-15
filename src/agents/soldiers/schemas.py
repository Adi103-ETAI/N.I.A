"""Polyglot IPC Schemas — The JSON contract between Brain (Host) and Body (Docker).

These Pydantic models define the structured communication protocol
for the MissionManifest (Brain → Soldier) and MissionResult (Soldier → Brain).

Usage:
    from src.agents.soldiers.schemas import MissionManifest, MissionResult

    manifest = MissionManifest(
        task_id="abc-123",
        soldier_type="coding",
        runtime="python",
        objective="Calculate 2 + 2",
        code="print(2 + 2)",
    )
    manifest.to_json_file(Path("mission.json"))
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class SoldierType(str, Enum):
    """Types of Soldiers in the Swarm."""
    CODING = "coding"
    WEB = "web"
    BUILDER = "builder"
    DESKTOP = "desktop"
    VISION = "vision"
    CONVERSATION = "conversation"


class RuntimeType(str, Enum):
    """Docker runtime environments."""
    PYTHON = "python"
    NODE = "node"
    PLAYWRIGHT = "playwright"
    BASH = "bash"
    CUSTOM = "custom"


class MissionStatus(str, Enum):
    """Outcome of a Soldier mission."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    NEEDS_HELP = "needs_help"


class OutputFormat(str, Enum):
    """Format of the mission output."""
    TEXT = "text"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    BINARY_PATH = "binary_path"


# =============================================================================
# MissionManifest — Brain → Soldier
# =============================================================================

class MissionManifest(BaseModel):
    """The mission briefing passed from the General to a Soldier.
    
    Written as mission.json to the shared Docker volume.
    Read by the entrypoint wrapper inside the container.
    """
    
    # --- Identity ---
    task_id: str = Field(description="Unique mission identifier")
    soldier_type: SoldierType = Field(
        default=SoldierType.CODING,
        description="Which Soldier blueprint to use",
    )
    runtime: RuntimeType = Field(
        default=RuntimeType.PYTHON,
        description="Docker runtime environment",
    )
    
    # --- Mission ---
    objective: str = Field(description="Human-readable goal for the Soldier")
    code: str = Field(
        default="",
        description="Pre-written code to execute (empty for Builder — it writes its own)",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Packages to install before execution (e.g., 'requests', 'cheerio')",
    )
    input_files: list[str] = Field(
        default_factory=list,
        description="Files pre-staged in /workspace/ for the Soldier",
    )
    
    # --- Context ---
    user_query: str = Field(
        default="",
        description="Original user message",
    )
    
    # --- Execution Config ---
    model_type: str = Field(
        default="fast",
        description="LLM tier: 'smart' (70B) or 'fast' (8B)",
    )
    timeout_seconds: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Hard kill deadline in seconds",
    )
    pty: bool = Field(
        default=False,
        description="Allocate pseudo-terminal (required for interactive CLIs like Codex)",
    )
    workdir: str = Field(
        default="/workspace",
        description="Container working directory for workdir isolation",
    )
    host_workdir: Optional[str] = Field(
        default=None,
        description="Host directory to bind mount to /workspace/project (The Wormhole)",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Max self-correction attempts (Builder Soldier)",
    )
    
    # --- Timestamp ---
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    
    # ---- Serialization Helpers ----
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)
    
    def to_json_file(self, path: Path) -> None:
        """Write manifest to a JSON file."""
        path.write_text(self.to_json(), encoding="utf-8")
    
    @classmethod
    def from_json(cls, raw: str) -> MissionManifest:
        """Deserialize from JSON string."""
        return cls.model_validate_json(raw)
    
    @classmethod
    def from_json_file(cls, path: Path) -> MissionManifest:
        """Load manifest from a JSON file."""
        return cls.from_json(path.read_text(encoding="utf-8"))


# =============================================================================
# MissionResult — Soldier → Brain
# =============================================================================

class MissionResult(BaseModel):
    """The final report from a Soldier before death.
    
    Written as result.json by the entrypoint wrapper inside Docker.
    Read by the DockerBridge on the Host.
    """
    
    # --- Identity ---
    task_id: str = Field(description="Matching task_id from the MissionManifest")
    
    # --- Outcome ---
    status: MissionStatus = Field(
        default=MissionStatus.SUCCESS,
        description="Mission outcome",
    )
    exit_code: int = Field(
        default=0,
        description="Process exit code (0 = success)",
    )
    
    # --- Output ---
    output: str = Field(
        default="",
        description="The primary result (answer, data, transcription, etc.)",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.TEXT,
        description="Format of the output field",
    )
    artifacts: list[str] = Field(
        default_factory=list,
        description="File paths created in /workspace/ (relative)",
    )
    
    # --- Error Info ---
    error: Optional[str] = Field(
        default=None,
        description="Error message or stack trace on failure",
    )
    
    # --- Diagnostics ---
    execution_time_seconds: float = Field(
        default=0.0,
        description="Wall-clock execution time",
    )
    retries_used: int = Field(
        default=0,
        description="Self-correction attempts used",
    )
    
    # --- Timestamp ---
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    
    # ---- Serialization Helpers ----
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)
    
    def to_json_file(self, path: Path) -> None:
        """Write result to a JSON file."""
        path.write_text(self.to_json(), encoding="utf-8")
    
    @classmethod
    def from_json(cls, raw: str) -> MissionResult:
        """Deserialize from JSON string."""
        return cls.model_validate_json(raw)
    
    @classmethod
    def from_json_file(cls, path: Path) -> MissionResult:
        """Load result from a JSON file."""
        return cls.from_json(path.read_text(encoding="utf-8"))
