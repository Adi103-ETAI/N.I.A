"""Builder Cache — Learn & Cache Protocol for Self-Evolving Skills.

When the Builder Soldier creates a new capability (e.g., STT, PDF parsing),
this module saves it as a proper skill in data/skills/ so the General can
discover and reuse it in future sessions.

v5.0 Phase 3 infrastructure, built now for forward compatibility.

Usage:
    from src.agents.soldiers.builder_cache import cache_learned_skill

    cache_learned_skill(
        name="stt",
        description="Audio transcription using OpenAI Whisper",
        runtime="python",
        code=transcribe_py_source_code,
        dependencies=["openai-whisper", "torch"],
    )
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("NIA.Soldiers.BuilderCache")

# Default skills directory
_SKILLS_DIR = Path(__file__).resolve().parents[2] / "data" / "skills"


def cache_learned_skill(
    name: str,
    description: str,
    runtime: str,
    code: str,
    dependencies: list[str] | None = None,
    pty: bool = False,
    workdir: str = "/workspace",
    builder_task_id: str = "unknown",
    skills_dir: Optional[Path] = None,
) -> Path:
    """Save a Builder Soldier's output as a reusable skill.

    Creates the standard skill folder structure:
        data/skills/{name}/
        ├── skill.md        (auto-generated with YAML frontmatter)
        └── source.py/js    (the code that was generated)

    Args:
        name: Skill identifier (lowercase, underscores). e.g., "stt", "pdf_parser"
        description: Human-readable description for the General's Skill Library.
        runtime: "python" or "node"
        code: The source code to save.
        dependencies: List of packages required (e.g., ["openai-whisper", "torch"]).
        pty: Whether this skill needs a pseudo-terminal.
        workdir: Container working directory.
        builder_task_id: The task_id of the Builder mission that created this.
        skills_dir: Override for the skills directory.

    Returns:
        Path to the created skill directory.
    """
    base = skills_dir or _SKILLS_DIR
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    deps = dependencies or []
    timestamp = datetime.now(timezone.utc).isoformat()

    # --- Generate skill.md ---
    deps_yaml = json.dumps(deps) if deps else "[]"
    skill_md = f"""---
name: {name}
description: {description}
runtime: {runtime}
dependencies: {deps_yaml}
pty: {"true" if pty else "false"}
workdir: {workdir}
created_by: builder_{builder_task_id}
created_at: {timestamp}
---

# 🤖 {name.replace("_", " ").title()} (Auto-Generated)

> This skill was automatically created by the Builder Soldier.
> Task ID: `{builder_task_id}`
> Created: {timestamp}

## Description

{description}

## Dependencies

{chr(10).join(f"- `{d}`" for d in deps) if deps else "None"}

## Usage

This skill is automatically discovered by the General via `load_docker_skills()`.
It will be executed in a `{runtime}` Docker container.
"""

    (skill_dir / "skill.md").write_text(skill_md.strip() + "\n", encoding="utf-8")

    # --- Write source code ---
    ext = "py" if runtime == "python" else "js"
    source_file = skill_dir / f"source.{ext}"
    source_file.write_text(code, encoding="utf-8")

    logger.info(
        f"💾 Cached learned skill: {name} → {skill_dir} "
        f"(runtime={runtime}, deps={len(deps)})"
    )

    return skill_dir


def skill_exists(name: str, skills_dir: Optional[Path] = None) -> bool:
    """Check if a skill already exists in the registry."""
    base = skills_dir or _SKILLS_DIR
    skill_dir = base / name
    return (skill_dir / "skill.md").exists()


def list_learned_skills(skills_dir: Optional[Path] = None) -> list[str]:
    """List all skills that were created by the Builder (not builtin).

    Returns skill names where skill.md contains 'created_by: builder_'.
    """
    base = skills_dir or _SKILLS_DIR
    if not base.exists():
        return []

    learned = []
    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        skill_file = item / "skill.md"
        if skill_file.exists():
            try:
                content = skill_file.read_text(encoding="utf-8")
                if "created_by: builder_" in content:
                    learned.append(item.name)
            except Exception:
                continue

    return learned
