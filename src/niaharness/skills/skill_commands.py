"""Shared slash command helpers for skills.

Ported from hermes-agent/agent/skill_commands.py (760 LOC).

Shared between CLI and gateway so both surfaces can invoke skills via
/skill-name commands.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_skill_commands: Dict[str, Dict[str, Any]] = {}
_skill_commands_platform: Optional[str] = None
_SKILL_INVALID_CHARS = re.compile(r"[^a-z0-9-]")
_SKILL_MULTI_HYPHEN = re.compile(r"-{2,}")

# Skill-scaffolding markers.
_SKILL_INVOCATION_PREFIX = "[IMPORTANT: The user has invoked the "
_SINGLE_SKILL_MARKER = "The full skill content is loaded below.]"
_SINGLE_SKILL_INSTRUCTION = (
    "The user has provided the following instruction alongside the skill invocation: "
)
_RUNTIME_NOTE = "\n\n[Runtime note:"
_BUNDLE_MARKER = " skill bundle,"
_BUNDLE_USER_INSTRUCTION = "\nUser instruction: "
_BUNDLE_FIRST_SKILL_BLOCK = "\n\n[Loaded as part of the "


def extract_user_instruction_from_skill_message(content: Any) -> Optional[str]:
    """Recover the user's instruction from a slash-skill-expanded turn.

    Ported from hermes-agent/agent/skill_commands.py line 58.
    """
    if not isinstance(content, str):
        return None
    if not content.startswith(_SKILL_INVOCATION_PREFIX):
        return content
    if _BUNDLE_MARKER in content:
        return _extract_bundle_user_instruction(content)
    if _SINGLE_SKILL_MARKER in content:
        return _extract_single_skill_user_instruction(content)
    return None


def _extract_single_skill_user_instruction(message: str) -> Optional[str]:
    marker_idx = message.rfind(_SINGLE_SKILL_INSTRUCTION)
    if marker_idx < 0:
        return None
    instruction = message[marker_idx + len(_SINGLE_SKILL_INSTRUCTION):]
    runtime_idx = instruction.find(_RUNTIME_NOTE)
    if runtime_idx >= 0:
        instruction = instruction[:runtime_idx]
    instruction = instruction.strip()
    return instruction or None


def _extract_bundle_user_instruction(message: str) -> Optional[str]:
    marker_idx = message.rfind(_BUNDLE_USER_INSTRUCTION)
    if marker_idx < 0:
        return None
    instruction = message[marker_idx + len(_BUNDLE_USER_INSTRUCTION):]
    runtime_idx = instruction.find(_RUNTIME_NOTE)
    if runtime_idx >= 0:
        instruction = instruction[:runtime_idx]
    instruction = instruction.strip()
    return instruction or None


def _resolve_skill_commands_platform() -> Optional[str]:
    """Return the current platform scope for skill filtering."""
    return os.environ.get("NIA_PLATFORM") or os.environ.get("HERMES_PLATFORM")


def _get_skills_dir() -> Path:
    """Return the NIA skills directory."""
    try:
        from niaharness.prompts.soul import get_nia_home
        return get_nia_home() / "skills"
    except Exception:
        return Path(os.path.expanduser("~/.nia/skills"))


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from SKILL.md content.

    Returns (frontmatter_dict, body_text).
    """
    if not content.startswith("---"):
        return {}, content
    end = content.find("\n---", 3)
    if end < 0:
        return {}, content
    frontmatter_text = content[3:end].strip()
    body = content[end + 4:].strip()
    # Simple YAML parse (key: value per line).
    fm: dict = {}
    for line in frontmatter_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip().strip('"').strip("'")
    return fm, body


def _get_external_skills_dirs() -> list[Path]:
    """Return external skills directories from config."""
    dirs: list[Path] = []
    try:
        from niaharness.tui_gateway.server import _load_cfg
        cfg = _load_cfg()
        ext = cfg.get("skills", {}).get("external_dirs", [])
        if isinstance(ext, list):
            for d in ext:
                p = Path(d).expanduser()
                if p.is_dir():
                    dirs.append(p)
    except Exception:
        pass
    return dirs


def _get_disabled_skill_names() -> set[str]:
    """Return user-disabled skill names from config."""
    try:
        from niaharness.tui_gateway.server import _load_cfg
        cfg = _load_cfg()
        disabled = cfg.get("skills", {}).get("disabled", [])
        if isinstance(disabled, list):
            return {str(n) for n in disabled}
    except Exception:
        pass
    return set()


def _skill_matches_platform(frontmatter: dict) -> bool:
    """Check if a skill matches the current platform."""
    platforms = frontmatter.get("platforms")
    if not platforms:
        return True
    import sys
    current = "macos" if sys.platform == "darwin" else "windows" if sys.platform == "win32" else "linux"
    # Parse [macos] [linux] format.
    if isinstance(platforms, str):
        return current in platforms.lower()
    return True


def _skill_matches_environment(frontmatter: dict) -> bool:
    """Check if a skill matches the current runtime environment."""
    env = frontmatter.get("metadata", {}).get("nia", {}).get("env")
    if not env:
        return True
    return True  # Permissive — offer-time only.


def _load_skill_payload(skill_dir: str, task_id: str | None = None) -> Optional[tuple[dict, Path, str]]:
    """Load a skill's SKILL.md payload from its directory.

    Returns (loaded_skill_dict, skill_dir_path, skill_name) or None.
    """
    skill_path = Path(skill_dir)
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        return None
    try:
        content = skill_md.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(content)
        name = frontmatter.get("name", skill_path.name)
        loaded = {
            "name": name,
            "description": frontmatter.get("description", ""),
            "body": body,
            "frontmatter": frontmatter,
        }
        return loaded, skill_path, name
    except Exception as e:
        logger.debug("Failed to load skill payload from %s: %s", skill_dir, e)
        return None


def _build_skill_message(
    loaded_skill: dict[str, Any],
    skill_dir: Path,
    activation_note: str,
    user_instruction: str = "",
    runtime_note: str = "",
    session_id: str | None = None,
) -> str:
    """Build the full skill invocation message.

    Ported from hermes-agent/agent/skill_commands.py line 245.
    """
    skill_name = loaded_skill.get("name", "")
    body = loaded_skill.get("body", "")
    parts = [activation_note]

    if user_instruction:
        parts.append(f"\n\n{body}")
        parts.append(f"\n\n{_SINGLE_SKILL_INSTRUCTION}{user_instruction}")
    else:
        parts.append(f"\n\n{body}")

    if runtime_note:
        parts.append(f"{_RUNTIME_NOTE} {runtime_note}]")

    return "".join(parts)


def scan_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Scan ~/.nia/skills/ and return a mapping of /command -> skill info.

    Ported from hermes-agent/agent/skill_commands.py line 348.

    Returns:
        Dict mapping "/skill-name" to {name, description, skill_md_path, skill_dir}.
    """
    global _skill_commands, _skill_commands_platform
    _skill_commands_platform = _resolve_skill_commands_platform()
    _skill_commands = {}
    try:
        disabled = _get_disabled_skill_names()
        seen_names: set = set()

        dirs_to_scan: list[Path] = []
        skills_dir = _get_skills_dir()
        if skills_dir.exists():
            dirs_to_scan.append(skills_dir)
        dirs_to_scan.extend(_get_external_skills_dirs())

        for scan_dir in dirs_to_scan:
            for skill_md in scan_dir.rglob("SKILL.md"):
                if any(part in {'.git', '.github', '.hub', '.archive'} for part in skill_md.parts):
                    continue
                try:
                    content = skill_md.read_text(encoding='utf-8')
                    frontmatter, body = _parse_frontmatter(content)
                    if not _skill_matches_platform(frontmatter):
                        continue
                    if not _skill_matches_environment(frontmatter):
                        continue
                    name = frontmatter.get('name', skill_md.parent.name)
                    if name in seen_names:
                        continue
                    if name in disabled:
                        continue
                    description = frontmatter.get('description', '')
                    if not description:
                        for line in body.strip().split('\n'):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                description = line[:80]
                                break
                    seen_names.add(name)
                    cmd_name = name.lower().replace(' ', '-').replace('_', '-')
                    cmd_name = _SKILL_INVALID_CHARS.sub('', cmd_name)
                    cmd_name = _SKILL_MULTI_HYPHEN.sub('-', cmd_name).strip('-')
                    if not cmd_name:
                        continue
                    _skill_commands[f"/{cmd_name}"] = {
                        "name": name,
                        "description": description or f"Invoke the {name} skill",
                        "skill_md_path": str(skill_md),
                        "skill_dir": str(skill_md.parent),
                    }
                except Exception:
                    continue
    except Exception:
        pass
    return _skill_commands


def get_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Return the current skill commands mapping (scan first if empty).

    Ported from hermes-agent/agent/skill_commands.py line 418.
    """
    if (
        not _skill_commands
        or _skill_commands_platform != _resolve_skill_commands_platform()
    ):
        scan_skill_commands()
    return _skill_commands


def reload_skills() -> Dict[str, Any]:
    """Re-scan the skills directory and return a diff of what changed.

    Ported from hermes-agent/agent/skill_commands.py line 433.
    """
    old_keys = set(_skill_commands.keys())
    scan_skill_commands()
    new_keys = set(_skill_commands.keys())
    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    return {
        "added": added,
        "removed": removed,
        "total": len(_skill_commands),
    }


def resolve_skill_command_key(command: str) -> Optional[str]:
    """Resolve a user-typed command to a canonical /skill key.

    Ported from hermes-agent/agent/skill_commands.py line 498.
    """
    if not command:
        return None
    cmd = command.lstrip("/").strip().lower()
    if not cmd:
        return None
    key = f"/{cmd}"
    commands = get_skill_commands()
    if key in commands:
        return key
    # Try with hyphens normalized.
    normalized = _SKILL_INVALID_CHARS.sub("", cmd.replace("_", "-"))
    normalized = _SKILL_MULTI_HYPHEN.sub("-", normalized).strip("-")
    key = f"/{normalized}"
    if key in commands:
        return key
    return None


def build_skill_invocation_message(
    cmd_key: str,
    user_instruction: str = "",
    task_id: str | None = None,
    runtime_note: str = "",
) -> Optional[str]:
    """Build the user message content for a skill slash command invocation.

    Ported from hermes-agent/agent/skill_commands.py line 517.
    """
    commands = get_skill_commands()
    skill_info = commands.get(cmd_key)
    if not skill_info:
        return None

    loaded = _load_skill_payload(skill_info["skill_dir"], task_id=task_id)
    if not loaded:
        return None

    loaded_skill, skill_dir, skill_name = loaded

    activation_note = (
        f'[IMPORTANT: The user has invoked the "{skill_name}" skill, indicating they want '
        "you to follow its instructions. The full skill content is loaded below.]"
    )
    return _build_skill_message(
        loaded_skill,
        skill_dir,
        activation_note,
        user_instruction=user_instruction,
        runtime_note=runtime_note,
        session_id=task_id,
    )


# ---------------------------------------------------------------------------
# Stacked slash-skill invocations
# ---------------------------------------------------------------------------

_MAX_STACKED_SKILLS = 5


def split_stacked_skill_commands(rest: str) -> tuple[list[str], str]:
    """Consume additional leading ``/skill`` tokens from *rest*.

    Ported from hermes-agent/agent/skill_commands.py line 581.
    """
    keys: list[str] = []
    remaining = rest or ""
    while len(keys) < _MAX_STACKED_SKILLS - 1:
        stripped = remaining.lstrip()
        if not stripped.startswith("/"):
            break
        parts = stripped.split(None, 1)
        token = parts[0]
        tail = parts[1] if len(parts) > 1 else ""
        cmd_key = resolve_skill_command_key(token.lstrip("/"))
        if cmd_key is None or cmd_key in keys:
            break
        keys.append(cmd_key)
        remaining = tail
    return keys, remaining.strip()


def build_stacked_skill_invocation_message(
    cmd_keys: list[str],
    user_instruction: str = "",
    task_id: str | None = None,
) -> Optional[str]:
    """Build a message that loads multiple skills + the user instruction.

    Ported from hermes-agent/agent/skill_commands.py line 613.
    """
    if not cmd_keys:
        return None
    commands = get_skill_commands()
    parts: list[str] = []
    skill_names: list[str] = []
    for cmd_key in cmd_keys:
        skill_info = commands.get(cmd_key)
        if not skill_info:
            continue
        loaded = _load_skill_payload(skill_info["skill_dir"], task_id=task_id)
        if not loaded:
            continue
        loaded_skill, skill_dir, skill_name = loaded
        skill_names.append(skill_name)
        body = loaded_skill.get("body", "")
        if len(parts) == 0:
            parts.append(
                f'[IMPORTANT: The user has invoked a skill bundle, loading '
                f'{", ".join(skill_names)}]'
            )
            parts.append(f"\n\n[Loaded as part of the \"{skill_name}\" skill:]\n\n{body}")
        else:
            parts.append(f"\n\n[Loaded as part of the \"{skill_name}\" skill:]\n\n{body}")

    if user_instruction:
        parts.append(f"{_BUNDLE_USER_INSTRUCTION}{user_instruction}")
    else:
        parts.append(f"{_BUNDLE_USER_INSTRUCTION}(no additional instruction — invoke the skill(s))")

    return "".join(parts)


def build_preloaded_skills_prompt(
    skill_names: list[str],
    *,
    task_id: str | None = None,
) -> str:
    """Build a system-prompt section listing pre-loaded skills.

    Ported from hermes-agent/agent/skill_commands.py line 695.
    """
    if not skill_names:
        return ""
    commands = get_skill_commands()
    lines = ["\n## Pre-loaded skills\n"]
    for name in skill_names:
        cmd_key = f"/{name.lower().replace(' ', '-').replace('_', '-')}"
        info = commands.get(cmd_key)
        if info:
            desc = info.get("description", "")
            lines.append(f"- **{info['name']}** (/{cmd_key.lstrip('/')}): {desc}")
    return "\n".join(lines)


__all__ = [
    "extract_user_instruction_from_skill_message",
    "scan_skill_commands",
    "get_skill_commands",
    "reload_skills",
    "resolve_skill_command_key",
    "build_skill_invocation_message",
    "split_stacked_skill_commands",
    "build_stacked_skill_invocation_message",
    "build_preloaded_skills_prompt",
]
