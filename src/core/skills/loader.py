"""
SkillLoader - OpenClaw-Style File-Based Skill System.

Loads capability definitions from SKILL.md files in the skills/ directory.
Skills are filtered by OS compatibility and injected into the System Prompt.

v3.1 - Operation SkillLoader:
    Initial implementation for dynamic skill loading.

Usage:
    from src.core.skills import load_skills
    
    skills_block = load_skills()  # Returns formatted text for System Prompt
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.core.logger import setup_logger
from src.core.context import get_os_context

logger = setup_logger("SkillLoader")


# =============================================================================
# Skill Data Structure
# =============================================================================

class Skill:
    """Represents a loaded skill from SKILL.md."""
    
    def __init__(
        self,
        name: str,
        description: str,
        platforms: List[str],
        instructions: str,
        source_path: Path,
    ):
        self.name = name
        self.description = description
        self.platforms = platforms  # ["windows", "linux", "darwin"]
        self.instructions = instructions
        self.source_path = source_path
    
    def is_compatible(self, current_os: str) -> bool:
        """Check if skill is compatible with current OS."""
        if not self.platforms:
            return True  # No platform restriction = universal
        return current_os in self.platforms or "all" in self.platforms
    
    def format_for_prompt(self) -> str:
        """Format skill for inclusion in System Prompt."""
        return (
            f"[SKILL: {self.name}]\n"
            f"(Description: {self.description})\n"
            f"INSTRUCTIONS:\n{self.instructions}"
        )


# =============================================================================
# YAML Frontmatter Parser (Lightweight - No PyYAML Required)
# =============================================================================

def _parse_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """
    Parse YAML frontmatter from markdown content.
    
    Expects format:
        ---
        name: Skill Name
        description: What this skill does
        platforms: ["windows", "linux"]
        ---
        
        Instructions here...
    
    Returns:
        Tuple of (frontmatter_dict, body_content).
    """
    frontmatter = {}
    body = content
    
    # Check for frontmatter delimiters
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            body = parts[2].strip()
            
            # Parse simple YAML (key: value pairs)
            for line in fm_text.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Handle lists: ["item1", "item2"]
                    if value.startswith("[") and value.endswith("]"):
                        # Parse simple list
                        list_content = value[1:-1]
                        items = []
                        for item in list_content.split(","):
                            item = item.strip().strip('"').strip("'")
                            if item:
                                items.append(item)
                        frontmatter[key] = items
                    else:
                        # Handle quoted strings
                        value = value.strip('"').strip("'")
                        frontmatter[key] = value
    
    return frontmatter, body


# =============================================================================
# SkillLoader Class
# =============================================================================

class SkillLoader:
    """
    Loads and manages skills from SKILL.md files.
    
    Skills are stored in the skills/ directory at project root.
    Each skill is a subdirectory containing a SKILL.md file.
    """
    
    def __init__(self, skills_dir: Optional[Path] = None):
        """
        Initialize SkillLoader.
        
        Args:
            skills_dir: Path to skills directory. Defaults to PROJECT_ROOT/skills/.
        """
        if skills_dir is None:
            # Default: PROJECT_ROOT/skills/
            skills_dir = Path(__file__).resolve().parents[2] / "capabilities"
        
        self.skills_dir = Path(skills_dir)
        self.skills: List[Skill] = []
        self._loaded = False
        
        logger.debug(f"SkillLoader initialized: {self.skills_dir}")
    
    def load_skills(self, directory: Optional[str] = None) -> str:
        """
        Load all compatible skills and return formatted prompt block.
        
        Args:
            directory: Optional override for skills directory.
            
        Returns:
            Formatted string for [DYNAMIC SKILLS] section of System Prompt.
        """
        if directory:
            skills_dir = Path(directory)
        else:
            skills_dir = self.skills_dir
        
        if not skills_dir.exists():
            logger.warning(f"Skills directory not found: {skills_dir}")
            return ""
        
        ctx = get_os_context()
        current_os = ctx.os_name
        
        self.skills = []
        loaded_count = 0
        filtered_count = 0
        
        # Walk through skills directory
        for item in skills_dir.iterdir():
            if item.is_dir():
                skill_file = item / "SKILL.md"
                if skill_file.exists():
                    skill = self._load_skill_file(skill_file)
                    if skill:
                        if skill.is_compatible(current_os):
                            self.skills.append(skill)
                            loaded_count += 1
                            logger.debug(f"Loaded skill: {skill.name}")
                        else:
                            filtered_count += 1
                            logger.debug(f"Filtered skill (OS mismatch): {skill.name}")
        
        self._loaded = True
        logger.info(f"Skills loaded: {loaded_count} active, {filtered_count} filtered (OS: {current_os})")
        
        return self._format_skills_block()
    
    def _load_skill_file(self, path: Path) -> Optional[Skill]:
        """Load a single SKILL.md file."""
        try:
            content = path.read_text(encoding="utf-8")
            frontmatter, body = _parse_frontmatter(content)
            
            name = frontmatter.get("name", path.parent.name.replace("_", " ").title())
            description = frontmatter.get("description", "No description provided.")
            platforms = frontmatter.get("platforms", [])
            
            # Normalize platform names
            if isinstance(platforms, str):
                platforms = [platforms]
            platforms = [p.lower() for p in platforms]
            
            return Skill(
                name=name,
                description=description,
                platforms=platforms,
                instructions=body.strip(),
                source_path=path,
            )
            
        except Exception as e:
            logger.error(f"Failed to load skill from {path}: {e}")
            return None
    
    def _format_skills_block(self) -> str:
        """Format all loaded skills into a prompt block."""
        if not self.skills:
            return ""
        
        lines = ["[DYNAMIC SKILLS]"]
        lines.append(f"The following {len(self.skills)} skills are available:\n")
        
        for skill in self.skills:
            lines.append(skill.format_for_prompt())
            lines.append("")  # Blank line between skills
        
        return "\n".join(lines)
    
    def get_skill_names(self) -> List[str]:
        """Get list of loaded skill names."""
        return [s.name for s in self.skills]
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a specific skill by name."""
        for skill in self.skills:
            if skill.name.lower() == name.lower():
                return skill
        return None


# =============================================================================
# Module-Level Singleton
# =============================================================================

_loader: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    """Get the global SkillLoader singleton."""
    global _loader
    if _loader is None:
        _loader = SkillLoader()
    return _loader


def load_skills(directory: Optional[str] = None) -> str:
    """
    Convenience function to load skills and get formatted prompt block.
    
    Args:
        directory: Optional override for skills directory.
        
    Returns:
        Formatted string for System Prompt injection.
    """
    loader = get_skill_loader()
    return loader.load_skills(directory)


__all__ = [
    "Skill",
    "SkillLoader",
    "get_skill_loader",
    "load_skills",
    "get_skill_source_code",
    "load_docker_skills",
]


# =============================================================================
# Docker Execution Helpers (v5.0 Polyglot Extension)
# =============================================================================

# Permanent built-in skills (git-tracked, syncs to GitHub)
_LIBRARY_SKILLS_DIR = Path(__file__).resolve().parent / "library"

# Learned/ephemeral skills (gitignored, created by Builder Soldier)
_DATA_SKILLS_DIR = Path(__file__).resolve().parents[3] / "data" / "skills"


def _all_skill_dirs() -> list[Path]:
    """Return all skill search directories (library first, then data)."""
    dirs = []
    if _LIBRARY_SKILLS_DIR.exists():
        dirs.append(_LIBRARY_SKILLS_DIR)
    if _DATA_SKILLS_DIR.exists():
        dirs.append(_DATA_SKILLS_DIR)
    return dirs


def _detect_runtime(skill_dir: Path) -> Optional[str]:
    """Detect the runtime from available source files."""
    if (skill_dir / "source.py").exists():
        return "python"
    if (skill_dir / "source.js").exists():
        return "node"
    return None


def _detect_source_file(skill_dir: Path) -> Optional[Path]:
    """Find the source code file in a skill directory."""
    for candidate in ["source.py", "source.js"]:
        path = skill_dir / candidate
        if path.exists():
            return path
    return None


def get_skill_source_code(name: str, skills_dir: Optional[Path] = None) -> Optional[str]:
    """Read the raw source code for a Docker-executable skill.
    
    Searches both library/ (permanent) and data/skills/ (learned).
    An explicit skills_dir overrides the dual-search.
    
    Args:
        name: Skill folder name.
        skills_dir: Override directory.
        
    Returns:
        Source code string, or None if not found.
    """
    if skills_dir:
        search_dirs = [skills_dir]
    else:
        search_dirs = _all_skill_dirs()
    
    for base in search_dirs:
        skill_dir = base / name
        if not skill_dir.is_dir():
            continue
        
        source_file = _detect_source_file(skill_dir)
        if source_file is None:
            continue
        
        try:
            return source_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read skill source {source_file}: {e}")
            continue
    
    return None


def _scan_skill_directory(base: Path, source_type: str) -> dict[str, dict]:
    """Scan a directory for skill.md folders and return dict of skills {name: metadata}."""
    skills = {}
    
    if not base.exists():
        return skills
    
    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        
        # v5.0 Standard: skill.md
        skill_file = item / "skill.md"
        if not skill_file.exists():
            continue
        
        runtime = _detect_runtime(item)
        source_file = _detect_source_file(item)
        
        # Parse skill.md frontmatter
        try:
            content = skill_file.read_text(encoding="utf-8")
            frontmatter, body = _parse_frontmatter(content)
        except Exception:
            frontmatter = {}
            body = ""
        
        # Parse dependencies
        deps = frontmatter.get("dependencies", [])
        if isinstance(deps, str):
            deps = [d.strip() for d in deps.split(",") if d.strip()]
        
        # Parse requires (binary requirements like codex, pi)
        # Check standard root 'requires' AND 'metadata.nia.requires'
        requires = frontmatter.get("requires", [])
        if not requires:
             meta = frontmatter.get("metadata", {})
             if isinstance(meta, dict):
                 nia_meta = meta.get("nia", {})
                 if isinstance(nia_meta, dict):
                     requires = nia_meta.get("requires", [])

        if isinstance(requires, str):
            requires = [r.strip() for r in requires.split(",") if r.strip()]
        
        # Parse boolean fields
        pty = str(frontmatter.get("pty", "false")).lower() in ("true", "1", "yes")
        
        name = frontmatter.get("name", item.name)
        skills[name] = {
            "name": name,
            "emoji": frontmatter.get("emoji", "🔧"),
            "description": frontmatter.get("description", body[:200].strip()),
            "runtime": frontmatter.get("runtime", runtime or "python"),
            "dependencies": deps,
            "requires": requires,
            "source_file": str(source_file) if source_file else None,
            "pty": pty,
            "workdir": frontmatter.get("workdir", "/workspace"),
            "body": body.strip(),
            "source": source_type,
        }
    
    return skills


def load_docker_skills(skills_dir: Optional[Path] = None) -> list[dict]:
    """Load metadata for all Docker-executable skills.
    
    Scans TWO directories with priority:
    1. src/core/skills/library/ — Permanent (High Priority)
    2. data/skills/ — Learned/Ephemeral (Low Priority)
    
    Conflict Resolution: If a skill exists in both, the Library version wins.
    
    Returns:
        List of dicts with skill metadata.
    """
    if skills_dir:
        # Single directory scan (test mode)
        skills_map = _scan_skill_directory(skills_dir, "custom")
        return list(skills_map.values())
    
    # Dual-scan: library (permanent) + data (learned)
    # Scan Data first (Low Priority)
    all_skills_map = {}
    
    data_skills = _scan_skill_directory(_DATA_SKILLS_DIR, "learned")
    all_skills_map.update(data_skills)
    
    # Scan Library second (High Priority - Overwrites Data)
    library_skills = _scan_skill_directory(_LIBRARY_SKILLS_DIR, "library")
    all_skills_map.update(library_skills)
    
    logger.info(
        f"Docker skills loaded: {len(all_skills_map)} total "
        f"(library={len(library_skills)}, data={len(data_skills)})"
    )
    
    # Return sorted list
    return sorted(list(all_skills_map.values()), key=lambda x: x["name"])


def get_skills_prompt(skills_dir: Optional[Path] = None) -> str:
    """Build a formatted Skill Library prompt for the General's LLM."""
    skills = load_docker_skills(skills_dir)
    
    if not skills:
        return ""
    
    lines = ["## 🛠️ Available Skills\n"]
    for s in skills:
        emoji = s.get("emoji", "🔧")
        pty_marker = " ⌨️ (interactive)" if s.get("pty") else ""
        requires_marker = f" [requires: {', '.join(s['requires'])}]" if s.get("requires") else ""
        lines.append(
            f"- {emoji} **{s['name']}** ({s['runtime']}{pty_marker}){requires_marker}: {s['description']}"
        )
    lines.append("")
    return "\n".join(lines)

