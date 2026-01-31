"""
SkillLoader - OpenClaw-Style File-Based Skill System.

Loads capability definitions from SKILL.md files in the skills/ directory.
Skills are filtered by OS compatibility and injected into the System Prompt.

v3.1 - Operation SkillLoader:
    Initial implementation for dynamic skill loading.

Usage:
    from core.skills import load_skills
    
    skills_block = load_skills()  # Returns formatted text for System Prompt
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from core.logger import setup_logger
from core.context import get_os_context

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
            skills_dir = Path(__file__).resolve().parents[2] / "skills"
        
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
]
