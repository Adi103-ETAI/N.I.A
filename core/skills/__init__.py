"""
Skills Package - File-Based Skill System.

OpenClaw-style modular skill loading from SKILL.md files.
"""
from .loader import SkillLoader, get_skill_loader, load_skills

__all__ = [
    "SkillLoader",
    "get_skill_loader",
    "load_skills",
]
