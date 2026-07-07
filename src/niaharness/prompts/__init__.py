"""System prompt builder for NiaHarness."""

from niaharness.prompts.claudemd import discover_claude_md_files, load_claude_md_prompt
from niaharness.prompts.context import build_runtime_system_prompt
from niaharness.prompts.soul import (
    DEFAULT_SOUL_MD,
    get_nia_home,
    get_soul_md_path,
    is_default_soul,
    load_soul_md,
)
from niaharness.prompts.system_prompt import build_system_prompt
from niaharness.prompts.environment import get_environment_info

__all__ = [
    "build_runtime_system_prompt",
    "build_system_prompt",
    "discover_claude_md_files",
    "get_environment_info",
    "load_claude_md_prompt",
    # SOUL.md identity system
    "DEFAULT_SOUL_MD",
    "get_nia_home",
    "get_soul_md_path",
    "is_default_soul",
    "load_soul_md",
]
