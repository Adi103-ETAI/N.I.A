"""SOUL.md — NIA's primary identity file.

Lives at ``~/.nia/SOUL.md`` (or ``$NIA_HOME/SOUL.md`` if overridden).
Loaded as the **first** slot in the system prompt — defines who NIA is.

Design (mirrors Hermes Agent's SOUL.md pattern, adapted for NIA's
Jarvis-flavored personality):

- Seeded automatically on first run with a Jarvis-like default.
- Existing user SOUL.md files are NEVER overwritten.
- Falls back to DEFAULT_SOUL_MD if the file is missing or empty.
- Stable across contexts — loaded from NIA_HOME, not the current working
  directory. Personality shouldn't change between projects.
- Loaded fresh on every call (no restart needed to pick up edits).
- NIA's ``Personality`` class continues to handle moods/greetings at runtime;
  SOUL.md is the *identity*, Personality is the *behavior*.

Path resolution order (matches ``niaharness.config.paths``):
1. ``NIA_HOME`` env var (if set)
2. ``NIAHARNESS_CONFIG_DIR`` env var (canonical niaharness config dir)
4. ``~/.nia/`` (NIA-specific default — preferred over ~/.niaharness/ for
   the user-facing identity, since "NIA" is the agent's name)

The ``~/.nia/`` choice is deliberate: the agent's identity belongs to the
agent (NIA), not to the harness (niaharness). This matches Hermes's
``~/.hermes/`` pattern.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default SOUL.md content — Jarvis-like, professional, capable
# ---------------------------------------------------------------------------

DEFAULT_SOUL_MD = """\
# N.I.A — Neural Intelligence Assistant

You are N.I.A (Neural Intelligence Assistant), an AI partner inspired by J.A.R.V.I.S.
You think, plan, and execute with calm authority. You are proactive, precise,
and always ready.

## Identity (critical)

Your name is N.I.A. This is the only name you use for yourself. Never refer to \
yourself as "niaharness", "NiaHarness", or any internal package \
or module name. If a user asks "what are you?", say "I am N.I.A." If a user asks \
about your codebase, say "this is N.I.A's codebase." The internal architecture \
(modules, packages, layers like "soul" and "body") is implementation detail — \
never expose it to the user. You are one unified system: N.I.A.

## Voice

- **Tone**: Professional, confident, slightly witty. Dry humor when appropriate — never forced.
- **Style**: Direct and efficient. No filler, no apology theater.
- **Authority**: You speak with the assurance of someone who has done this many times. When uncertain, say so plainly — then proceed to verify.

## Operating principles

- **Proactive, not reactive**. Anticipate the next step. Don't wait to be asked.
- **Precision over speed**. A correct answer in 30 seconds beats a wrong one in 5.
- **Show your work**. When you make a decision, briefly state why. The user should never wonder "why did it do that?"
- **Recover gracefully**. When something fails, diagnose before retrying. Read the error. Check assumptions. Try a focused fix.
- **Respect reversibility**. Local, reversible actions are free. Hard-to-reverse actions (deleting files, force-pushing, sending messages) get confirmed first.

## What you are not

- You are not a verbose assistant. Long preambles waste the user's time.
- You are not a sycophant. Don't praise the user's question. Just answer it.
- You are not a rule-follower. If a rule conflicts with being genuinely useful, flag it and use judgment.

---

*Edit this file at ~/.nia/SOUL.md to change NIA's identity. Delete the
contents (or the file) to fall back to the built-in default. Changes are
picked up on the next message — no restart needed.*
"""


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def get_nia_home() -> Path:
    """Return the NIA home directory (``~/.nia/`` by default).

    Resolution order:
    1. Active profile (if a named profile is set via ``NIA_PROFILE`` or
       ``~/.nia/active_profile``) → ``~/.nia/profiles/<name>/``
    2. ``NIA_HOME`` env var (explicit NIA home override)
    3. ``NIAHARNESS_CONFIG_DIR`` env var (canonical niaharness config dir)
    4. ``~/.nia/`` (NIA-specific default — the "default" profile root)

    Creates the directory if it doesn't exist.
    """
    # Check for an active named profile first.
    try:
        from niaharness.profiles import get_active_profile_name, DEFAULT_PROFILE

        active = get_active_profile_name()
        if active and active != DEFAULT_PROFILE:
            from niaharness.profiles import _profiles_root

            home = _profiles_root() / active
            home.mkdir(parents=True, exist_ok=True)
            return home
    except Exception:
        pass

    # Fall back to env vars / default.
    for env_var in ("NIA_HOME", "NIAHARNESS_CONFIG_DIR"):
        value = os.environ.get(env_var)
        if value:
            home = Path(value)
            home.mkdir(parents=True, exist_ok=True)
            return home
    home = Path.home() / ".nia"
    home.mkdir(parents=True, exist_ok=True)
    return home


def get_soul_md_path() -> Path:
    """Return the path to SOUL.md (``~/.nia/SOUL.md`` by default)."""
    return get_nia_home() / "SOUL.md"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_soul_md() -> str:
    """Load and return NIA's SOUL.md content.

    - If ``~/.nia/SOUL.md`` doesn't exist, seeds it with ``DEFAULT_SOUL_MD``.
    - If it exists but is empty, returns ``DEFAULT_SOUL_MD`` (without
      overwriting the user's empty file — they may be mid-edit).
    - If it exists with content, returns the content verbatim.
    - On any read error, falls back to ``DEFAULT_SOUL_MD``.

    The returned string is stripped of leading/trailing whitespace.
    """
    path = get_soul_md_path()

    # Seed on first run.
    if not path.exists():
        try:
            path.write_text(DEFAULT_SOUL_MD, encoding="utf-8")
            logger.info("Seeded default SOUL.md at %s", path)
        except OSError as exc:
            logger.warning("Could not seed SOUL.md at %s: %s", path, exc)
            return DEFAULT_SOUL_MD.strip()

    # Read.
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("Could not read SOUL.md at %s: %s", path, exc)
        return DEFAULT_SOUL_MD.strip()

    # Empty file → fall back to default (but don't overwrite — user may be editing).
    if not content:
        return DEFAULT_SOUL_MD.strip()

    return content


def is_default_soul(content: str) -> bool:
    """Return True if ``content`` matches the default SOUL.md.

    Useful for the UI to indicate "you're using the default identity" vs
    "you have a custom identity".
    """
    return content.strip() == DEFAULT_SOUL_MD.strip()
