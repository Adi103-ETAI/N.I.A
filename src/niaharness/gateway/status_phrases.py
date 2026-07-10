"""P1 Gateway status phrases — human-friendly "still working" indicators.

Ported from Hermes Agent's ``gateway/status_phrases.py`` (228 LOC), scoped
to NIA's architecture. Provides short, generic status phrases for
long-running gateway operations — "still on it", "one sec", "checking
that now".

These phrases are shown in chat when the agent is working on a long
task (e.g. running a build, waiting for a tool result) so the user
knows the agent hasn't died. They deliberately avoid relaying raw model
scratch text — only configured phrase strings are used.

Configuration:
  - Built-in defaults (hardcoded below).
  - User overrides via ``~/.nia/status_phrases.yaml`` or
    ``~/.nia/status_phrases/*.yaml``.
  - Config: ``display.status_phrases`` in config.yaml.

Only configured phrase strings are used; raw tool args, commands,
previews, and reasoning text are never interpolated into the returned
phrase.
"""

from __future__ import annotations

import logging
import random as _random
from collections.abc import Mapping, MutableSequence
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# These are NIA UI surfaces, not app/vendor/domain buckets. Keep this
# long-running-only: regular tool/thinking/interim chatter is intentionally
# not rewritten into generic placeholders because that gets noisy fast in chat.
_STATUS_SURFACES = ("status", "generic")
_MAX_CUSTOM_PHRASES_PER_SURFACE = 80
_MAX_PHRASE_CHARS = 160
_CONVENTIONAL_RELATIVE_PATHS = ("status_phrases.yaml", "status_phrases")

_FALLBACK_PHRASES: dict[str, list[str]] = {
    "status": [
        "still on it",
        "still working through it",
        "waiting for the result",
        "thinking...",
        "almost done",
    ],
    "generic": [
        "on it",
        "one sec",
        "checking that now",
        "let me see",
        "got it",
    ],
}


def _get_nia_home() -> Path:
    """Return NIA_HOME (default ~/.nia)."""
    try:
        from niaharness.config.paths import get_nia_home
        return Path(get_nia_home())
    except Exception:
        import os
        return Path(os.environ.get("NIA_HOME", os.path.expanduser("~/.nia")))


def _clean_phrase_list(value: Any) -> list[str]:
    """Sanitize a list of phrases: strip, dedupe, cap length."""
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value[:_MAX_CUSTOM_PHRASES_PER_SURFACE]:
        phrase = str(item or "").strip()
        if not phrase or len(phrase) > _MAX_PHRASE_CHARS or phrase in seen:
            continue
        cleaned.append(phrase)
        seen.add(phrase)
    return cleaned


def _merge_phrase_mapping(
    catalog: dict[str, list[str]],
    section: Mapping[str, Any],
    *,
    inherited_mode: str | None = None,
) -> None:
    """Merge a phrase mapping section into the catalog."""
    mode = str(section.get("mode") or inherited_mode or "append").strip().lower()
    replace = mode == "replace"
    phrase_map = (
        section.get("phrases")
        if isinstance(section.get("phrases"), Mapping)
        else section
    )
    for surface in _STATUS_SURFACES:
        phrases = (
            _clean_phrase_list(phrase_map.get(surface))
            if isinstance(phrase_map, Mapping)
            else []
        )
        if not phrases:
            continue
        catalog[surface] = (
            phrases if replace else [*catalog.get(surface, []), *phrases]
        )


def _merge_phrase_file(
    catalog: dict[str, list[str]],
    path: Path,
    *,
    inherited_mode: str | None = None,
) -> None:
    """Load a YAML phrase file and merge it into the catalog."""
    try:
        import yaml
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Could not load phrase file %s: %s", path, exc)
        return
    if isinstance(loaded, Mapping):
        _merge_phrase_mapping(catalog, loaded, inherited_mode=inherited_mode)


def _relative_path_under(base_dir: Path, raw_path: Any) -> Path | None:
    """Resolve raw_path relative to base_dir, rejecting absolute + .. escapes."""
    raw = str(raw_path or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    base = base_dir.resolve()
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        return None
    return resolved


def _iter_phrase_files(path: Path) -> list[Path]:
    """Return YAML files at path (single file or directory)."""
    if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
        return [path]
    if path.is_dir():
        return sorted(
            child for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in {".yaml", ".yml"}
        )
    return []


def _merge_phrase_paths(
    catalog: dict[str, list[str]],
    paths: Any,
    *,
    base_dir: Path,
    inherited_mode: str | None = None,
) -> None:
    """Merge phrase paths (file or directory) into the catalog."""
    if paths is None:
        return
    raw_paths = paths if isinstance(paths, list) else [paths]
    for raw_path in raw_paths:
        resolved = _relative_path_under(base_dir, raw_path)
        if resolved is None:
            continue
        for phrase_file in _iter_phrase_files(resolved):
            _merge_phrase_file(catalog, phrase_file, inherited_mode=inherited_mode)


def _load_builtin_catalog() -> dict[str, list[str]]:
    """Load the built-in fallback phrases."""
    return {surface: list(phrases) for surface, phrases in _FALLBACK_PHRASES.items()}


_DEFAULT_PHRASES: dict[str, list[str]] = _load_builtin_catalog()


def _copy_default_catalog() -> dict[str, list[str]]:
    return {surface: list(phrases) for surface, phrases in _DEFAULT_PHRASES.items()}


def _merge_phrase_config(
    catalog: dict[str, list[str]],
    section: Any,
    *,
    base_dir: Path | None = None,
) -> None:
    """Merge one display.status_phrases-style section into the catalog."""
    if not isinstance(section, Mapping):
        return
    mode = str(section.get("mode") or "append").strip().lower()
    if base_dir is not None:
        _merge_phrase_paths(
            catalog, section.get("path"), base_dir=base_dir, inherited_mode=mode
        )
        _merge_phrase_paths(
            catalog, section.get("paths"), base_dir=base_dir, inherited_mode=mode
        )
    _merge_phrase_mapping(catalog, section)


def resolve_status_phrase_catalog(
    user_config: Mapping[str, Any] | None,
    platform_key: str | None = None,
) -> dict[str, list[str]]:
    """Resolve built-in + user-configured generic status phrases.

    Resolution order: built-ins → conventional profile-relative user
    files → global ``display.status_phrases`` →
    ``display.platforms.<platform>.status_phrases``.
    """
    catalog = _copy_default_catalog()
    nia_home = _get_nia_home()
    _merge_phrase_paths(
        catalog, list(_CONVENTIONAL_RELATIVE_PATHS), base_dir=nia_home
    )

    display = (
        (user_config or {}).get("display")
        if isinstance(user_config, Mapping)
        else None
    )
    if not isinstance(display, Mapping):
        return catalog

    _merge_phrase_config(
        catalog, display.get("generic_status_phrases"), base_dir=nia_home
    )
    _merge_phrase_config(
        catalog, display.get("status_phrases"), base_dir=nia_home
    )

    platforms = display.get("platforms")
    if platform_key and isinstance(platforms, Mapping):
        platform_display = platforms.get(platform_key)
        if isinstance(platform_display, Mapping):
            _merge_phrase_config(
                catalog,
                platform_display.get("generic_status_phrases"),
                base_dir=nia_home,
            )
            _merge_phrase_config(
                catalog,
                platform_display.get("status_phrases"),
                base_dir=nia_home,
            )
    return catalog


def classify_status_context(
    kind: str,
    *,
    tool_name: str | None = None,
    preview: str | None = None,
    args: Any = None,
) -> str:
    """Classify an internal gateway event into a NIA UI-surface bucket."""
    normalized = str(kind or "").strip().lower()
    if normalized in {"heartbeat", "waiting", "long_running", "status"}:
        return "status"
    return "generic"


def choose_status_phrase(
    kind: str,
    *,
    tool_name: str | None = None,
    preview: str | None = None,
    args: Any = None,
    recent: MutableSequence[str] | None = None,
    rng: Any = None,
    catalog: Mapping[str, list[str]] | None = None,
) -> str:
    """Pick a short generic status phrase, avoiding recent repeats.

    ``preview`` and ``args`` are accepted for callback compatibility, but
    their raw contents are never embedded in the returned phrase.
    """
    phrase_catalog = catalog or _DEFAULT_PHRASES
    category = classify_status_context(
        kind, tool_name=tool_name, preview=preview, args=args
    )
    candidates = list(
        phrase_catalog.get(category)
        or phrase_catalog.get("generic")
        or _DEFAULT_PHRASES["generic"]
    )
    if recent:
        recent_set = set(recent)
        fresh = [phrase for phrase in candidates if phrase not in recent_set]
        if fresh:
            candidates = fresh
    picker = rng or _random
    phrase = picker.choice(candidates)
    if recent is not None:
        recent.append(phrase)
        del recent[:-6]
    return phrase


__all__ = [
    "choose_status_phrase",
    "classify_status_context",
    "resolve_status_phrase_catalog",
]
