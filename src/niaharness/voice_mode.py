"""Voice Mode — Push-to-talk audio recording and playback.

Focused extraction from hermes-agent/tools/voice_mode.py (1,219 LOC).

The full Hermes module has audio capture via sounddevice, WAV encoding,
STT dispatch via transcription_tools, and TTS playback. This port provides
the essential ``check_voice_requirements()`` API that ``voice.toggle status``
depends on. In headless environments (no PortAudio), it correctly reports
"not available".
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _is_termux() -> bool:
    """Detect Termux environment."""
    return os.environ.get("TERMUX_VERSION") is not None or "com.termux" in os.environ.get("PREFIX", "")


def _audio_available() -> bool:
    """Return True if audio libraries (sounddevice + numpy) can be imported."""
    try:
        import sounddevice  # noqa: F401
        import numpy  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def _termux_voice_capture_available() -> bool:
    """Check if Termux:API microphone recording is available."""
    if not _is_termux():
        return False
    return shutil.which("termux-microphone-record") is not None


def detect_audio_environment() -> dict:
    """Detect the audio environment and return warnings.

    Ported from hermes-agent/tools/voice_mode.py line 142 (simplified).
    """
    warnings: List[str] = []
    notices: List[str] = []
    available = True

    # Check for headless / SSH environment.
    if not sys.stdin.isatty() and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
            warnings.append("SSH session detected — audio may not be available")
            available = False
        elif os.environ.get("DOCKER_CONTAINER"):
            warnings.append("Docker container detected — audio devices not mounted")
            available = False

    # Check for PortAudio.
    if not _audio_available() and not _termux_voice_capture_available():
        warnings.append("sounddevice/numpy not installed — audio capture unavailable")
        available = False

    return {
        "available": available,
        "warnings": warnings,
        "notices": notices,
    }


def check_voice_requirements() -> Dict[str, Any]:
    """Check if all voice mode requirements are met.

    Ported from hermes-agent/tools/voice_mode.py line 1124.

    Returns a dict with ``available``, ``audio_available``, ``stt_available``,
    ``missing_packages``, and ``details``.
    """
    # Determine STT provider availability.
    stt_available = False
    try:
        from niaharness.tools.transcription_tools import _get_provider, _load_stt_config, is_stt_enabled  # type: ignore
        stt_config = _load_stt_config()
        stt_enabled = is_stt_enabled(stt_config)
        stt_provider = _get_provider(stt_config)
        stt_available = stt_enabled and stt_provider != "none"
    except ImportError:
        # TODO(feature-gap): niaharness.tools.transcription_tools not yet built.
        # Check for common STT env vars.
        stt_available = bool(
            os.environ.get("GROQ_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

    missing: List[str] = []
    termux_capture = _termux_voice_capture_available()
    has_audio = _audio_available() or termux_capture

    if not has_audio:
        missing.extend(["sounddevice", "numpy"])

    env_check = detect_audio_environment()

    available = has_audio and stt_available and env_check["available"]
    details_parts = []

    if termux_capture:
        details_parts.append("Audio capture: OK (Termux:API microphone)")
    elif has_audio:
        details_parts.append("Audio capture: OK")
    else:
        install_hint = (
            "pkg install python-numpy portaudio && python -m pip install sounddevice"
            if _is_termux()
            else "pip install sounddevice numpy"
        )
        details_parts.append(f"Audio capture: MISSING ({install_hint})")

    if not stt_available:
        details_parts.append(
            "STT provider: MISSING (pip install faster-whisper, "
            "or set GROQ_API_KEY / OPENAI_API_KEY for cloud STT)"
        )
    else:
        details_parts.append("STT provider: OK")

    for warning in env_check["warnings"]:
        details_parts.append(f"Environment: {warning}")

    return {
        "available": available,
        "audio_available": has_audio,
        "stt_available": stt_available,
        "missing_packages": missing,
        "details": "\n".join(details_parts),
        "environment": env_check,
    }


def cleanup_temp_recordings(max_age_seconds: int = 3600) -> int:
    """Remove old temporary voice recording files.

    Ported from hermes-agent/tools/voice_mode.py line 1191.
    """
    temp_dir = os.path.join(os.environ.get("TEMP", "/tmp"), "nia_voice")
    if not os.path.isdir(temp_dir):
        return 0

    import time
    deleted = 0
    now = time.time()

    for entry in os.scandir(temp_dir):
        if entry.is_file() and entry.name.startswith("recording_") and entry.name.endswith(".wav"):
            try:
                age = now - entry.stat().st_mtime
                if age > max_age_seconds:
                    os.unlink(entry.path)
                    deleted += 1
            except OSError:
                pass

    return deleted


__all__ = ["check_voice_requirements", "detect_audio_environment", "cleanup_temp_recordings"]
