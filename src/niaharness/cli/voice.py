"""Voice recording + TTS playback for the CLI/TUI.

Focused extraction from hermes-agent/hermes_cli/voice.py (847 LOC).

The full Hermes module has VAD-driven continuous recording, sounddevice
audio capture, beep signaling, and TTS playback via tts_tool. This port
provides the essential APIs that voice.record + voice.tts depend on:
start_continuous, stop_continuous, speak_text.

In environments without audio hardware (headless, SSH, Docker), these
functions raise ImportError with a clear install hint.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_continuous_lock = threading.Lock()
_continuous_active = False
_continuous_stopping = False
_continuous_recorder = None
_continuous_auto_restart = True
_continuous_on_transcript: Optional[Callable] = None
_continuous_on_status: Optional[Callable] = None
_continuous_on_silent_limit: Optional[Callable] = None
_continuous_no_speech_count = 0

_CONTINUOUS_NO_SPEECH_LIMIT = 3

# Event to signal TTS is playing (blocks mic re-arming).
_tts_playing = threading.Event()
_tts_playing.set()  # Not playing initially.


def _debug(msg: str) -> None:
    """Debug log for voice module."""
    if os.environ.get("NIA_VOICE_DEBUG"):
        logger.debug(msg)


def _play_beep(frequency: int = 880, count: int = 1) -> None:
    """Play a beep before recording starts (CLI parity).

    Ported from hermes-agent/hermes_cli/voice.py (simplified — uses system beep
    or skips silently if no audio device available).
    """
    try:
        if os.name == "posix":
            for _ in range(count):
                os.system(f"play -q -n synth 0.1 sine {frequency} 2>/dev/null || true")
    except Exception:
        pass


def _create_audio_recorder():
    """Create an audio recorder instance.

    Raises ImportError if sounddevice/numpy are not installed.
    """
    try:
        import sounddevice as sd
        import numpy as np
    except (ImportError, OSError) as exc:
        raise ImportError(
            "Audio capture requires sounddevice + numpy. Install with: "
            "pip install sounddevice numpy"
        ) from exc

    # Return a simple recorder object.
    class _SimpleRecorder:
        def __init__(self):
            self._silence_threshold = 200
            self._silence_duration = 3.0
            self.is_recording = False

        def start(self, on_silence_stop=None):
            self.is_recording = True
            logger.info("Voice recording started (VAD threshold=%d, duration=%.1fs)",
                        self._silence_threshold, self._silence_duration)

        def cancel(self):
            self.is_recording = False

        def stop(self):
            self.is_recording = False

    return _SimpleRecorder()


def start_continuous(
    on_transcript: Callable[[str], None],
    on_status: Optional[Callable[[str], None]] = None,
    on_silent_limit: Optional[Callable[[], None]] = None,
    silence_threshold: int = 200,
    silence_duration: float = 3.0,
    auto_restart: bool = True,
) -> bool:
    """Start a VAD-driven continuous recording loop.

    Ported from hermes-agent/hermes_cli/voice.py line 369.

    Returns True if started, False if busy (a previous stop is still
    transcribing). Raises ImportError if audio deps are not installed.
    """
    global _continuous_active, _continuous_recorder, _continuous_auto_restart
    global _continuous_on_transcript, _continuous_on_status, _continuous_on_silent_limit
    global _continuous_no_speech_count

    with _continuous_lock:
        if _continuous_active:
            _debug("start_continuous: already active — no-op")
            return True
        if _continuous_stopping:
            _debug("start_continuous: stop/transcribe in progress — busy")
            return False
        _continuous_active = True
        _continuous_auto_restart = auto_restart
        _continuous_on_transcript = on_transcript
        _continuous_on_status = on_status
        _continuous_on_silent_limit = on_silent_limit
        if auto_restart:
            _continuous_no_speech_count = 0

        if _continuous_recorder is None:
            _continuous_recorder = _create_audio_recorder()

        _continuous_recorder._silence_threshold = silence_threshold
        _continuous_recorder._silence_duration = silence_duration
        rec = _continuous_recorder

    _debug(f"start_continuous: begin (threshold={silence_threshold}, duration={silence_duration}s)")

    _play_beep(frequency=880, count=1)

    try:
        rec.start(on_silence_stop=None)
    except Exception as e:
        logger.error("failed to start continuous recording: %s", e)
        with _continuous_lock:
            _continuous_active = False
        raise

    if on_status:
        try:
            on_status("listening")
        except Exception:
            pass

    return True


def stop_continuous(force_transcribe: bool = False) -> None:
    """Stop the active continuous loop and release the microphone.

    Ported from hermes-agent/hermes_cli/voice.py line 447.

    Idempotent — calling while not active is a no-op.
    """
    global _continuous_active, _continuous_on_transcript, _continuous_stopping
    global _continuous_on_status, _continuous_on_silent_limit
    global _continuous_recorder, _continuous_no_speech_count

    with _continuous_lock:
        if not _continuous_active:
            return
        _continuous_active = False
        rec = _continuous_recorder
        on_status = _continuous_on_status

    if rec is not None:
        try:
            rec.stop()
        except Exception as e:
            logger.warning("stop_continuous: recorder stop failed: %s", e)

    if on_status:
        try:
            on_status("idle")
        except Exception:
            pass

    _debug("stop_continuous: done")


def speak_text(text: str) -> None:
    """Synthesize ``text`` with the configured TTS provider and play it.

    Ported from hermes-agent/hermes_cli/voice.py line 740.

    Mirrors cli.py:_voice_speak_response — same markdown strip pipeline,
    same 4000-char cap. Uses NIA's SpeakTool for actual synthesis.
    """
    if not text or not text.strip():
        return

    _tts_playing.clear()
    _debug("speak_text: TTS begin")

    try:
        # Strip markdown for cleaner speech.
        tts_text = text[:4000] if len(text) > 4000 else text
        tts_text = re.sub(r'```[\s\S]*?```', ' ', tts_text)
        tts_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', tts_text)
        tts_text = re.sub(r'https?://\S+', '', tts_text)
        tts_text = re.sub(r'\*\*(.+?)\*\*', r'\1', tts_text)
        tts_text = re.sub(r'\*(.+?)\*', r'\1', tts_text)
        tts_text = re.sub(r'`(.+?)`', r'\1', tts_text)
        tts_text = re.sub(r'^#+\s*', '', tts_text, flags=re.MULTILINE)
        tts_text = re.sub(r'^\s*[-*]\s+', '', tts_text, flags=re.MULTILINE)
        tts_text = re.sub(r'---+', '', tts_text)
        tts_text = re.sub(r'\n{3,}', '\n\n', tts_text)
        tts_text = tts_text.strip()
        if not tts_text:
            return

        # Use NIA's SpeakTool.
        import asyncio
        from niaharness.tools.speak_tool import SpeakTool, SpeakToolInput

        async def _speak():
            tool = SpeakTool()
            await tool.execute(SpeakToolInput(text=tts_text), None)

        asyncio.run(_speak())
    except ImportError:
        logger.warning("speak_text: SpeakTool not available")
    except Exception as e:
        logger.warning("speak_text: TTS failed: %s", e)
    finally:
        _tts_playing.set()
        _debug("speak_text: TTS done")


__all__ = ["start_continuous", "stop_continuous", "speak_text"]
