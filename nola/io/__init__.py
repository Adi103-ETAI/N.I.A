"""N.O.L.A. I/O Package - Hardware Drivers for Audio.

TTS: Microsoft Edge Neural Voices (Primary) with Piper Fallback
STT: Vosk Offline Speech Recognition

This package provides the "Mouth" (TTS) and "Ears" (STT) for N.I.A.

Singleton Pattern:
    - Use get_async_ear() for STT instance
    - Use get_async_tts() for TTS instance

Usage:
    from nola.io import HybridTTS, VoskSTT, get_async_tts, get_async_ear
"""
from __future__ import annotations

# Speech (TTS) - "The Mouth"
from .speech import (
    HybridTTS,
    AsyncTTS,
    get_async_tts,
    TTS_CACHE,
    EDGE_VOICE,
)

# Hearing (STT) - "The Ears"
from .hearing import (
    RecognitionResult,
    VoskSTT,
    AsyncEar,
    get_async_ear,
    VOSK_MODEL_PATH,
)


# =============================================================================
# Exports (Backward Compatible)
# =============================================================================

__all__ = [
    # TTS
    "HybridTTS",
    "AsyncTTS",
    "get_async_tts",
    "TTS_CACHE",
    "EDGE_VOICE",
    # STT
    "RecognitionResult",
    "VoskSTT",
    "AsyncEar",
    "get_async_ear",
    "VOSK_MODEL_PATH",
]
