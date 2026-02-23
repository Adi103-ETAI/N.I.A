"""N.O.L.A. I/O Subpackage — Speech Output and Hearing Input.

Exports both the concrete drivers and their singleton accessor functions
so the NOLAManager and any other consumers only need to import from here.

Modules:
    speech.py   — ``HybridTTS`` (Edge TTS + Piper fallback) + ``AsyncTTS`` wrapper
    hearing.py  — ``VoskSTT`` (offline Vosk model) + ``AsyncEar`` wrapper
"""

from .speech import HybridTTS, AsyncTTS, get_async_tts
from .hearing import VoskSTT, AsyncEar, RecognitionResult, get_async_ear

__all__ = [
    "RecognitionResult",
    "HybridTTS",
    "VoskSTT",
    "AsyncEar",
    "AsyncTTS",
    "get_async_ear",
    "get_async_tts",
]
