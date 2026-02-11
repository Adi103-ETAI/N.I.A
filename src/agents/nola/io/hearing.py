"""N.O.L.A. Hearing Input - VoskSTT (Offline Speech-to-Text).

Provides speech recognition using Vosk for fully offline operation.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

from src.core.logger import setup_logger

logger = setup_logger("NOLA_STT")

# Optional dependencies
try:
    import vosk
    _HAS_VOSK = True
except ImportError:
    _HAS_VOSK = False
    logger.warning("vosk not installed. STT will not be available.")

try:
    import sounddevice as sd
    _HAS_SD = True
except (ImportError, Exception):
    _HAS_SD = False
    logger.warning("sounddevice not available.")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RecognitionResult:
    """Result from speech recognition."""
    text: str
    confidence: float = 0.0
    is_partial: bool = False


# =============================================================================
# VoskSTT - Offline Speech Recognition
# =============================================================================

class VoskSTT:
    """Offline speech-to-text using Vosk.
    
    Uses a local Vosk model for privacy-first speech recognition.
    No data is sent to any server.
    """

    def __init__(self, model_path: Optional[str] = None, sample_rate: int = 16000):
        """Initialize Vosk STT engine.
        
        Args:
            model_path: Path to Vosk model directory.
            sample_rate: Audio sample rate in Hz.
        """
        self._sample_rate = sample_rate
        self._is_running = False
        self._stop_event = threading.Event()
        self._audio_queue: queue.Queue = queue.Queue()
        
        # Find model path
        if model_path is None:
            model_path = str(Path(__file__).parent.parent / "vosk_model")
        
        self._model = None
        self._recognizer = None
        
        if _HAS_VOSK and os.path.exists(model_path):
            try:
                vosk.SetLogLevel(-1)  # Suppress vosk logs
                self._model = vosk.Model(model_path)
                self._recognizer = vosk.KaldiRecognizer(self._model, sample_rate)
                logger.info(f"Vosk model loaded from: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load Vosk model: {e}")
        else:
            logger.warning(f"Vosk model not found at: {model_path}")

    def stream(self) -> Generator[str, None, None]:
        """Stream recognized speech as text.
        
        Yields:
            Recognized text strings.
        """
        if not _HAS_VOSK or not _HAS_SD or not self._model:
            logger.error("STT not available (missing vosk/sounddevice/model)")
            return

        self._is_running = True
        self._stop_event.clear()

        def audio_callback(indata, frames, time_info, status):
            """Callback for sounddevice stream."""
            if status:
                logger.debug(f"Audio status: {status}")
            self._audio_queue.put(bytes(indata))

        try:
            with sd.RawInputStream(
                samplerate=self._sample_rate,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=audio_callback,
            ):
                logger.debug("Microphone stream opened")
                
                while self._is_running and not self._stop_event.is_set():
                    try:
                        data = self._audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    if self._recognizer.AcceptWaveform(data):
                        result = json.loads(self._recognizer.Result())
                        text = result.get("text", "").strip()
                        if text:
                            yield text
                    else:
                        # Partial result (optional)
                        partial = json.loads(self._recognizer.PartialResult())
                        partial_text = partial.get("partial", "").strip()
                        if partial_text:
                            logger.debug(f"Partial: {partial_text}")

        except OSError as e:
            logger.error(f"Audio device error: {e}")
            raise
        except Exception as e:
            logger.error(f"STT stream error: {e}")
            raise
        finally:
            self._is_running = False
            logger.debug("Microphone stream closed")

    def stop(self) -> None:
        """Stop the STT stream."""
        self._is_running = False
        self._stop_event.set()

    @property
    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self._is_running


# =============================================================================
# Async Ear Wrapper
# =============================================================================

class AsyncEar:
    """Async wrapper around VoskSTT for thread-safe usage."""

    def __init__(self):
        self._stt: Optional[VoskSTT] = None

    def _get_stt(self) -> VoskSTT:
        if self._stt is None:
            self._stt = VoskSTT()
        return self._stt

    def stream(self) -> Generator[str, None, None]:
        """Stream recognized speech."""
        return self._get_stt().stream()

    def stop(self) -> None:
        """Stop listening."""
        if self._stt:
            self._stt.stop()

    @property
    def is_listening(self) -> bool:
        """Check if listening."""
        if self._stt:
            return self._stt.is_listening
        return False


# =============================================================================
# Singleton Accessor
# =============================================================================

_async_ear: Optional[AsyncEar] = None


def get_async_ear() -> AsyncEar:
    """Get the global AsyncEar singleton.
    
    Returns:
        The AsyncEar instance.
    """
    global _async_ear
    if _async_ear is None:
        _async_ear = AsyncEar()
    return _async_ear


__all__ = [
    "RecognitionResult",
    "VoskSTT",
    "AsyncEar",
    "get_async_ear",
]
