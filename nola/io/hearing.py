"""NOLA Hearing Module - Speech-to-Text Hardware Driver ("The Ears").

STT: Vosk Offline Speech Recognition

Usage:
    from nola.io.hearing import VoskSTT, get_async_ear, AsyncEar
    
    ear = get_async_ear()  # Singleton
    ear.start()
    text = ear.get_text(timeout=5.0)
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional

# Centralized logging
from core.logger import setup_logger
logger = setup_logger("NOLA.HEARING")


# =============================================================================
# Configuration Constants
# =============================================================================

NOLA_DIR = Path(__file__).parent.parent  # nola/io_pkg -> nola/
VOSK_MODEL_PATH = NOLA_DIR / "vosk_model"


# =============================================================================
# Recognition Result Container
# =============================================================================

@dataclass
class RecognitionResult:
    """Container for ASR recognition results."""
    text: str
    confidence: float = 1.0
    timestamp: float = 0.0
    is_final: bool = True
    
    def __bool__(self) -> bool:
        return bool(self.text and self.text.strip())


# =============================================================================
# VoskSTT - Offline Speech Recognition
# =============================================================================

class VoskSTT:
    """Vosk-based offline speech-to-text.
    
    Provides continuous speech recognition via a generator interface.
    
    Usage:
        stt = VoskSTT()
        for text in stt.stream():
            print(f"Heard: {text}")
    """
    
    def __init__(self, device_index: int = None) -> None:
        """Initialize Vosk STT.
        
        Args:
            device_index: Specific microphone device index.
        """
        self._device_index = device_index
        self._is_running = False
        self._stop_event = threading.Event()
        self._model = None
        self._recognizer = None
        
        # Check dependencies
        self._has_vosk = False
        self._has_sounddevice = False
        
        try:
            import vosk
            vosk.SetLogLevel(-1)
            self._vosk_module = vosk
            self._has_vosk = True
        except ImportError:
            logger.error("vosk not installed. Run: pip install vosk")
        
        try:
            import sounddevice
            self._has_sounddevice = True
        except ImportError:
            logger.error("sounddevice not installed. Run: pip install sounddevice")
        
        # Load model
        if self._has_vosk and VOSK_MODEL_PATH.exists():
            try:
                self._model = self._vosk_module.Model(str(VOSK_MODEL_PATH))
                self._recognizer = self._vosk_module.KaldiRecognizer(self._model, 16000)
                logger.info("VoskSTT: Model loaded and ready")
            except OSError as e:
                logger.error(f"Failed to load Vosk model (I/O error): {e}", exc_info=True)
            except RuntimeError as e:
                logger.error(f"Failed to load Vosk model (runtime error): {e}", exc_info=True)
        else:
            if not VOSK_MODEL_PATH.exists():
                logger.warning("Vosk model not found at: %s", VOSK_MODEL_PATH)
    
    def stream(self) -> Generator[str, None, None]:
        """Stream recognized text continuously.
        
        Yields:
            Recognized text strings.
        """
        if not self._has_vosk or not self._has_sounddevice or not self._recognizer:
            logger.error("VoskSTT not ready")
            return
        
        import sounddevice as sd
        
        self._is_running = True
        self._stop_event.clear()
        
        try:
            with sd.RawInputStream(
                samplerate=16000,
                blocksize=8000,
                device=self._device_index,
                dtype='int16',
                channels=1,
            ) as stream:
                logger.info("VoskSTT streaming started")
                
                while self._is_running and not self._stop_event.is_set():
                    try:
                        data, overflowed = stream.read(4000)
                        if len(data) == 0:
                            continue
                        
                        if self._recognizer.AcceptWaveform(bytes(data)):
                            result = json.loads(self._recognizer.Result())
                            text = result.get('text', '').strip()
                            if text:
                                yield text
                    
                    except (OSError, json.JSONDecodeError) as e:
                        logger.debug(f"Stream processing error: {e}")
                        time.sleep(0.1)
                        
        except OSError as e:
            logger.error(f"VoskSTT audio device error: {e}", exc_info=True)
        except RuntimeError as e:
            logger.error(f"VoskSTT stream runtime error: {e}", exc_info=True)
        finally:
            self._is_running = False
    
    def stop(self) -> None:
        """Stop the stream."""
        self._is_running = False
        self._stop_event.set()
    
    @property
    def is_running(self) -> bool:
        """Check if streaming."""
        return self._is_running
    
    @property
    def is_ready(self) -> bool:
        """Check if STT is ready."""
        return self._has_vosk and self._has_sounddevice and self._recognizer is not None


# =============================================================================
# Singleton Instance
# =============================================================================

_stt_instance: Optional[VoskSTT] = None


def get_async_ear() -> VoskSTT:
    """Get or create the STT singleton.
    
    Returns:
        VoskSTT instance.
    """
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = VoskSTT()
    return _stt_instance


# =============================================================================
# Legacy Compatibility Wrapper
# =============================================================================

class AsyncEar:
    """Legacy wrapper for backward compatibility."""
    
    def __init__(self, wake_words: List[str] = None, **kwargs) -> None:
        self._stt = get_async_ear()
        self._wake_words = [w.lower() for w in (wake_words or ["jarvis", "nia", "hey nia"])]
        self._queue: queue.Queue = queue.Queue(maxsize=50)
        self._is_running = False
        self._is_paused = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self) -> bool:
        if self._is_running:
            return True
        
        self._is_running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        return True
    
    def _listen_loop(self) -> None:
        for text in self._stt.stream():
            if not self._is_running:
                break
            if self._is_paused:
                continue
            
            result = RecognitionResult(text=text, timestamp=time.time())
            try:
                self._queue.put_nowait(result)
            except queue.Full:
                pass
    
    def stop(self, timeout: float = 3.0) -> None:
        self._is_running = False
        self._stt.stop()
    
    def pause(self) -> None:
        self._is_paused = True
    
    def resume(self) -> None:
        self._is_paused = False
    
    def get_text(self, timeout: float = None) -> Optional[RecognitionResult]:
        try:
            return self._queue.get(timeout=timeout) if timeout else self._queue.get_nowait()
        except queue.Empty:
            return None
    
    @property
    def is_running(self) -> bool:
        return self._is_running


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "RecognitionResult",
    "VoskSTT",
    "AsyncEar",
    "get_async_ear",
    "VOSK_MODEL_PATH",
]
