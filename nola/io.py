"""N.O.L.A. I/O Module - Async Audio Input/Output Handling.

This module provides NOLA-controlled async wrappers for audio I/O,
abstracting the underlying plugin implementations.

Classes:
    RecognitionResult: Container for ASR results
    AsyncEar: Non-blocking microphone listener
    AsyncTTS: Non-blocking text-to-speech engine

The classes here wrap the raw plugin drivers from plugins/ and provide
a consistent interface for the NOLAManager to use.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# Configure module logger
logger = logging.getLogger(__name__)


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
        """Return True if text is non-empty."""
        return bool(self.text)


# =============================================================================
# AsyncEar - Non-blocking Microphone Listener
# =============================================================================

class AsyncEar:
    """Thread-safe, non-blocking ASR engine using speech_recognition.

    The microphone listener runs in a background thread, continuously
    capturing audio and pushing recognized text to an output queue.

    This class wraps the speech_recognition library and provides
    NOLA-specific features like pause/resume for echo cancellation.

    Usage:
        ear = AsyncEar()
        ear.start()
        
        result = ear.get_text(timeout=0.1)
        if result:
            print(f"Heard: {result.text}")
        
        ear.pause()   # While TTS speaking
        ear.resume()  # After TTS done
        ear.stop()    # Cleanup
    """

    def __init__(
        self,
        max_queue_size: int = 50,
        energy_threshold: Optional[int] = None,
        pause_threshold: float = 0.8,
        phrase_time_limit: Optional[float] = 10.0,
        recognizer_engine: str = "google",
        device_index: Optional[int] = None,
    ) -> None:
        """Initialize the async ASR system.

        Args:
            max_queue_size: Maximum pending recognitions before oldest dropped.
            energy_threshold: Microphone energy threshold (None for auto).
            pause_threshold: Seconds of silence to mark end of phrase.
            phrase_time_limit: Max seconds for a single phrase.
            recognizer_engine: Recognition backend ('google', 'sphinx', 'whisper').
            device_index: Specific microphone device index.
        """
        self._queue: queue.Queue[RecognitionResult] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()

        # Configuration
        self._energy_threshold = energy_threshold
        self._pause_threshold = pause_threshold
        self._phrase_time_limit = phrase_time_limit
        self._recognizer_engine = recognizer_engine.lower()
        self._device_index = device_index

        # State tracking
        self._is_running = False
        self._is_paused = False
        self._has_sr = False
        self._sr_module = None
        self._recognizer = None
        self._microphone = None
        self._stop_listening_fn: Optional[Callable] = None

        # Statistics
        self._total_recognitions = 0
        self._failed_recognitions = 0

        # Check if speech_recognition is available
        try:
            import speech_recognition as sr  # type: ignore
            self._sr_module = sr
            self._has_sr = True
            logger.debug("speech_recognition module available")
        except ImportError:
            logger.warning("speech_recognition not installed; ASR unavailable")

    def start(self) -> bool:
        """Start the background listening thread.
        
        Returns:
            True if started successfully.
        """
        with self._lock:
            if self._is_running:
                logger.debug("AsyncEar already running")
                return True

            if not self._has_sr:
                logger.error("Cannot start: speech_recognition not available")
                return False

            try:
                self._initialize_recognizer()
                self._start_background_listener()
                self._is_running = True
                self._pause_event.clear()
                self._stop_event.clear()
                logger.info("AsyncEar started")
                return True
            except Exception as exc:
                logger.exception("Failed to start AsyncEar: %s", exc)
                return False

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the background listening thread gracefully."""
        with self._lock:
            if not self._is_running:
                return

            logger.info("Stopping AsyncEar...")
            self._stop_event.set()

            if self._stop_listening_fn:
                try:
                    self._stop_listening_fn(wait_for_stop=False)
                except Exception as exc:
                    logger.debug("Error stopping listener: %s", exc)

        with self._lock:
            self._is_running = False
            self._stop_listening_fn = None
            self._recognizer = None
            self._microphone = None

        logger.info("AsyncEar stopped (recognized: %d, failed: %d)",
                   self._total_recognitions, self._failed_recognitions)

    def pause(self) -> None:
        """Temporarily pause recognition."""
        self._pause_event.set()
        self._is_paused = True
        logger.debug("AsyncEar paused")

    def resume(self) -> None:
        """Resume recognition after pause."""
        self._pause_event.clear()
        self._is_paused = False
        logger.debug("AsyncEar resumed")

    def get_text(self, timeout: Optional[float] = None) -> Optional[RecognitionResult]:
        """Get the next recognized text (non-blocking by default).

        Args:
            timeout: Seconds to wait. None/0 for non-blocking.

        Returns:
            RecognitionResult if available, None otherwise.
        """
        try:
            if timeout is None or timeout <= 0:
                return self._queue.get_nowait()
            return self._queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def get_all_pending(self) -> List[RecognitionResult]:
        """Get all pending recognized text."""
        results = []
        while True:
            try:
                results.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return results

    def clear_queue(self) -> int:
        """Clear all pending recognitions."""
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        return cleared

    def is_listening(self) -> bool:
        """Check if actively listening."""
        return self._is_running and not self._is_paused

    def has_pending(self) -> bool:
        """Check if there are pending recognitions."""
        return not self._queue.empty()

    @property
    def is_running(self) -> bool:
        """Check if ear is running."""
        return self._is_running

    @property
    def is_paused(self) -> bool:
        """Check if ear is paused."""
        return self._is_paused

    @property
    def stats(self) -> Dict[str, int]:
        """Get recognition statistics."""
        return {
            "total_recognitions": self._total_recognitions,
            "failed_recognitions": self._failed_recognitions,
            "pending_in_queue": self._queue.qsize(),
        }

    def _initialize_recognizer(self) -> None:
        """Initialize the speech recognizer and microphone."""
        sr = self._sr_module
        
        self._recognizer = sr.Recognizer()
        
        if self._energy_threshold is not None:
            self._recognizer.energy_threshold = self._energy_threshold
            self._recognizer.dynamic_energy_threshold = False
        else:
            self._recognizer.dynamic_energy_threshold = True
            
        self._recognizer.pause_threshold = self._pause_threshold

        try:
            self._microphone = sr.Microphone(device_index=self._device_index)
            
            with self._microphone as source:
                logger.debug("Calibrating for ambient noise...")
                self._recognizer.adjust_for_ambient_noise(source, duration=1.0)
                logger.debug("Calibration complete (threshold: %d)",
                           self._recognizer.energy_threshold)
        except Exception as exc:
            logger.error("Failed to initialize microphone: %s", exc)
            raise

    def _start_background_listener(self) -> None:
        """Start the background listener."""
        def audio_callback(recognizer, audio):
            if self._stop_event.is_set():
                return
                
            if self._pause_event.is_set():
                return

            try:
                text = self._recognize_audio(recognizer, audio)
                
                if text:
                    result = RecognitionResult(
                        text=text,
                        timestamp=time.time(),
                    )
                    
                    try:
                        self._queue.put_nowait(result)
                        self._total_recognitions += 1
                        logger.debug("Recognized: %s", text[:50])
                    except queue.Full:
                        try:
                            self._queue.get_nowait()
                            self._queue.put_nowait(result)
                        except queue.Empty:
                            pass

            except Exception as exc:
                self._failed_recognitions += 1
                logger.debug("Recognition failed: %s", exc)

        self._stop_listening_fn = self._recognizer.listen_in_background(
            self._microphone,
            audio_callback,
            phrase_time_limit=self._phrase_time_limit,
        )

    def _recognize_audio(self, recognizer, audio) -> Optional[str]:
        """Perform speech recognition on audio data."""
        sr = self._sr_module
        
        try:
            if self._recognizer_engine == "google":
                return recognizer.recognize_google(audio)
            elif self._recognizer_engine == "sphinx":
                return recognizer.recognize_sphinx(audio)
            elif self._recognizer_engine == "whisper":
                return recognizer.recognize_whisper(audio)
            else:
                return recognizer.recognize_google(audio)
                
        except sr.UnknownValueError:
            return None
        except sr.RequestError as exc:
            logger.warning("Recognition API error: %s", exc)
            raise


# =============================================================================
# AsyncTTS - Non-blocking Text-to-Speech
# =============================================================================

class AsyncTTS:
    """Thread-safe, non-blocking TTS engine using pyttsx3.

    The engine runs in a dedicated daemon thread, consuming text from a queue.
    This ensures the main thread is never blocked by speech synthesis.

    Usage:
        tts = AsyncTTS()
        tts.start()
        tts.speak("Hello!")  # Returns immediately
        tts.stop()  # Graceful shutdown
    """

    _STOP_SENTINEL = object()

    def __init__(
        self,
        max_queue_size: int = 100,
        engine_restart_delay: float = 1.0,
        max_restart_attempts: int = 3,
    ) -> None:
        """Initialize the async TTS system.

        Args:
            max_queue_size: Maximum pending messages before blocking.
            engine_restart_delay: Seconds before engine restart on crash.
            max_restart_attempts: Max restarts before fallback mode.
        """
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._engine_ready = threading.Event()
        self._lock = threading.Lock()

        self._restart_delay = engine_restart_delay
        self._max_restarts = max_restart_attempts

        self._is_running = False
        self._is_speaking = False
        self._has_pyttsx3 = False
        self._consecutive_failures = 0

        try:
            import pyttsx3  # type: ignore
            self._has_pyttsx3 = True
        except ImportError:
            logger.warning("pyttsx3 not installed; TTS will use fallback")

    def start(self) -> None:
        """Start the TTS worker thread."""
        with self._lock:
            if self._is_running:
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="AsyncTTS-Worker",
                daemon=True,
            )
            self._is_running = True
            self._thread.start()

            if self._has_pyttsx3:
                self._engine_ready.wait(timeout=5.0)

            logger.info("AsyncTTS started")

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop the TTS worker thread."""
        with self._lock:
            if not self._is_running:
                return

            logger.info("Stopping AsyncTTS...")
            self._stop_event.set()

            try:
                self._queue.put_nowait(self._STOP_SENTINEL)
            except queue.Full:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        with self._lock:
            self._is_running = False
            self._thread = None
            self._engine_ready.clear()

        logger.info("AsyncTTS stopped")

    def speak(self, text: str) -> Dict[str, Any]:
        """Queue text for speech synthesis (non-blocking).

        Args:
            text: Text to speak.

        Returns:
            Dict with 'ok' status.
        """
        if not text or not text.strip():
            return {"ok": False, "error": "No text provided"}

        if not self._is_running:
            self.start()

        try:
            self._queue.put(text.strip(), block=True, timeout=1.0)
            return {"ok": True, "queued": True}
        except queue.Full:
            logger.warning("TTS queue full")
            return {"ok": False, "error": "TTS queue full"}

    def is_speaking(self) -> bool:
        """Check if there are pending messages."""
        return not self._queue.empty() or self._is_speaking

    def clear_queue(self) -> int:
        """Clear all pending messages."""
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        return cleared

    @property
    def is_running(self) -> bool:
        """Check if TTS is running."""
        return self._is_running

    def _worker_loop(self) -> None:
        """Main worker loop."""
        engine = None

        while not self._stop_event.is_set():
            if engine is None and self._has_pyttsx3:
                engine = self._init_engine()
                if engine:
                    self._engine_ready.set()
                    self._consecutive_failures = 0

            try:
                try:
                    item = self._queue.get(block=True, timeout=0.5)
                except queue.Empty:
                    continue

                if item is self._STOP_SENTINEL:
                    break

                text = str(item)
                self._is_speaking = True

                if engine is not None:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                        logger.debug("Spoke: %s", text[:50])
                    except Exception as exc:
                        logger.error("pyttsx3 failed: %s", exc)
                        self._consecutive_failures += 1

                        if self._consecutive_failures <= self._max_restarts:
                            engine = self._cleanup_engine(engine)
                            time.sleep(self._restart_delay)
                        else:
                            engine = self._cleanup_engine(engine)
                            self._has_pyttsx3 = False
                else:
                    print(f"[TTS fallback] {text}", file=sys.stderr)

                self._is_speaking = False
                self._queue.task_done()

            except Exception as exc:
                logger.exception("TTS worker error: %s", exc)
                self._is_speaking = False
                time.sleep(0.1)

        if engine is not None:
            self._cleanup_engine(engine)

    def _init_engine(self) -> Optional[Any]:
        """Initialize pyttsx3 engine."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            logger.debug("pyttsx3 engine initialized")
            return engine
        except Exception as exc:
            logger.error("Failed to init pyttsx3: %s", exc)
            return None

    def _cleanup_engine(self, engine: Any) -> None:
        """Cleanup engine."""
        if engine is None:
            return None
        try:
            engine.stop()
        except Exception:
            pass
        return None


# =============================================================================
# Module-level Singletons
# =============================================================================

_async_ear_instance: Optional[AsyncEar] = None
_async_tts_instance: Optional[AsyncTTS] = None
_instance_lock = threading.Lock()


def get_async_ear(**kwargs) -> AsyncEar:
    """Get or create the AsyncEar singleton."""
    global _async_ear_instance
    with _instance_lock:
        if _async_ear_instance is None:
            _async_ear_instance = AsyncEar(**kwargs)
        return _async_ear_instance


def get_async_tts(**kwargs) -> AsyncTTS:
    """Get or create the AsyncTTS singleton."""
    global _async_tts_instance
    with _instance_lock:
        if _async_tts_instance is None:
            _async_tts_instance = AsyncTTS(**kwargs)
        return _async_tts_instance


__all__ = [
    "RecognitionResult",
    "AsyncEar",
    "AsyncTTS",
    "get_async_ear",
    "get_async_tts",
]
