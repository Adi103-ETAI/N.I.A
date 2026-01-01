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
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
# AsyncEar - Non-blocking Microphone Listener (Vosk Offline STT)
# =============================================================================

class AsyncEar:
    """Thread-safe, non-blocking ASR engine using Vosk (fully offline).

    Uses Vosk for continuous speech recognition without internet dependency.
    Supports "one-shot" commands where wake word + command are captured together.

    Usage:
        ear = AsyncEar(wake_words=["jarvis", "nia"])
        ear.start()
        
        result = ear.get_text(timeout=0.1)
        if result:
            print(f"Heard: {result.text}")
        
        ear.pause()   # While TTS speaking
        ear.resume()  # After TTS done
        ear.stop()    # Cleanup
    """

    # Vosk model download URL
    VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"

    def __init__(
        self,
        max_queue_size: int = 50,
        wake_words: Optional[List[str]] = None,
        device_index: Optional[int] = None,
    ) -> None:
        """Initialize the async ASR system with Vosk.

        Args:
            max_queue_size: Maximum pending recognitions before oldest dropped.
            wake_words: List of wake words to detect (e.g., ["jarvis", "nia"]).
            device_index: Specific microphone device index (None for default).
        """
        self._queue: queue.Queue[RecognitionResult] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()

        # Wake words for one-shot detection
        self._wake_words = [w.lower() for w in (wake_words or ["jarvis", "nia", "hey nia"])]
        self._device_index = device_index

        # State tracking
        self._is_running = False
        self._is_paused = False
        self._thread: Optional[threading.Thread] = None
        
        # Active Listening Window (Conversation Mode)
        self._last_wake_time: float = 0  # Last time wake word was triggered
        self._active_window: float = 15.0  # Seconds to stay in active mode

        # Vosk components (lazy loaded)
        self._model = None
        self._recognizer = None
        self._has_vosk = False
        self._has_sounddevice = False

        # Statistics
        self._total_recognitions = 0
        self._failed_recognitions = 0

        # Setup paths
        self._nola_dir = os.path.dirname(os.path.abspath(__file__))
        self._model_path = os.path.join(self._nola_dir, "vosk_model")

        # Check dependencies
        try:
            import vosk  # type: ignore
            self._vosk_module = vosk
            self._has_vosk = True
        except ImportError:
            logger.warning("vosk not installed; run: pip install vosk")

        try:
            import sounddevice  # type: ignore
            self._has_sounddevice = True
        except ImportError:
            logger.warning("sounddevice not installed; run: pip install sounddevice")

    def start(self) -> bool:
        """Start the background listening thread.
        
        Returns:
            True if started successfully.
        """
        with self._lock:
            if self._is_running:
                logger.debug("AsyncEar already running")
                return True

            if not self._has_vosk or not self._has_sounddevice:
                logger.error("Cannot start: vosk or sounddevice not available")
                return False

            try:
                # Ensure model exists
                self._ensure_model_exists()
                
                # Initialize Vosk
                self._initialize_vosk()
                
                # Start listening thread
                self._stop_event.clear()
                self._pause_event.clear()
                self._thread = threading.Thread(
                    target=self._listen_loop,
                    name="AsyncEar-VoskListener",
                    daemon=True,
                )
                self._thread.start()
                self._is_running = True
                logger.info("AsyncEar started (Vosk offline)")
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

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        with self._lock:
            self._is_running = False
            self._thread = None

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

    def _ensure_model_exists(self) -> None:
        """Auto-downloads Vosk model if not present."""
        if os.path.exists(self._model_path):
            return

        try:
            import requests
            import zipfile
        except ImportError:
            logger.error("requests/zipfile not available; cannot download Vosk model")
            raise

        print("⬇️ Downloading Vosk Model (~40MB)...")
        logger.info("Downloading Vosk model from %s", self.VOSK_MODEL_URL)

        zip_path = os.path.join(self._nola_dir, "vosk_model.zip")

        try:
            # Download
            response = requests.get(self.VOSK_MODEL_URL, stream=True)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract
            print("📦 Extracting Vosk Model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self._nola_dir)

            # Rename extracted folder to 'vosk_model'
            extracted_path = os.path.join(self._nola_dir, self.VOSK_MODEL_NAME)
            if os.path.exists(extracted_path):
                os.rename(extracted_path, self._model_path)

            # Cleanup
            os.remove(zip_path)
            print("✅ Vosk Model Ready.")
            logger.info("Vosk model installed successfully")

        except Exception as exc:
            logger.error("Failed to download Vosk model: %s", exc)
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise

    def _initialize_vosk(self) -> None:
        """Initialize Vosk model and recognizer."""
        vosk = self._vosk_module

        print("👂 Loading Vosk Model...")
        vosk.SetLogLevel(-1)  # Silence Vosk logs

        self._model = vosk.Model(self._model_path)
        self._recognizer = vosk.KaldiRecognizer(self._model, 16000)
        
        print("✅ Vosk Ear Ready.")
        logger.info("Vosk recognizer initialized (16kHz)")

    def _listen_loop(self) -> None:
        """Continuous audio streaming and recognition loop."""
        import json
        import sounddevice as sd

        try:
            with sd.RawInputStream(
                samplerate=16000,
                blocksize=8000,
                device=self._device_index,
                dtype='int16',
                channels=1,
            ) as stream:
                print("👂 Listening (Vosk Offline)...")
                logger.info("Vosk listening started")

                while not self._stop_event.is_set():
                    # Skip if paused (echo cancellation)
                    if self._pause_event.is_set():
                        time.sleep(0.1)
                        continue

                    try:
                        data, overflowed = stream.read(4000)
                        if len(data) == 0:
                            continue

                        # Feed to Vosk
                        if self._recognizer.AcceptWaveform(bytes(data)):
                            # Final sentence completed
                            result = json.loads(self._recognizer.Result())
                            text = result.get('text', '').strip()
                            if text:
                                self._process_text(text)
                        else:
                            # Partial result (streaming) - could use for fast wake detection
                            # partial = json.loads(self._recognizer.PartialResult())
                            # partial_text = partial.get('partial', '')
                            pass

                    except Exception as exc:
                        self._failed_recognitions += 1
                        logger.debug("Vosk read error: %s", exc)
                        time.sleep(0.1)

        except Exception as exc:
            logger.exception("Vosk listen loop error: %s", exc)

    def _process_text(self, text: str) -> None:
        """Process recognized text with Active Listening Window.
        
        After wake word activation, listens for follow-up commands
        for `_active_window` seconds without requiring wake word.
        """
        text = text.lower().strip()
        if not text:
            return

        current_time = time.time()
        is_active = (current_time - self._last_wake_time) < self._active_window
        
        # Check for wake words
        command = text
        wake_triggered = False

        for ww in self._wake_words:
            if text.startswith(ww):
                wake_triggered = True
                # Strip wake word: "jarvis time" -> "time"
                command = text[len(ww):].strip()
                break

        # === ACTIVE LISTENING WINDOW LOGIC ===
        if wake_triggered:
            # Wake word detected - activate/reset window
            if not is_active:
                print("🔴→🟢 Activated (listening for 15s)...")
            self._last_wake_time = current_time
            
        elif is_active:
            # No wake word, but we're in active window - accept command
            print(f"🟢 Active: '{text}'")
            logger.debug("Active window command: %s", text)
            command = text
            # Reset timer to extend conversation
            self._last_wake_time = current_time
            
        else:
            # Not active and no wake word - ignore
            logger.debug("Ignored (standby): %s", text)
            return

        # Create result (command may be empty if just wake word)
        result = RecognitionResult(
            text=command if command else " ",  # Space means "wake up only"
            confidence=1.0,
            timestamp=current_time,
            is_final=True,
        )

        # Log what was heard
        if wake_triggered:
            if command:
                print(f"🎤 One-Shot: '{text}' → '{command}'")
                logger.info("One-shot command: %s -> %s", text, command)
            else:
                print(f"🎤 Wake Word: '{text}'")
                logger.info("Wake word only: %s", text)

        try:
            self._queue.put_nowait(result)
            self._total_recognitions += 1
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(result)
            except queue.Empty:
                pass


# =============================================================================
# AsyncTTS - Non-blocking Text-to-Speech with Piper Binary (Subprocess Wrapper)
# =============================================================================

class AsyncTTS:
    """Thread-safe, non-blocking TTS engine using Piper Binary (piper.exe).

    Uses Piper executable via subprocess for stable, isolated TTS synthesis.
    No Python bindings - avoids DLL/crash issues on Windows.
    The engine runs in a dedicated daemon thread, consuming text from a queue.

    Usage:
        tts = AsyncTTS()
        tts.speak("Hello!")  # Returns immediately
        tts.stop_speaking()  # Interrupt current speech
    """

    # Piper release URL (Windows AMD64)
    PIPER_RELEASE_URL = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"

    def __init__(self, max_queue_size: int = 100) -> None:
        """Initialize the async TTS system.

        Args:
            max_queue_size: Maximum pending messages before dropping oldest.
        """
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._is_running = True
        self._is_speaking = False
        
        # Setup Paths
        self._nola_dir = os.path.dirname(os.path.abspath(__file__))
        self._bin_dir = os.path.join(self._nola_dir, "piper_bin")
        self._exe_path = os.path.join(self._bin_dir, "piper", "piper.exe")
        
        # Model paths (reuse existing models folder)
        self._model_dir = os.path.join(self._nola_dir, "models")
        self._model_name = "en_GB-alan-low"
        self._onnx_path = os.path.join(self._model_dir, f"{self._model_name}.onnx")
        self._conf_path = os.path.join(self._model_dir, f"{self._model_name}.onnx.json")
        
        # Cleanup old voice files from previous attempts
        self._cleanup_old_model("en_US-lessac-medium")
        self._cleanup_old_model("en_IN-kusal-medium")
        self._cleanup_old_model("en_IN-kusal-low")
        
        # Temp audio file path
        self._temp_wav = os.path.join(self._nola_dir, "temp_speech.wav")
        
        # Check for sounddevice
        self._has_sounddevice = False
        try:
            import sounddevice  # type: ignore
            self._has_sounddevice = True
        except ImportError:
            logger.warning("sounddevice not installed; run: pip install sounddevice")
        
        # Auto-setup: download binary and model if missing
        try:
            self._ensure_binary_exists()
            self._ensure_model_exists()
            self._ready = os.path.exists(self._exe_path) and os.path.exists(self._onnx_path)
        except Exception as exc:
            logger.error("Failed to setup Piper: %s", exc)
            self._ready = False
        
        if self._ready:
            print("🔊 Piper TTS (Binary Mode) Ready.")
            logger.info("AsyncTTS started (Piper Binary Mode)")
        else:
            print("⚠️ Piper TTS not available - will print to console instead.")
            logger.warning("Piper TTS not ready - fallback to console output")
        
        # Start worker thread
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="AsyncTTS-PiperBinaryWorker",
            daemon=True,
        )
        self._thread.start()

    def _ensure_binary_exists(self) -> None:
        """Auto-downloads Piper executable if not present."""
        if os.path.exists(self._exe_path):
            return
        
        try:
            import requests
            import zipfile
        except ImportError:
            logger.error("requests/zipfile not available; cannot download Piper binary")
            return
        
        print("⬇️ Downloading Piper Binary (Standalone ~20MB)...")
        logger.info("Downloading Piper binary from GitHub...")
        
        zip_path = os.path.join(self._nola_dir, "piper_bin.zip")
        
        try:
            # Download zip
            response = requests.get(self.PIPER_RELEASE_URL, stream=True)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract
            print("📦 Extracting Piper...")
            if not os.path.exists(self._bin_dir):
                os.makedirs(self._bin_dir)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self._bin_dir)
            
            # Cleanup zip
            os.remove(zip_path)
            print("✅ Piper Binary Installed.")
            logger.info("Piper binary installed successfully")
            
        except Exception as exc:
            logger.error("Failed to download Piper binary: %s", exc)
            # Cleanup partial download
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise

    def _cleanup_old_model(self, old_model_name: str) -> None:
        """Delete old model files to save space when switching voices."""
        old_onnx = os.path.join(self._model_dir, f"{old_model_name}.onnx")
        old_conf = os.path.join(self._model_dir, f"{old_model_name}.onnx.json")
        
        cleaned = False
        if os.path.exists(old_onnx):
            try:
                os.remove(old_onnx)
                cleaned = True
                logger.info("Deleted old model: %s", old_onnx)
            except Exception as exc:
                logger.warning("Failed to delete old model: %s", exc)
        
        if os.path.exists(old_conf):
            try:
                os.remove(old_conf)
                cleaned = True
                logger.info("Deleted old config: %s", old_conf)
            except Exception as exc:
                logger.warning("Failed to delete old config: %s", exc)
        
        if cleaned:
            print("🧹 Cleaned up old US voice.")

    def _ensure_model_exists(self) -> None:
        """Auto-downloads the Piper voice model if not present."""
        try:
            import requests
        except ImportError:
            logger.error("requests not installed; cannot download model")
            return
        
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/low"
        
        # Cleanup corrupt/empty files from failed downloads
        if os.path.exists(self._onnx_path) and os.path.getsize(self._onnx_path) < 1024:
            print("🧹 Cleaning up corrupt download...")
            logger.warning("Removing corrupt model file (< 1KB)")
            os.remove(self._onnx_path)
            if os.path.exists(self._conf_path):
                os.remove(self._conf_path)
        
        # Download ONNX model
        if not os.path.exists(self._onnx_path):
            print(f"⬇️ Downloading Piper Voice Model (~60MB)...")
            logger.info("Downloading Piper model: %s", self._model_name)
            try:
                response = requests.get(f"{base_url}/{self._model_name}.onnx", stream=True)
                response.raise_for_status()
                with open(self._onnx_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("✅ Model Downloaded.")
                logger.info("Piper model downloaded successfully")
            except Exception as exc:
                logger.error("Failed to download Piper model: %s", exc)
                raise
        
        # Download config JSON
        if not os.path.exists(self._conf_path):
            print("⬇️ Downloading Config...")
            logger.info("Downloading Piper config...")
            try:
                response = requests.get(f"{base_url}/{self._model_name}.onnx.json")
                response.raise_for_status()
                with open(self._conf_path, "wb") as f:
                    f.write(response.content)
                print("✅ Config Downloaded.")
                logger.info("Piper config downloaded successfully")
            except Exception as exc:
                logger.error("Failed to download Piper config: %s", exc)
                raise
        
        # INTEGRITY CHECK: Model should be > 1MB (actual size ~60MB)
        if os.path.exists(self._onnx_path) and os.path.getsize(self._onnx_path) < 1024 * 1024:
            print("⚠️ Warning: Model file looks too small (corrupt). Deleting...")
            logger.warning("Model file corrupted (too small), deleting for re-download")
            os.remove(self._onnx_path)
            if os.path.exists(self._conf_path):
                os.remove(self._conf_path)
            raise ValueError("Model download corrupted. Please restart NIA to try again.")

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
            return {"ok": False, "error": "TTS not running"}

        try:
            self._queue.put_nowait(text.strip())
            return {"ok": True, "queued": True}
        except queue.Full:
            # Queue full - drop oldest and add new
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(text.strip())
                return {"ok": True, "queued": True, "dropped_oldest": True}
            except queue.Empty:
                pass
            logger.warning("TTS queue full")
            return {"ok": False, "error": "TTS queue full"}

    def stop_speaking(self) -> None:
        """Stop current audio and clear queue."""
        # Stop sounddevice audio if playing
        if self._has_sounddevice:
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
        
        # Clear queue
        with self._queue.mutex:
            self._queue.queue.clear()
        
        self._is_speaking = False

    def is_speaking(self) -> bool:
        """Check if there are pending messages or currently speaking."""
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

    def stop(self) -> None:
        """Stop the TTS engine."""
        self._is_running = False
        self.stop_speaking()
        # Cleanup temp file
        if os.path.exists(self._temp_wav):
            try:
                os.remove(self._temp_wav)
            except Exception:
                pass
        logger.info("AsyncTTS stopped")

    def _worker_loop(self) -> None:
        """Main worker loop - generates and plays audio via Piper subprocess."""
        import subprocess
        import wave
        import numpy as np
        import sounddevice as sd
        
        while self._is_running:
            try:
                text = self._queue.get(timeout=1.0)
                self._is_speaking = True
                
                if self._ready and self._has_sounddevice:
                    try:
                        # 1. Generate Audio via Subprocess (Safe & Stable)
                        # Command: echo text | piper.exe --model model.onnx --output_file output.wav
                        cmd = [
                            self._exe_path,
                            "--model", self._onnx_path,
                            "--output_file", self._temp_wav
                        ]
                        
                        # Run Piper with text piped to stdin
                        process = subprocess.run(
                            cmd,
                            input=text.encode('utf-8'),
                            capture_output=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                        )
                        
                        if process.returncode != 0:
                            error_msg = process.stderr.decode() if process.stderr else "Unknown error"
                            logger.error("Piper failed: %s", error_msg)
                            print(f"⚠️ Piper Failed: {error_msg}")
                            self._queue.task_done()
                            self._is_speaking = False
                            continue
                        
                        # 2. Play Audio
                        if os.path.exists(self._temp_wav):
                            with wave.open(self._temp_wav, 'rb') as wf:
                                samplerate = wf.getframerate()
                                data = wf.readframes(-1)
                                # Convert to int16 numpy array
                                audio_np = np.frombuffer(data, dtype=np.int16)
                            
                            sd.play(audio_np, samplerate)
                            sd.wait()  # Block thread until finished
                            
                            # Cleanup temp file
                            try:
                                os.remove(self._temp_wav)
                            except Exception:
                                pass
                        
                        logger.debug("Spoke: %s", text[:50])
                        
                    except Exception as exc:
                        logger.error("⚠️ TTS Error: %s", exc)
                        print(f"⚠️ TTS Error: {exc}")
                else:
                    # No TTS available - print to console
                    print(f"🔊 {text}")
                
                self._is_speaking = False
                self._queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as exc:
                logger.exception("TTS worker error: %s", exc)
                self._is_speaking = False


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
