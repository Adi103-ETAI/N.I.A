"""N.O.L.A. I/O Module - Hardware Drivers for Audio.

TTS: Microsoft Edge Neural Voices (Primary) with Piper Fallback
STT: Vosk Offline Speech Recognition

Singleton Pattern:
    - Use get_async_ear() for STT instance
    - Use get_async_tts() for TTS instance
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

# Suppress pygame welcome message BEFORE importing pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Constants
# =============================================================================

# TTS cache directory
TTS_CACHE = Path("data/tts_cache")
TTS_CACHE.mkdir(parents=True, exist_ok=True)

# Edge TTS voice (Cortana-like)
EDGE_VOICE = "en-US-AriaNeural"

# Piper paths (fallback)
NOLA_DIR = Path(__file__).parent
PIPER_BIN = NOLA_DIR / "piper_bin" / ("piper.exe" if sys.platform == "win32" else "piper")
PIPER_MODEL = NOLA_DIR / "piper_bin" / "en_GB-alan-low.onnx"

# Alternative Piper model location
PIPER_MODEL_ALT = NOLA_DIR / "models" / "en_GB-alan-low.onnx"

# Vosk model
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
# HybridTTS - Edge Neural TTS with Piper Fallback
# =============================================================================

class HybridTTS:
    """Hybrid TTS: Microsoft Edge Neural Voices (Primary) + Piper (Fallback).
    
    Uses Edge TTS for high-quality cloud voices with automatic fallback
    to local Piper if Edge fails or is unavailable.
    
    Usage:
        tts = HybridTTS()
        tts.speak("Hello, I am N.I.A.")
    """
    
    def __init__(self, voice: str = EDGE_VOICE) -> None:
        """Initialize hybrid TTS.
        
        Args:
            voice: Edge TTS voice name.
        """
        self.voice = voice
        self._is_speaking = False
        self._lock = threading.Lock()
        self._speech_count = 0
        
        # Check dependency availability
        self._has_edge_tts = False
        self._has_pygame = False
        self._has_piper = False
        
        # Check edge-tts
        try:
            import edge_tts
            self._has_edge_tts = True
        except ImportError:
            logger.warning("edge-tts not installed. Run: pip install edge-tts")
        
        # Check pygame for audio playback
        try:
            import pygame
            pygame.mixer.init(frequency=24000)
            self._has_pygame = True
        except ImportError:
            logger.warning("pygame not installed. Run: pip install pygame")
        except Exception as e:
            logger.warning("pygame.mixer init failed: %s", e)
        
        # Check Piper availability
        self._piper_model = None
        if PIPER_BIN.exists():
            if PIPER_MODEL.exists():
                self._piper_model = PIPER_MODEL
                self._has_piper = True
            elif PIPER_MODEL_ALT.exists():
                self._piper_model = PIPER_MODEL_ALT
                self._has_piper = True
        
        # Log status
        if self._has_edge_tts and self._has_pygame:
            logger.info("HybridTTS: Edge TTS ready (voice: %s)", self.voice)
            print(f"🔊 Edge TTS Ready (Voice: {self.voice})")
        elif self._has_piper:
            logger.info("HybridTTS: Using Piper fallback")
            print("🔊 Piper TTS Ready (Fallback Mode)")
        else:
            logger.warning("HybridTTS: No TTS backend available")
            print("⚠️ TTS not available - will print to console")
    
    def speak(self, text: str) -> bool:
        """Speak text. Blocks until speech is complete.
        
        Args:
            text: Text to speak.
            
        Returns:
            True if audio played successfully.
        """
        if not text or not text.strip():
            return False
        
        text = text.strip()
        
        with self._lock:
            self._is_speaking = True
        
        try:
            # Try Edge TTS first (primary)
            if self._has_edge_tts and self._has_pygame:
                try:
                    return self._speak_edge(text)
                except Exception as e:
                    logger.warning("Edge TTS failed, trying fallback: %s", e)
            
            # Fallback to Piper
            if self._has_piper:
                try:
                    return self._speak_piper(text)
                except Exception as e:
                    logger.error("Piper TTS also failed: %s", e)
            
            # No TTS available - print to console
            print(f"🔊 [Console] {text}")
            return False
            
        finally:
            with self._lock:
                self._is_speaking = False
    
    def _speak_edge(self, text: str) -> bool:
        """Speak using Edge TTS with pygame playback."""
        import edge_tts
        import pygame
        
        # Generate unique filename
        self._speech_count += 1
        mp3_file = TTS_CACHE / f"edge_{self._speech_count}.mp3"
        
        try:
            # Generate audio asynchronously
            asyncio.run(self._async_generate_edge(text, str(mp3_file)))
            
            if not mp3_file.exists():
                logger.error("Edge TTS failed to generate audio file")
                return False
            
            # Play with pygame (blocking)
            pygame.mixer.music.load(str(mp3_file))
            pygame.mixer.music.play()
            
            # Block until playback completes
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Cleanup
            pygame.mixer.music.unload()
            try:
                mp3_file.unlink()
            except Exception:
                pass
            
            logger.debug("Edge TTS spoke: %s", text[:50])
            return True
            
        except Exception as e:
            logger.error("Edge TTS playback error: %s", e)
            # Cleanup on error
            try:
                if mp3_file.exists():
                    mp3_file.unlink()
            except Exception:
                pass
            raise
    
    async def _async_generate_edge(self, text: str, output_path: str) -> None:
        """Async Edge TTS audio generation."""
        import edge_tts
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)
    
    def _speak_piper(self, text: str) -> bool:
        """Speak using Piper binary with sounddevice playback."""
        self._speech_count += 1
        wav_file = TTS_CACHE / f"piper_{self._speech_count}.wav"
        
        try:
            # Run Piper subprocess
            cmd = [
                str(PIPER_BIN),
                "--model", str(self._piper_model),
                "--output_file", str(wav_file)
            ]
            
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            result = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                timeout=30,
                creationflags=creationflags
            )
            
            if result.returncode != 0:
                logger.error("Piper failed: %s", result.stderr.decode()[:200])
                return False
            
            if not wav_file.exists():
                return False
            
            # Play with sounddevice
            try:
                import sounddevice as sd
                import numpy as np
                import wave
                
                with wave.open(str(wav_file), 'rb') as wf:
                    samplerate = wf.getframerate()
                    data = wf.readframes(-1)
                    audio = np.frombuffer(data, dtype=np.int16)
                
                sd.play(audio, samplerate)
                sd.wait()
                
            except ImportError:
                logger.error("sounddevice not available for Piper playback")
                return False
            
            # Cleanup
            try:
                wav_file.unlink()
            except Exception:
                pass
            
            logger.debug("Piper TTS spoke: %s", text[:50])
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Piper timed out")
            return False
        except Exception as e:
            logger.error("Piper error: %s", e)
            raise
    
    def stop(self) -> None:
        """Stop current audio playback."""
        if self._has_pygame:
            try:
                import pygame
                pygame.mixer.music.stop()
            except Exception:
                pass
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        with self._lock:
            return self._is_speaking


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
                logger.info("👂 VoskSTT: Model loaded and ready")
            except Exception as e:
                logger.error("Failed to load Vosk model: %s", e)
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
                    
                    except Exception as e:
                        logger.debug("Stream error: %s", e)
                        time.sleep(0.1)
                        
        except Exception as e:
            logger.error("VoskSTT stream error: %s", e)
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
# Singleton Instances
# =============================================================================

_tts_instance: Optional[HybridTTS] = None
_stt_instance: Optional[VoskSTT] = None


def get_async_tts() -> HybridTTS:
    """Get or create the TTS singleton.
    
    Returns:
        HybridTTS instance.
    """
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = HybridTTS()
    return _tts_instance


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
# Legacy Compatibility Wrappers
# =============================================================================

class AsyncTTS:
    """Legacy wrapper for backward compatibility."""
    
    def __init__(self, **kwargs) -> None:
        self._tts = get_async_tts()
        self._is_running = True
    
    def speak(self, text: str) -> Dict[str, Any]:
        ok = self._tts.speak(text)
        return {"ok": ok}
    
    def stop_speaking(self) -> None:
        self._tts.stop()
    
    def is_speaking(self) -> bool:
        return self._tts.is_speaking
    
    def stop(self) -> None:
        self._is_running = False
    
    @property
    def is_running(self) -> bool:
        return self._is_running


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
    "HybridTTS",
    "VoskSTT",
    "AsyncEar",
    "AsyncTTS",
    "get_async_tts",
    "get_async_ear",
]
