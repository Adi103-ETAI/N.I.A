"""NOLA Speech Module - Text-to-Speech Hardware Driver ("The Mouth").

TTS: Microsoft Edge Neural Voices (Primary) with Piper Fallback

Usage:
    from nola.io.speech import HybridTTS, get_async_tts
    
    tts = get_async_tts()  # Singleton
    tts.speak("Hello, I am N.I.A.")
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Suppress pygame welcome message BEFORE importing pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

# Centralized logging
from core.logger import setup_logger
logger = setup_logger("NOLA.SPEECH")

# Centralized configuration
from core.config import settings


# =============================================================================
# Configuration Constants
# =============================================================================

# TTS cache directory
TTS_CACHE = settings.DATA_DIR / "tts_cache"
TTS_CACHE.mkdir(parents=True, exist_ok=True)

# Edge TTS voice (from settings)
EDGE_VOICE = settings.TTS_VOICE

# Piper paths (fallback)
NOLA_DIR = Path(__file__).parent.parent  # nola/io_pkg -> nola/
PIPER_BIN = NOLA_DIR / "piper_bin" / ("piper.exe" if sys.platform == "win32" else "piper")
PIPER_MODEL = NOLA_DIR / "piper_bin" / "en_GB-alan-low.onnx"
PIPER_MODEL_ALT = NOLA_DIR / "models" / "en_GB-alan-low.onnx"

import json

def _load_voice_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config" / "nola" / "voice.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load voice.json: {e}")
        return {}

_VOICE_CONFIG = _load_voice_config()
_SPEECH_CONFIG = _VOICE_CONFIG.get("speech", {})

# Audio timing constants
PLAYBACK_POLL_INTERVAL = _SPEECH_CONFIG.get("playback_poll_interval", 0.1)
PIPER_TIMEOUT_SEC = _SPEECH_CONFIG.get("piper_timeout_sec", 30)


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
        except OSError as e:
            logger.warning(f"pygame.mixer init failed (audio device error): {e}")
        except RuntimeError as e:
            logger.warning(f"pygame.mixer init failed (runtime error): {e}")
        
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
            logger.debug("HybridTTS: Edge TTS ready (voice: %s)", self.voice)
            # print(f"🔊 Edge TTS Ready (Voice: {self.voice})")
        elif self._has_piper:
            logger.debug("HybridTTS: Using Piper fallback")
            # print("🔊 Piper TTS Ready (Fallback Mode)")
        else:
            logger.warning("HybridTTS: No TTS backend available")
            # print("⚠️ TTS not available - will print to console")
    
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
                except (subprocess.SubprocessError, OSError) as e:
                    logger.error(f"Piper TTS also failed: {e}", exc_info=True)
            
            # No TTS available - print to console
            # No TTS available - print to console
            # print(f"🔊 [Console] {text}")
            logger.warning(f"[Console Fallback] {text}")
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
                time.sleep(PLAYBACK_POLL_INTERVAL)
            
            # Cleanup
            pygame.mixer.music.unload()
            try:
                mp3_file.unlink()
            except OSError:
                pass
            
            logger.debug("Edge TTS spoke: %s", text[:50])
            return True
            
        except (OSError, RuntimeError) as e:
            logger.error(f"Edge TTS playback error: {e}", exc_info=True)
            # Cleanup on error
            try:
                if mp3_file.exists():
                    mp3_file.unlink()
            except OSError:
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
                timeout=PIPER_TIMEOUT_SEC,
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
            except OSError:
                pass
            
            logger.debug("Piper TTS spoke: %s", text[:50])
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Piper timed out")
            return False
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"Piper subprocess error: {e}", exc_info=True)
            raise
    
    def stop(self) -> None:
        """Stop current audio playback."""
        if self._has_pygame:
            try:
                import pygame
                pygame.mixer.music.stop()
            except (ImportError, OSError, RuntimeError):
                pass
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        with self._lock:
            return self._is_speaking


# =============================================================================
# Singleton Instance
# =============================================================================

_tts_instance: Optional[HybridTTS] = None


def get_async_tts() -> HybridTTS:
    """Get or create the TTS singleton.
    
    Returns:
        HybridTTS instance.
    """
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = HybridTTS()
    return _tts_instance


# =============================================================================
# Legacy Compatibility Wrapper
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


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "HybridTTS",
    "AsyncTTS",
    "get_async_tts",
    "TTS_CACHE",
    "EDGE_VOICE",
]
