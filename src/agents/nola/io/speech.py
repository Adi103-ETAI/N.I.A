"""N.O.L.A. Speech Output - HybridTTS (Edge TTS + Piper fallback).

Provides text-to-speech using:
- Primary: Edge TTS (Microsoft online, high quality)
- Fallback: Piper TTS (offline, local, fast)
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import subprocess
import threading
import tempfile
from pathlib import Path
from typing import Optional

from src.core.logger import setup_logger

logger = setup_logger("NOLA_TTS")

# Optional dependencies
try:
    import edge_tts
    _HAS_EDGE_TTS = True
except ImportError:
    _HAS_EDGE_TTS = False
    logger.warning("edge-tts not installed. TTS may be limited.")

try:
    import pygame
    pygame.mixer.init()
    _HAS_PYGAME = True
except (ImportError, Exception):
    _HAS_PYGAME = False
    logger.warning("pygame not available for audio playback.")


# =============================================================================
# Async TTS Wrapper
# =============================================================================

class AsyncTTS:
    """Async wrapper around HybridTTS for thread-safe usage."""

    def __init__(self):
        self._tts: Optional[HybridTTS] = None
        self._lock = threading.Lock()

    def _get_tts(self) -> HybridTTS:
        if self._tts is None:
            self._tts = HybridTTS()
        return self._tts

    def speak(self, text: str) -> bool:
        """Speak text (blocking, thread-safe)."""
        with self._lock:
            return self._get_tts().speak(text)

    def stop(self) -> None:
        """Stop current playback."""
        if self._tts:
            self._tts.stop()

    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        if self._tts:
            return self._tts.is_speaking
        return False


# =============================================================================
# HybridTTS - Edge TTS (Primary) + Piper (Fallback)
# =============================================================================

class HybridTTS:
    """Text-to-Speech using Edge TTS with Piper TTS fallback.
    
    Edge TTS provides high-quality online voices.
    Piper TTS provides offline fallback capability.
    """

    def __init__(self, voice: str = "en-US-AriaNeural"):
        self.voice = voice
        self._is_speaking = False
        self._stop_requested = False
        
        # Piper binary path (optional offline fallback)
        self._piper_path = Path(__file__).parent.parent / "piper_bin"
        self._sounds_dir = Path(__file__).parent.parent.parent.parent.parent / "sounds"

    def speak(self, text: str) -> bool:
        """Speak text using Edge TTS, falling back to Piper.
        
        Args:
            text: Text to speak.
            
        Returns:
            True if speech was successful.
        """
        if not text or not text.strip():
            return False

        self._is_speaking = True
        self._stop_requested = False

        try:
            # Try Edge TTS first (online, high quality)
            if _HAS_EDGE_TTS:
                try:
                    return self._speak_edge(text)
                except Exception as e:
                    logger.warning(f"Edge TTS failed, trying Piper fallback: {e}")
            
            # Fallback to Piper (offline)
            try:
                return self._speak_piper(text)
            except Exception as e:
                logger.error(f"All TTS methods failed: {e}")
                return False
        finally:
            self._is_speaking = False

    def _speak_edge(self, text: str) -> bool:
        """Speak using Edge TTS."""
        async def _edge_speak():
            communicate = edge_tts.Communicate(text, self.voice)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                await communicate.save(tmp_path)
                
                if self._stop_requested:
                    return False
                
                self._play_audio(tmp_path)
                return True
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # Run async edge_tts in a new event loop
        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(_edge_speak())
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            raise

    def _speak_piper(self, text: str) -> bool:
        """Speak using Piper TTS (offline fallback)."""
        piper_exe = self._piper_path / "piper.exe"
        
        if not piper_exe.exists():
            logger.debug("Piper executable not found, skipping fallback")
            return False

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            process = subprocess.run(
                [str(piper_exe), "--output_file", tmp_path],
                input=text.encode(),
                capture_output=True,
                timeout=30,
            )

            if process.returncode == 0 and os.path.exists(tmp_path):
                self._play_audio(tmp_path)
                return True
            
            return False
        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except (OSError, UnboundLocalError):
                pass

    def _play_audio(self, filepath: str) -> None:
        """Play audio file using pygame."""
        if not _HAS_PYGAME:
            logger.warning("No audio player available")
            return

        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                if self._stop_requested:
                    pygame.mixer.music.stop()
                    break
                pygame.time.wait(50)
        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    def stop(self) -> None:
        """Stop current speech."""
        self._stop_requested = True
        if _HAS_PYGAME:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    @property
    def is_speaking(self) -> bool:
        """Check if currently playing audio."""
        return self._is_speaking


# =============================================================================
# Singleton Accessor
# =============================================================================

_async_tts: Optional[AsyncTTS] = None


def get_async_tts() -> AsyncTTS:
    """Get the global AsyncTTS singleton.
    
    Returns:
        The AsyncTTS instance.
    """
    global _async_tts
    if _async_tts is None:
        _async_tts = AsyncTTS()
    return _async_tts


__all__ = [
    "HybridTTS",
    "AsyncTTS",
    "get_async_tts",
]
