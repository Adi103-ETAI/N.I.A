"""N.O.L.A. Manager - Neural Operator for Language & Audio.

This module provides the NOLAManager class that orchestrates all voice I/O
operations for NIA, cleanly separating audio handling from the reasoning brain.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        NOLAManager                               │
    │  ┌─────────────┐    ┌─────────────┐    ┌───────────────────┐   │
    │  │  AsyncEar   │───►│  Security   │───►│  Output Queue     │   │
    │  │  (Listen)   │    │  Sanitizer  │    │  (to Brain)       │   │
    │  └─────────────┘    └─────────────┘    └───────────────────┘   │
    │                                                                  │
    │  ┌─────────────┐    ┌─────────────────────────────────────────┐ │
    │  │  AsyncTTS   │◄───│  Input Queue (from Brain)               │ │
    │  │  (Speak)    │    └─────────────────────────────────────────┘ │
    │  └─────────────┘                                                │
    └─────────────────────────────────────────────────────────────────┘

Features:
    - Wake word detection (optional, configurable)
    - Input sanitization with dangerous command blocking
    - Non-blocking I/O with queued communication
    - Coordinated pause/resume for echo cancellation
    - Comprehensive logging and error handling
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Import from NOLA submodules
from .security import InputSanitizer, SanitizedInput, SecurityLevel
from .wakeword import WakeWordDetector
from .io import AsyncEar, AsyncTTS, RecognitionResult

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass 
class NOLAConfig:
    """Configuration for NOLAManager.
    
    Attributes:
        wake_word_enabled: Whether to require wake word activation.
        wake_words: List of trigger phrases.
        wake_word_timeout: Seconds to stay active after wake word.
        security_enabled: Whether to enable input sanitization.
        custom_blocked_patterns: Additional regex patterns to block.
        custom_warning_patterns: Additional patterns for warnings.
        pause_ear_while_speaking: Pause listening during TTS.
        auto_resume_delay: Seconds to wait after TTS before resuming.
        max_input_queue_size: Max pending inputs for brain.
        max_output_queue_size: Max pending TTS messages.
    """
    # Wake word settings
    wake_word_enabled: bool = True
    wake_words: List[str] = field(default_factory=lambda: ["jarvis", "nia", "hey nia"])
    wake_word_timeout: float = 30.0
    
    # Security settings
    security_enabled: bool = True
    custom_blocked_patterns: List[str] = field(default_factory=list)
    custom_warning_patterns: List[str] = field(default_factory=list)
    
    # Audio settings
    pause_ear_while_speaking: bool = True
    auto_resume_delay: float = 0.3
    
    # Queue settings
    max_input_queue_size: int = 20
    max_output_queue_size: int = 50


class NOLAManager:
    """Neural Operator for Language & Audio - Voice I/O Orchestrator.
    
    Coordinates AsyncEar (listening), AsyncTTS (speaking), wake word
    detection, and input sanitization into a unified interface.
    
    Example:
        nola = NOLAManager()
        nola.start()
        
        while True:
            result = nola.get_input(timeout=0.5)
            if result:
                response = brain.process(result.text)
                nola.speak(response)
        
        nola.stop()
    """
    
    def __init__(
        self,
        config: Optional[NOLAConfig] = None,
        ear: Optional[AsyncEar] = None,
        tts: Optional[AsyncTTS] = None,
        on_security_block: Optional[Callable[[SanitizedInput], None]] = None,
    ) -> None:
        """Initialize the NOLA manager.
        
        Args:
            config: Configuration options.
            ear: Optional pre-configured AsyncEar instance.
            tts: Optional pre-configured AsyncTTS instance.
            on_security_block: Callback when input is blocked for security.
        """
        self.config = config or NOLAConfig()
        self._on_security_block = on_security_block
        
        # Initialize components (lazy - created on start if not provided)
        self._ear = ear
        self._tts = tts
        self._sanitizer = InputSanitizer(
            blocked_patterns=self.config.custom_blocked_patterns,
            warning_patterns=self.config.custom_warning_patterns,
        ) if self.config.security_enabled else None
        
        self._wake_detector = WakeWordDetector(
            wake_words=self.config.wake_words,
            timeout=self.config.wake_word_timeout,
        ) if self.config.wake_word_enabled else None
        
        # Output queue (sanitized input ready for brain)
        self._output_queue: queue.Queue[SanitizedInput] = queue.Queue(
            maxsize=self.config.max_input_queue_size
        )
        
        # State management
        self._is_running = False
        self._is_speaking = False
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "total_inputs": 0,
            "blocked_inputs": 0,
            "wake_word_activations": 0,
            "speak_calls": 0,
        }
    
    def start(self) -> bool:
        """Start the NOLA system (ear and processing loop).
        
        Returns:
            True if started successfully.
        """
        with self._lock:
            if self._is_running:
                logger.debug("NOLAManager already running")
                return True
            
            try:
                # Initialize ear if not provided
                if self._ear is None:
                    self._ear = AsyncEar()
                
                # Initialize TTS if not provided
                if self._tts is None:
                    self._tts = AsyncTTS()
                
                # Start components
                if hasattr(self._ear, 'start'):
                    self._ear.start()
                if hasattr(self._tts, 'start'):
                    self._tts.start()
                
                # Start processing loop
                self._stop_event.clear()
                self._processing_thread = threading.Thread(
                    target=self._processing_loop,
                    name="NOLA-Processor",
                    daemon=True,
                )
                self._processing_thread.start()
                
                self._is_running = True
                logger.info("NOLAManager started")
                return True
                
            except Exception as exc:
                logger.exception("Failed to start NOLAManager: %s", exc)
                return False
    
    def stop(self, timeout: float = 5.0) -> None:
        """Stop the NOLA system gracefully.
        
        Args:
            timeout: Maximum seconds to wait for shutdown.
        """
        with self._lock:
            if not self._is_running:
                return
            
            logger.info("Stopping NOLAManager...")
            self._stop_event.set()
        
        # Wait for processing thread
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=timeout)
        
        # Stop components
        if self._ear and hasattr(self._ear, 'stop'):
            try:
                self._ear.stop()
            except Exception as exc:
                logger.debug("Error stopping ear: %s", exc)
        
        if self._tts and hasattr(self._tts, 'stop'):
            try:
                self._tts.stop()
            except Exception as exc:
                logger.debug("Error stopping TTS: %s", exc)
        
        with self._lock:
            self._is_running = False
        
        logger.info("NOLAManager stopped (stats: %s)", self._stats)
    
    def speak(self, text: str, block_listening: bool = True) -> Dict[str, Any]:
        """Queue text for speech output.
        
        Args:
            text: Text to speak.
            block_listening: Whether to pause listening while speaking.
            
        Returns:
            Dict with operation status.
        """
        if not text:
            return {"ok": False, "error": "No text provided"}
        
        if not self._tts:
            logger.warning("TTS not available")
            return {"ok": False, "error": "TTS not initialized"}
        
        self._stats["speak_calls"] += 1
        
        # Pause ear while speaking (echo cancellation)
        if block_listening and self.config.pause_ear_while_speaking:
            if self._ear and hasattr(self._ear, 'pause'):
                self._ear.pause()
                self._is_speaking = True
        
        try:
            result = self._tts.speak(text)
            
            # Schedule ear resume after delay
            if block_listening and self.config.pause_ear_while_speaking:
                threading.Timer(
                    self.config.auto_resume_delay,
                    self._resume_ear_after_speak,
                ).start()
            
            return result
            
        except Exception as exc:
            logger.exception("TTS speak failed: %s", exc)
            self._resume_ear_after_speak()
            return {"ok": False, "error": str(exc)}
    
    def _resume_ear_after_speak(self) -> None:
        """Resume ear listening after TTS completes."""
        if self._ear and hasattr(self._ear, 'resume'):
            self._ear.resume()
        self._is_speaking = False
    
    def get_input(self, timeout: Optional[float] = None) -> Optional[SanitizedInput]:
        """Get the next sanitized voice input.
        
        Args:
            timeout: Seconds to wait. None/0 for non-blocking.
            
        Returns:
            SanitizedInput if available, None otherwise.
        """
        try:
            if timeout is None or timeout <= 0:
                return self._output_queue.get_nowait()
            return self._output_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all_pending_inputs(self) -> List[SanitizedInput]:
        """Get all pending sanitized inputs."""
        results = []
        while True:
            try:
                results.append(self._output_queue.get_nowait())
            except queue.Empty:
                break
        return results
    
    def clear_input_queue(self) -> int:
        """Clear pending inputs."""
        cleared = 0
        while True:
            try:
                self._output_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        return cleared
    
    def _processing_loop(self) -> None:
        """Main loop that processes ear input and applies security."""
        logger.debug("NOLA processing loop started")
        
        while not self._stop_event.is_set():
            try:
                # Skip processing while speaking
                if self._is_speaking:
                    time.sleep(0.1)
                    continue
                
                # Get raw input from ear
                if not self._ear or not hasattr(self._ear, 'get_text'):
                    time.sleep(0.1)
                    continue
                
                result = self._ear.get_text(timeout=0.2)
                if not result:
                    continue
                
                raw_text = result.text if hasattr(result, 'text') else str(result)
                if not raw_text:
                    continue
                
                self._stats["total_inputs"] += 1
                logger.debug("Processing: %s", raw_text[:50])
                
                # Wake word check
                wake_word_detected = True
                processed_text = raw_text
                
                if self._wake_detector:
                    wake_word_detected, processed_text = self._wake_detector.check(raw_text)
                    
                    if not wake_word_detected:
                        logger.debug("Ignoring (no wake word): %s", raw_text[:30])
                        continue
                    
                    if wake_word_detected and processed_text != raw_text:
                        self._stats["wake_word_activations"] += 1
                        logger.info("Wake word: %s", processed_text[:50])
                
                # Security sanitization
                if self._sanitizer:
                    sanitized = self._sanitizer.sanitize(processed_text)
                else:
                    sanitized = SanitizedInput(
                        text=processed_text,
                        original_text=raw_text,
                        security_level=SecurityLevel.SAFE,
                    )
                    
                sanitized.wake_word_detected = wake_word_detected
                sanitized.timestamp = time.time()
                
                # Handle blocked input
                if sanitized.is_blocked:
                    self._stats["blocked_inputs"] += 1
                    
                    if self._on_security_block:
                        try:
                            self._on_security_block(sanitized)
                        except Exception as exc:
                            logger.debug("Security callback error: %s", exc)
                    
                    self.speak(
                        "I'm sorry, but I cannot execute that command for security reasons.",
                        block_listening=True,
                    )
                    continue
                
                # Queue for brain processing
                try:
                    self._output_queue.put_nowait(sanitized)
                    logger.debug("Queued: %s", sanitized.text[:50])
                except queue.Full:
                    logger.warning("Input queue full")
                
            except Exception as exc:
                logger.exception("Processing loop error: %s", exc)
                time.sleep(0.1)
        
        logger.debug("NOLA processing loop exited")
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def is_running(self) -> bool:
        """Check if NOLA is running."""
        return self._is_running
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking
    
    @property
    def is_wake_active(self) -> bool:
        """Check if wake word is currently active."""
        if self._wake_detector:
            return self._wake_detector.is_active()
        return True
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get NOLA statistics."""
        return dict(self._stats)
    
    @property
    def ear(self) -> Optional[AsyncEar]:
        """Get the AsyncEar instance."""
        return self._ear
    
    @property
    def tts(self) -> Optional[AsyncTTS]:
        """Get the AsyncTTS instance."""
        return self._tts
    
    # =========================================================================
    # Configuration Methods
    # =========================================================================
    
    def get_security_audit(self) -> List[Dict[str, Any]]:
        """Get the security audit log of blocked commands."""
        if self._sanitizer:
            return self._sanitizer.get_blocked_attempts()
        return []
    
    def set_wake_words(self, words: List[str]) -> None:
        """Update wake words dynamically."""
        if self._wake_detector:
            self._wake_detector.set_wake_words(words)
    
    def toggle_wake_word(self, enabled: bool) -> None:
        """Enable or disable wake word detection."""
        self.config.wake_word_enabled = enabled
        if not enabled:
            self._wake_detector = None
        elif self._wake_detector is None:
            self._wake_detector = WakeWordDetector(
                wake_words=self.config.wake_words,
                timeout=self.config.wake_word_timeout,
            )
    
    def toggle_security(self, enabled: bool) -> None:
        """Enable or disable security sanitization."""
        self.config.security_enabled = enabled
        if not enabled:
            self._sanitizer = None
        elif self._sanitizer is None:
            self._sanitizer = InputSanitizer(
                blocked_patterns=self.config.custom_blocked_patterns,
                warning_patterns=self.config.custom_warning_patterns,
            )


# =============================================================================
# Module-level Singleton
# =============================================================================

_nola_instance: Optional[NOLAManager] = None
_instance_lock = threading.Lock()


def get_nola_manager(**kwargs) -> NOLAManager:
    """Get or create the module-level NOLAManager singleton."""
    global _nola_instance
    with _instance_lock:
        if _nola_instance is None:
            config_kwargs = {k: v for k, v in kwargs.items() if hasattr(NOLAConfig, k)}
            config = NOLAConfig(**config_kwargs) if config_kwargs else NOLAConfig()
            _nola_instance = NOLAManager(config=config)
        return _nola_instance


# =============================================================================
# Demo Function
# =============================================================================

def demo():
    """Run a quick demo of NOLA functionality."""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    print("=" * 60)
    print("N.O.L.A. Demo - Neural Operator for Language & Audio")
    print("=" * 60)
    print("Wake words: 'jarvis', 'nia', 'hey nia'")
    print("Say a wake word followed by your command.")
    print("Press Ctrl+C to exit.\n")
    
    nola = NOLAManager(
        config=NOLAConfig(
            wake_word_enabled=True,
            wake_words=["jarvis", "nia", "hey nia"],
        ),
    )
    
    try:
        if not nola.start():
            print("Failed to start NOLA. Check microphone.")
            return
        
        print("[NOLA] Listening...\n")
        
        while True:
            result = nola.get_input(timeout=0.5)
            if result:
                if result.is_blocked:
                    print(f"[BLOCKED] {result.blocked_reason}")
                else:
                    print(f"[INPUT] {result.text}")
                    nola.speak(f"You said: {result.text}")
    
    except KeyboardInterrupt:
        print("\n[NOLA] Shutting down...")
    finally:
        nola.stop()
        print("[NOLA] Goodbye!")


if __name__ == "__main__":
    demo()
