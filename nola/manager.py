"""N.O.L.A. Manager - Voice I/O Orchestrator with Wake Word State Machine.

Neural Operator for Language & Audio - manages voice input/output
with strict wake word filtering and singleton pattern.

State Machine:
    ASLEEP → (wake word) → AWAKE
    AWAKE → (command processed) → ASLEEP
    AWAKE → (sleep command) → ASLEEP

Singleton Pattern:
    from nola.manager import get_nola_manager
    manager = get_nola_manager()
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

# Import I/O classes (explicit submodules)
from .io.speech import HybridTTS, get_async_tts
from .io.hearing import VoskSTT, get_async_ear

# Centralized logging
from core.logger import setup_logger
logger = setup_logger("NOLA")


# =============================================================================
# State Constants
# =============================================================================

STATE_ASLEEP = "ASLEEP"
STATE_AWAKE = "AWAKE"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class NOLAConfig:
    """Configuration for NOLAManager."""
    
    # Wake word settings
    wake_word_enabled: bool = True
    wake_words: List[str] = field(default_factory=lambda: ["nia", "jarvis", "hey nia"])
    wake_word_timeout: float = 30.0
    
    # Sleep commands
    sleep_commands: List[str] = field(default_factory=lambda: [
        "stop listening", "go to sleep", "sleep mode",
        "goodbye", "bye bye", "that's all",
    ])
    
    # Behavior settings
    auto_sleep_after_command: bool = True
    active_window_seconds: float = 20.0
    
    # Audio settings
    pause_ear_while_speaking: bool = True
    
    # Security
    security_enabled: bool = True


# =============================================================================
# NOLA Manager - Voice I/O Orchestrator
# =============================================================================

class NOLAManager:
    """Neural Operator for Language & Audio.
    
    Manages voice input with a strict wake word state machine.
    Uses singleton I/O drivers to prevent hardware conflicts.
    
    Example:
        manager = NOLAManager()
        manager.start()
        
        # Get voice input (blocks until wake word + command)
        text = manager.get_input(timeout=30)
        if text:
            response = brain.process(text)
            manager.speak(response)
    """
    
    def __init__(
        self,
        config: Optional[NOLAConfig] = None,
        on_wake: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize NOLA manager.
        
        Args:
            config: Configuration options.
            on_wake: Callback when wake word detected.
        """
        self.config = config or NOLAConfig()
        self._on_wake = on_wake
        
        # State machine
        self.state = STATE_ASLEEP
        self._last_wake_time: float = 0
        
        # Use singleton I/O drivers
        self._tts = get_async_tts()
        self._stt = get_async_ear()
        
        # Input queue for processed commands
        self._input_queue: queue.Queue[str] = queue.Queue(maxsize=20)
        
        # Thread control
        self._is_running = False
        self._is_paused = False
        self._user_paused = False  # Track if user explicitly paused (vs TTS auto-pause)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Normalize wake words and sleep commands
        self._wake_words = [w.lower().strip() for w in self.config.wake_words]
        self._sleep_commands = [c.lower().strip() for c in self.config.sleep_commands]
        
        logger.info("NOLAManager initialized (wake_words: %s)", self._wake_words)
    
    # =========================================================================
    # Lifecycle Methods
    # =========================================================================
    
    def start(self) -> bool:
        """Start the NOLA voice system.
        
        Returns:
            True if started successfully.
        """
        if self._is_running:
            return True
        
        self._is_running = True
        self._stop_event.clear()
        
        # Start processing thread
        self._thread = threading.Thread(
            target=self._process_loop,
            name="NOLA-VoiceLoop",
            daemon=True,
        )
        self._thread.start()
        
        print(f"🎙️ NOLA Voice System Started")
        print(f"💤 State: {self.state} | Say '{self._wake_words[0]}' to wake")
        logger.info("NOLAManager started")
        return True
    
    def stop(self, timeout: float = 3.0) -> None:
        """Stop the NOLA voice system.
        
        Args:
            timeout: Maximum seconds to wait for thread.
        """
        if not self._is_running:
            return
        
        logger.info("Stopping NOLAManager...")
        self._is_running = False
        self._stop_event.set()
        self._stt.stop()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        
        logger.info("NOLAManager stopped")
    
    # =========================================================================
    # Processing Loop (True Hardware Mute)
    # =========================================================================
    
    def _process_loop(self) -> None:
        """Main voice processing loop - runs in background thread.
        
        Features:
        - True hardware mute: stream closes when paused
        - Auto-restart on failure with retry limit
        - Clean shutdown on stop
        """
        retry_count = 0
        max_retries = 5
        
        logger.info("🎙️ Voice Loop Started")
        
        while self._is_running and not self._stop_event.is_set():
            # =================================================================
            # WAIT STATE: If paused, don't touch hardware - just sleep
            # =================================================================
            if self._is_paused:
                time.sleep(0.2)
                continue
            
            # =================================================================
            # ACTIVE STATE: Open hardware stream and process speech
            # =================================================================
            try:
                logger.info("🎤 Opening Microphone Stream...")
                retry_count = 0  # Reset on successful start
                
                for text in self._stt.stream():
                    # CHECKPOINT 1: Stop requested
                    if not self._is_running or self._stop_event.is_set():
                        logger.info("🔇 Closing Microphone Stream (shutdown)...")
                        return
                    
                    # CHECKPOINT 2: Pause requested - BREAK to close hardware
                    if self._is_paused:
                        logger.info("🔇 Closing Microphone Stream (paused)...")
                        self._stt.stop()  # Signal STT to stop
                        break  # Exit loop, closes stream via 'with' block
                    
                    # Process recognized text
                    text = text.lower().strip()
                    if not text:
                        continue
                    
                    self._handle_input(text)
                
                # After for loop ends (break or natural end), check if we should restart
                if self._is_paused:
                    logger.info("🔇 Microphone Hardware Released")
                    continue  # Go back to wait state
                    
            except OSError as e:
                retry_count += 1
                logger.error(f"Audio device error (attempt {retry_count}/{max_retries}): {e}", exc_info=True)
                
                if retry_count >= max_retries:
                    logger.error("Max retries reached. Voice input disabled.")
                    break
                
                time.sleep(1.0)
                logger.info("Restarting STT stream...")
            except Exception as e:
                retry_count += 1
                logger.error(f"STT stream error (attempt {retry_count}/{max_retries}): {e}", exc_info=True)
                
                if retry_count >= max_retries:
                    logger.error("Max retries reached. Voice input disabled.")
                    break
                
                time.sleep(1.0)
                logger.info("Restarting STT stream...")
    
    def _handle_input(self, text: str) -> None:
        """Handle recognized speech based on current state.
        
        Args:
            text: Recognized text (lowercase, stripped).
        """
        if self.state == STATE_ASLEEP:
            # Only listen for wake words
            wake_triggered, command = self._check_wake_word(text)
            
            if wake_triggered:
                self._wake_up()
                
                if command:
                    # One-shot: "Hey Nia what time is it?"
                    logger.info(f"One-shot command: '{text}' → '{command}'")
                    self._enqueue_input(command)
                else:
                    # Just wake word, wait for next utterance
                    logger.debug("Wake word only, awaiting command...")
                    
        elif self.state == STATE_AWAKE:
            # Check for sleep command
            if self._is_sleep_command(text):
                self._go_to_sleep()
                return
            
            # Check active window timeout
            if not self.config.auto_sleep_after_command:
                elapsed = time.time() - self._last_wake_time
                if elapsed > self.config.active_window_seconds:
                    logger.info(f"Timeout after {elapsed:.1f}s - going to sleep")
                    self._go_to_sleep()
                    return
            
            # Accept command
            logger.info(f"Command received: '{text}'")
            self._enqueue_input(text)
            
            # Auto-sleep after command
            if self.config.auto_sleep_after_command:
                self._go_to_sleep()
    
    def _check_wake_word(self, text: str) -> tuple:
        """Check if text contains wake word.
        
        Returns:
            (wake_triggered, remaining_command)
        """
        for ww in self._wake_words:
            if text.startswith(ww):
                command = text[len(ww):].strip()
                return True, command
            elif ww in text:
                return True, ""
        return False, ""
    
    def _is_sleep_command(self, text: str) -> bool:
        """Check if text is a sleep command."""
        return any(cmd in text for cmd in self._sleep_commands)
    
    def _wake_up(self) -> None:
        """Transition to AWAKE state."""
        old_state = self.state
        self.state = STATE_AWAKE
        self._last_wake_time = time.time()
        
        logger.info(f"State transition: {old_state} → {self.state}")
        
        if self._on_wake:
            try:
                self._on_wake()
            except TypeError as e:
                logger.warning(f"on_wake callback type error: {e}")
            except RuntimeError as e:
                logger.warning(f"on_wake callback runtime error: {e}")
    
    def _go_to_sleep(self) -> None:
        """Transition to ASLEEP state."""
        old_state = self.state
        self.state = STATE_ASLEEP
        logger.info(f"State transition: {old_state} → {self.state}")
    
    def _enqueue_input(self, text: str) -> None:
        """Add command to input queue."""
        try:
            self._input_queue.put_nowait(text)
        except queue.Full:
            try:
                self._input_queue.get_nowait()
                self._input_queue.put_nowait(text)
            except queue.Empty:
                pass
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def get_input(self, timeout: Optional[float] = None) -> Optional[str]:
        """Get next voice command (after wake word processing).
        
        Args:
            timeout: Seconds to wait. None for non-blocking.
            
        Returns:
            Command text or None.
        """
        try:
            if timeout is None:
                return self._input_queue.get_nowait()
            return self._input_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def speak(self, text: str, block_listening: bool = True) -> bool:
        """Speak text via TTS.
        
        Args:
            text: Text to speak.
            block_listening: Pause listening while speaking.
            
        Returns:
            True if successful.
        """
        if not text:
            return False
        
        # Remember if already paused BEFORE we auto-pause for TTS
        was_user_paused = self._user_paused
        
        if block_listening:
            self._is_paused = True
        
        try:
            result = self._tts.speak(text)
        finally:
            if block_listening:
                time.sleep(0.3)
                # ONLY resume if user didn't explicitly pause
                if not was_user_paused:
                    self._is_paused = False
        
        return result
    
    def stop_speaking(self) -> None:
        """Stop current TTS playback."""
        self._tts.stop()
    
    def pause_listening(self) -> None:
        """Pause microphone input (releases hardware)."""
        self._is_paused = True
        self._user_paused = True  # Mark as user-initiated
        self._stt.stop()  # Signal stream to close
        logger.info("🔇 NOLA: Microphone paused (hardware released)")
    
    def resume_listening(self) -> None:
        """Resume microphone input (reopens hardware)."""
        self._is_paused = False
        self._user_paused = False  # Clear user-pause flag
        logger.info("🎤 NOLA: Microphone resumed (awaiting stream open)")
    
    def wake(self) -> None:
        """Manually wake up."""
        if self.state == STATE_ASLEEP:
            self._wake_up()
    
    def sleep(self) -> None:
        """Manually go to sleep."""
        if self.state == STATE_AWAKE:
            self._go_to_sleep()
    
    def set_wake_words(self, words: List[str]) -> None:
        """Update wake words."""
        self._wake_words = [w.lower().strip() for w in words]
        logger.info("Wake words updated: %s", self._wake_words)
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def is_running(self) -> bool:
        """Check if NOLA is running."""
        return self._is_running
    
    @property
    def is_awake(self) -> bool:
        """Check if in AWAKE state."""
        return self.state == STATE_AWAKE
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._tts.is_speaking


# =============================================================================
# Singleton Instance
# =============================================================================

_nola_manager_instance: Optional[NOLAManager] = None


def get_nola_manager(config: Optional[NOLAConfig] = None) -> NOLAManager:
    """Get or create the NOLAManager singleton.
    
    Args:
        config: Configuration (only used on first call).
        
    Returns:
        The singleton NOLAManager instance.
    """
    global _nola_manager_instance
    if _nola_manager_instance is None:
        _nola_manager_instance = NOLAManager(config=config)
    return _nola_manager_instance


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "NOLAConfig",
    "NOLAManager",
    "STATE_ASLEEP",
    "STATE_AWAKE",
    "get_nola_manager",
]
