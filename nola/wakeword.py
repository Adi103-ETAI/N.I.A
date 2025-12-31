"""N.O.L.A. Wake Word Module - Voice Activation Detection.

This module provides wake word detection for voice-activated assistants,
enabling hands-free activation through trigger phrases.

Classes:
    WakeWordDetector: Pattern-matching wake word detector with timeout
"""
from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional, Set, Tuple

# Configure module logger
logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Simple wake word detection for voice activation.
    
    Implements a basic pattern-matching wake word detector with
    optional timeout for automatic deactivation.
    
    Features:
        - Multiple wake words support
        - Configurable timeout after activation
        - Optional wake word stripping from returned text
        - Thread-safe operation
    
    Example:
        detector = WakeWordDetector(wake_words=["jarvis", "nia"])
        
        is_active, text = detector.check("jarvis what time is it")
        # is_active = True, text = "what time is it"
        
        is_active, text = detector.check("hello there")  
        # is_active = False (or True if within timeout window)
    """
    
    # Common follow-up punctuation to strip after wake word
    STRIP_PUNCTUATION = {",", ".", "!", "?", ";", ":"}
    
    def __init__(
        self,
        wake_words: Optional[List[str]] = None,
        timeout: float = 30.0,
        strip_wake_word: bool = True,
    ) -> None:
        """Initialize the wake word detector.
        
        Args:
            wake_words: List of trigger phrases (case-insensitive).
            timeout: Seconds to stay active after wake word detection.
            strip_wake_word: Whether to remove wake word from returned text.
        """
        self._wake_words: Set[str] = set()
        for word in (wake_words or ["jarvis", "nia", "hey nia"]):
            self._wake_words.add(word.lower().strip())
        
        self._timeout = timeout
        self._strip_wake_word = strip_wake_word
        self._last_activation: float = 0.0
        self._is_active = False
        self._lock = threading.Lock()
    
    def check(self, text: str) -> Tuple[bool, str]:
        """Check for wake word and return activation status.
        
        Args:
            text: Input text to check.
            
        Returns:
            Tuple of (is_activated, processed_text).
            processed_text has wake word removed if strip_wake_word is True.
        """
        if not text:
            return False, ""
        
        normalized = text.lower().strip()
        current_time = time.time()
        
        with self._lock:
            # Check if still in active window
            if self._is_active and (current_time - self._last_activation) < self._timeout:
                return True, text.strip()
            
            # Look for wake word at start of input
            for wake_word in self._wake_words:
                if normalized.startswith(wake_word):
                    self._is_active = True
                    self._last_activation = current_time
                    
                    logger.debug("Wake word detected: '%s'", wake_word)
                    
                    # Strip wake word from text if configured
                    if self._strip_wake_word:
                        remaining = text.strip()[len(wake_word):].strip()
                        # Remove common follow-up punctuation
                        for prefix in self.STRIP_PUNCTUATION:
                            if remaining.startswith(prefix):
                                remaining = remaining[1:].strip()
                        return True, remaining
                    
                    return True, text.strip()
            
            # No wake word and not in active window
            self._is_active = False
            return False, text.strip()
    
    def activate(self) -> None:
        """Manually activate the wake word listener."""
        with self._lock:
            self._is_active = True
            self._last_activation = time.time()
            logger.debug("Wake word manually activated")
    
    def deactivate(self) -> None:
        """Manually deactivate the wake word listener."""
        with self._lock:
            self._is_active = False
            self._last_activation = 0.0
            logger.debug("Wake word deactivated")
    
    def is_active(self) -> bool:
        """Check if currently in active listening mode."""
        with self._lock:
            if not self._is_active:
                return False
            if (time.time() - self._last_activation) >= self._timeout:
                self._is_active = False
                return False
            return True
    
    def extend_timeout(self, seconds: Optional[float] = None) -> None:
        """Extend the active window.
        
        Args:
            seconds: Additional seconds to add. If None, resets to full timeout.
        """
        with self._lock:
            if self._is_active:
                self._last_activation = time.time()
                logger.debug("Wake word timeout extended")
    
    def time_remaining(self) -> float:
        """Get remaining time in active window.
        
        Returns:
            Seconds remaining, or 0 if not active.
        """
        with self._lock:
            if not self._is_active:
                return 0.0
            remaining = self._timeout - (time.time() - self._last_activation)
            return max(0.0, remaining)
    
    def set_wake_words(self, words: List[str]) -> None:
        """Update wake words.
        
        Args:
            words: New list of wake words.
        """
        with self._lock:
            self._wake_words = {w.lower().strip() for w in words if w.strip()}
            logger.info("Wake words updated: %s", self._wake_words)
    
    def add_wake_word(self, word: str) -> None:
        """Add a single wake word.
        
        Args:
            word: Wake word to add.
        """
        if word and word.strip():
            with self._lock:
                self._wake_words.add(word.lower().strip())
    
    def remove_wake_word(self, word: str) -> bool:
        """Remove a wake word.
        
        Args:
            word: Wake word to remove.
            
        Returns:
            True if word was removed.
        """
        with self._lock:
            normalized = word.lower().strip()
            if normalized in self._wake_words:
                self._wake_words.discard(normalized)
                return True
            return False
    
    @property
    def wake_words(self) -> List[str]:
        """Get current wake words."""
        with self._lock:
            return list(self._wake_words)
    
    @property
    def timeout(self) -> float:
        """Get current timeout setting."""
        return self._timeout
    
    @timeout.setter
    def timeout(self, value: float) -> None:
        """Set timeout value."""
        self._timeout = max(0.0, value)


__all__ = [
    "WakeWordDetector",
]
