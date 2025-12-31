"""N.O.L.A. Security Module - Input Sanitization & Command Filtering.

This module provides security-focused input processing for voice commands,
implementing pattern-based detection of potentially dangerous operations.

Classes:
    SecurityLevel: Enum for input classification (SAFE, WARNING, BLOCKED)
    SanitizedInput: Container for processed input with security metadata
    InputSanitizer: Main security filter with configurable patterns
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

# Configure module logger
logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security classification for input commands."""
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass
class SanitizedInput:
    """Container for processed voice input."""
    text: str
    original_text: str
    security_level: SecurityLevel = SecurityLevel.SAFE
    wake_word_detected: bool = False
    timestamp: float = 0.0
    blocked_reason: Optional[str] = None
    
    @property
    def is_safe(self) -> bool:
        """Check if input passed security checks."""
        return self.security_level == SecurityLevel.SAFE
    
    @property
    def is_blocked(self) -> bool:
        """Check if input was blocked for security."""
        return self.security_level == SecurityLevel.BLOCKED
    
    @property
    def is_warning(self) -> bool:
        """Check if input triggered a security warning."""
        return self.security_level == SecurityLevel.WARNING


class InputSanitizer:
    """Security layer for voice input sanitization.
    
    Implements pattern-based detection of potentially dangerous commands
    to prevent accidental or malicious system damage via voice control.
    
    Attributes:
        DEFAULT_BLOCKED_PATTERNS: Commands that are never executed
        DEFAULT_WARNING_PATTERNS: Commands that trigger caution flags
    
    Example:
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize("rm -rf /")
        assert result.is_blocked
    """
    
    # Dangerous patterns that should be BLOCKED (never executed)
    DEFAULT_BLOCKED_PATTERNS: List[str] = [
        # Unix/Linux dangerous commands
        r"\brm\s+(-rf?|--recursive)\b",
        r"\brm\s+-rf?\s*/\b",
        r"\bsudo\s+rm\b",
        r"\bmkfs\b",
        r"\bdd\s+if=",
        r"\b:()\s*{\s*:\s*\|\s*:\s*&\s*}\s*;\s*:",  # Fork bomb
        r"\bchmod\s+(-R\s+)?777\s+/",
        r"\bchown\s+.*\s+/",
        r"\b>\s*/dev/sd[a-z]",
        
        # Windows dangerous commands
        r"\bformat\s+[a-z]:\b",
        r"\bdel\s+/[sfq]\s+[a-z]:\\",
        r"\brd\s+/s\s+/q\s+[a-z]:\\",
        r"\bdiskpart\b",
        r"\bcmd\s*/c\s*del\b",
        r"\bpowershell\s.*-enc",
        r"\breg\s+delete\s+hk[a-z]+\\",
        
        # Network/Remote dangerous
        r"\bnc\s+-e\b",  # Netcat reverse shell
        r"\bcurl.*\|\s*(ba)?sh",
        r"\bwget.*\|\s*(ba)?sh",
        
        # Credential access
        r"\bcat\s+/etc/(passwd|shadow)",
        r"\bmimikatz\b",
        r"\bsecretsdump\b",
    ]
    
    # Warning patterns (logged but allowed with caution flag)
    DEFAULT_WARNING_PATTERNS: List[str] = [
        r"\bsudo\b",
        r"\bsu\s+-\b",
        r"\badmin(istrator)?\b",
        r"\bpassword\b",
        r"\bapi[_-]?key\b",
        r"\bsecret\b",
        r"\btoken\b",
        r"\bcredentials?\b",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bkill\s+-9\b",
        r"\bpkill\b",
    ]
    
    def __init__(
        self,
        blocked_patterns: Optional[List[str]] = None,
        warning_patterns: Optional[List[str]] = None,
        case_sensitive: bool = False,
    ) -> None:
        """Initialize the sanitizer.
        
        Args:
            blocked_patterns: Additional patterns to block (merged with defaults).
            warning_patterns: Additional patterns to warn about.
            case_sensitive: Whether pattern matching is case-sensitive.
        """
        self._flags = 0 if case_sensitive else re.IGNORECASE
        
        # Compile blocked patterns
        all_blocked = self.DEFAULT_BLOCKED_PATTERNS + (blocked_patterns or [])
        self._blocked_patterns: List[re.Pattern] = []
        for pattern in all_blocked:
            try:
                self._blocked_patterns.append(re.compile(pattern, self._flags))
            except re.error as exc:
                logger.warning("Invalid blocked pattern '%s': %s", pattern, exc)
        
        # Compile warning patterns
        all_warning = self.DEFAULT_WARNING_PATTERNS + (warning_patterns or [])
        self._warning_patterns: List[re.Pattern] = []
        for pattern in all_warning:
            try:
                self._warning_patterns.append(re.compile(pattern, self._flags))
            except re.error as exc:
                logger.warning("Invalid warning pattern '%s': %s", pattern, exc)
        
        # Track blocked attempts for security auditing
        self._blocked_attempts: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def sanitize(self, text: str) -> SanitizedInput:
        """Analyze and sanitize input text.
        
        Args:
            text: Raw input text from ASR.
            
        Returns:
            SanitizedInput with security classification.
        """
        if not text:
            return SanitizedInput(
                text="",
                original_text="",
                security_level=SecurityLevel.SAFE,
                timestamp=time.time(),
            )
        
        original = text.strip()
        normalized = original.lower()
        
        # Check for blocked patterns
        for pattern in self._blocked_patterns:
            match = pattern.search(normalized)
            if match:
                blocked_text = match.group()
                logger.warning(
                    "SECURITY: Blocked dangerous input: '%s' (matched: %s)",
                    original[:100], blocked_text
                )
                
                # Record for audit
                with self._lock:
                    self._blocked_attempts.append({
                        "timestamp": time.time(),
                        "input": original,
                        "matched_pattern": pattern.pattern,
                        "matched_text": blocked_text,
                    })
                
                return SanitizedInput(
                    text="",
                    original_text=original,
                    security_level=SecurityLevel.BLOCKED,
                    timestamp=time.time(),
                    blocked_reason=f"Dangerous command detected: {blocked_text}",
                )
        
        # Check for warning patterns
        for pattern in self._warning_patterns:
            if pattern.search(normalized):
                logger.info(
                    "SECURITY: Warning-level input detected: '%s'",
                    original[:100]
                )
                return SanitizedInput(
                    text=original,
                    original_text=original,
                    security_level=SecurityLevel.WARNING,
                    timestamp=time.time(),
                )
        
        # Input is safe
        return SanitizedInput(
            text=original,
            original_text=original,
            security_level=SecurityLevel.SAFE,
            timestamp=time.time(),
        )
    
    def add_blocked_pattern(self, pattern: str) -> bool:
        """Add a new blocked pattern at runtime.
        
        Args:
            pattern: Regex pattern to block.
            
        Returns:
            True if pattern was added successfully.
        """
        try:
            compiled = re.compile(pattern, self._flags)
            self._blocked_patterns.append(compiled)
            return True
        except re.error as exc:
            logger.warning("Invalid pattern '%s': %s", pattern, exc)
            return False
    
    def add_warning_pattern(self, pattern: str) -> bool:
        """Add a new warning pattern at runtime."""
        try:
            compiled = re.compile(pattern, self._flags)
            self._warning_patterns.append(compiled)
            return True
        except re.error as exc:
            logger.warning("Invalid pattern '%s': %s", pattern, exc)
            return False
    
    def get_blocked_attempts(self) -> List[Dict[str, Any]]:
        """Get list of blocked command attempts (for security audit)."""
        with self._lock:
            return list(self._blocked_attempts)
    
    def clear_audit_log(self) -> int:
        """Clear the blocked attempts log.
        
        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._blocked_attempts)
            self._blocked_attempts.clear()
            return count


__all__ = [
    "SecurityLevel",
    "SanitizedInput",
    "InputSanitizer",
]
