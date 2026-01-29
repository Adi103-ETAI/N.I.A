"""N.I.A. Centralized Logging Module.

Provides a unified logging setup with dual output:
- Console: INFO level (DEBUG if --debug flag is set)
- File: DEBUG level with rotation for debugging

Global Debug Mode:
    Call init_logging(debug=True) at application startup (before other imports)
    to enable DEBUG level console output across all modules.

Usage:
    # In main.py (before other imports):
    from core.logger import init_logging
    init_logging(debug=args.debug)
    
    # In any module:
    from core.logger import setup_logger
    
    logger = setup_logger("BRAIN")
    logger.info("System initialized")
    logger.debug("Detailed debug info")  # Only visible if --debug flag
    logger.error("Something failed", exc_info=True)
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# =============================================================================
# Configuration
# =============================================================================

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "nia.log"
MAX_BYTES = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 3

# Format strings
CONSOLE_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
FILE_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d]: %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Noisy libraries to suppress even in debug mode
NOISY_LIBRARIES = [
    "httpx",
    "httpcore", 
    "urllib3",
    "asyncio",
    "vosk",
    "PIL",
    "comtypes",
    "uiautomation",
    "pyautogui",
    "pygame",
    "chromadb",
    "langchain",
    "langsmith",
    "openai",
    "numba",
    "chardet",
]


# =============================================================================
# Global Debug State
# =============================================================================

_debug_mode: bool = False
_initialized: bool = False
_loggers: dict[str, logging.Logger] = {}


def init_logging(debug: bool = False) -> None:
    """Initialize global logging configuration.
    
    MUST be called at application startup BEFORE any other module imports
    that use setup_logger(). This sets the global debug mode flag.
    
    Args:
        debug: If True, console output shows DEBUG level logs.
               If False (default), console shows INFO level and above.
    
    Example:
        # In main.py, at the very start:
        from core.logger import init_logging
        init_logging(debug=args.debug)
        
        # Then import other modules...
        from core.engine import NIAAssistant
    """
    global _debug_mode, _initialized
    
    _debug_mode = debug
    _initialized = True
    
    # Silence noisy third-party libraries (even in debug mode)
    for lib_name in NOISY_LIBRARIES:
        logging.getLogger(lib_name).setLevel(logging.WARNING)
    
    # If loggers already exist (e.g., re-init), update their console level
    if _loggers:
        set_console_level(logging.DEBUG if debug else logging.INFO)


def is_debug_mode() -> bool:
    """Check if global debug mode is enabled.
    
    Returns:
        True if debug mode was enabled via init_logging(debug=True).
    """
    return _debug_mode


# =============================================================================
# Setup Function
# =============================================================================

def setup_logger(
    name: str, 
    console_level: Optional[int] = None, 
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """Create or retrieve a configured logger.
    
    Args:
        name: Logger name (e.g., "BRAIN", "NOLA", "TARA").
        console_level: Logging level for console output.
                       If None, uses DEBUG (if global debug mode) or INFO.
        file_level: Logging level for file output (default: DEBUG).
        
    Returns:
        Configured logging.Logger instance.
        
    Example:
        logger = setup_logger("MAIN")
        logger.info("Application started")
        logger.debug("This shows only if --debug flag was used")
    """
    # Return cached logger if exists
    if name in _loggers:
        return _loggers[name]
    
    # Determine console level based on global debug mode
    if console_level is None:
        console_level = logging.DEBUG if _debug_mode else logging.INFO
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all, handlers filter
    logger.propagate = False  # Prevent duplicate logs from root
    
    # Only add handlers if none exist
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(console_handler)
        
        # File Handler (DEBUG+ with rotation)
        try:
            _ensure_log_dir()
            file_handler = RotatingFileHandler(
                LOG_FILE,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
            logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            # Fall back to console-only if file logging fails
            logger.warning(f"Could not create file handler: {e}. Using console only.")
    
    # Cache and return
    _loggers[name] = logger
    return logger


def _ensure_log_dir() -> None:
    """Create logs directory if it doesn't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Runtime Level Control
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one.
    
    Alias for setup_logger() for cleaner imports.
    """
    return setup_logger(name)


def set_console_level(level: int = logging.INFO) -> None:
    """Set the logging level for all console handlers.
    
    Used to toggle debug mode at runtime (after initialization).
    
    Args:
        level: New logging level (e.g., logging.DEBUG or logging.INFO).
        
    Example:
        # Enable debug mode at runtime
        set_console_level(logging.DEBUG)
        
        # Disable debug mode
        set_console_level(logging.INFO)
    """
    global _debug_mode
    _debug_mode = (level == logging.DEBUG)
    
    for logger in _loggers.values():
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
                handler.setLevel(level)


def set_debug_mode(enabled: bool) -> None:
    """Enable or disable debug mode for all loggers.
    
    Convenience wrapper around set_console_level().
    
    Args:
        enabled: True to enable DEBUG output, False for INFO only.
    """
    set_console_level(logging.DEBUG if enabled else logging.INFO)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "init_logging",
    "setup_logger",
    "get_logger",
    "set_console_level",
    "set_debug_mode",
    "is_debug_mode",
    "LOG_DIR",
    "LOG_FILE",
]
