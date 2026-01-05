"""N.I.A. Centralized Logging Module.

Provides a unified logging setup with dual output:
- Console: INFO level for user visibility
- File: DEBUG level with rotation for debugging

Usage:
    from core.logger import setup_logger
    
    logger = setup_logger("BRAIN")
    logger.info("System initialized")
    logger.debug("Detailed debug info")
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


# =============================================================================
# Logger Cache (prevent duplicate handlers)
# =============================================================================

_loggers: dict[str, logging.Logger] = {}


# =============================================================================
# Setup Function
# =============================================================================

def setup_logger(name: str, console_level: int = logging.INFO, file_level: int = logging.DEBUG) -> logging.Logger:
    """Create or retrieve a configured logger.
    
    Args:
        name: Logger name (e.g., "BRAIN", "NOLA", "TARA").
        console_level: Logging level for console output.
        file_level: Logging level for file output.
        
    Returns:
        Configured logging.Logger instance.
        
    Example:
        logger = setup_logger("MAIN")
        logger.info("Application started")
    """
    # Return cached logger if exists
    if name in _loggers:
        return _loggers[name]
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all, handlers filter
    logger.propagate = False  # Prevent duplicate logs from root
    
    # Only add handlers if none exist
    if not logger.handlers:
        # Console Handler (INFO+)
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
# Convenience Getters
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one.
    
    Alias for setup_logger() for cleaner imports.
    """
    return setup_logger(name)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "setup_logger",
    "get_logger",
    "LOG_DIR",
    "LOG_FILE",
]
