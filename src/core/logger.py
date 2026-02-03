"""N.I.A. Centralized Logging Module (v2.0 - dictConfig Architecture).

Uses logging.config.dictConfig for centralized control, eliminating the
timing issues where module-level logger instantiation bypassed filters.

Key Architecture Changes:
    - All logger configuration is defined in LOGGING_CONFIG dictionary
    - Noisy sources (TARA.Nodes, NIA.Nodes, etc.) are set to WARNING by default
    - setup_logger() no longer creates handlers - just returns configured logger
    - init_logging() dynamically adjusts config before applying dictConfig

Usage:
    # In main.py (BEFORE any other imports):
    from core.logger import init_logging
    init_logging(debug=args.debug)
    
    # In any module:
    from core.logger import setup_logger
    logger = setup_logger("BRAIN")
    logger.info("System initialized")
"""
from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Optional

# =============================================================================
# Configuration Constants
# =============================================================================

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "nia.log"
MAX_BYTES = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 3

# Format strings
CONSOLE_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
FILE_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d]: %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Noisy third-party libraries to always suppress
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

# Internal components that are chatty at INFO level
NOISY_SOURCES = [
    "TARA.Nodes", 
    "TARA.Interface", 
    "NIA.Graph", 
    "NIA.Nodes", 
    "MEMORY",
]

# =============================================================================
# Global State
# =============================================================================

_debug_mode: bool = False
_initialized: bool = False


# =============================================================================
# Logging Configuration Dictionary (The Heart of the Fix)
# =============================================================================

def _build_logging_config(debug: bool = False) -> dict:
    """Build the logging configuration dictionary.
    
    Args:
        debug: If True, sets console to DEBUG and lifts noisy source restrictions.
        
    Returns:
        Complete logging configuration dictionary for dictConfig.
    """
    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Base console level
    console_level = "DEBUG" if debug else "INFO"
    noisy_level = "DEBUG" if debug else "WARNING"
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        
        # === FORMATTERS ===
        "formatters": {
            "console": {
                "format": CONSOLE_FORMAT,
                "datefmt": DATE_FORMAT,
            },
            "file": {
                "format": FILE_FORMAT,
                "datefmt": DATE_FORMAT,
            },
        },
        
        # === HANDLERS ===
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": console_level,
                "formatter": "console",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "file",
                "filename": str(LOG_FILE),
                "maxBytes": MAX_BYTES,
                "backupCount": BACKUP_COUNT,
                "encoding": "utf-8",
            },
        },
        
        # === LOGGERS ===
        # Explicit configuration for noisy internal sources
        "loggers": {
            # Internal noisy sources - suppress at INFO level unless debugging
            "TARA.Nodes": {
                "level": noisy_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "TARA.Interface": {
                "level": noisy_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "TARA.Workflow": {
                "level": noisy_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "TARA.Prompts": {
                "level": noisy_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "NIA.Graph": {
                "level": noisy_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "NIA.Nodes": {
                "level": noisy_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "MEMORY": {
                "level": noisy_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            # Third-party noisy libraries - always WARNING
            **{lib: {"level": "WARNING", "propagate": False} for lib in NOISY_LIBRARIES},
        },
        
        # === ROOT LOGGER ===
        "root": {
            "level": "DEBUG",  # Capture all, handlers filter
            "handlers": ["console", "file"],
        },
    }
    
    return config


# =============================================================================
# Initialization Function (The Trigger)
# =============================================================================

def init_logging(debug: bool = False) -> None:
    """Initialize global logging configuration using dictConfig.
    
    MUST be called at application startup BEFORE any other module imports
    that use setup_logger(). This ensures all loggers pick up the correct
    configuration.
    
    Args:
        debug: If True, console output shows DEBUG level logs and
               noisy sources are not suppressed.
    
    Example:
        # In main.py, at the very start:
        from core.logger import init_logging
        init_logging(debug=args.debug)
        
        # Then import other modules...
        from core.engine import NIAAssistant
    """
    global _debug_mode, _initialized
    
    _debug_mode = debug
    
    # Build configuration with appropriate levels
    config = _build_logging_config(debug=debug)
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    _initialized = True
    
    # Log initialization (will use new config)
    logger = logging.getLogger("SYSTEM")
    if debug:
        logger.debug("✅ Logging initialized (DEBUG mode)")
    else:
        logger.debug("✅ Logging initialized (INFO mode, noisy sources suppressed)")


def is_debug_mode() -> bool:
    """Check if global debug mode is enabled.
    
    Returns:
        True if debug mode was enabled via init_logging(debug=True).
    """
    return _debug_mode


def is_initialized() -> bool:
    """Check if logging has been initialized.
    
    Returns:
        True if init_logging() has been called.
    """
    return _initialized


# =============================================================================
# Logger Factory (Simplified - No Handler Creation)
# =============================================================================

def setup_logger(name: str, console_level: Optional[int] = None, file_level: int = logging.DEBUG) -> logging.Logger:
    """Get a configured logger by name.
    
    BACKWARD COMPATIBLE: This function signature matches the old API,
    but internal behavior has changed. The console_level and file_level
    parameters are now ignored - all configuration comes from dictConfig.
    
    Args:
        name: Logger name (e.g., "BRAIN", "NOLA", "TARA").
        console_level: DEPRECATED - Ignored. Levels set via dictConfig.
        file_level: DEPRECATED - Ignored. Levels set via dictConfig.
        
    Returns:
        Configured logging.Logger instance.
        
    Example:
        logger = setup_logger("MAIN")
        logger.info("Application started")
        logger.debug("This shows only if --debug flag was used")
    """
    # If init_logging hasn't been called yet, apply default config
    # This handles the case where modules are imported before main() runs
    if not _initialized:
        _apply_fallback_config()
    
    return logging.getLogger(name)


def _apply_fallback_config() -> None:
    """Apply minimal fallback config if init_logging wasn't called yet.
    
    This prevents the "No handlers could be found" warning when modules
    import and use loggers before main.py calls init_logging().
    """
    global _initialized
    
    if _initialized:
        return
    
    # Apply default (non-debug) configuration
    config = _build_logging_config(debug=False)
    logging.config.dictConfig(config)
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one.
    
    Alias for setup_logger() for cleaner imports.
    """
    return setup_logger(name)


# =============================================================================
# Runtime Level Control
# =============================================================================

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
    
    # Get root logger and update console handler
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not hasattr(handler, 'baseFilename'):
            handler.setLevel(level)
    
    # Also update noisy source loggers if switching to debug
    if _debug_mode:
        for source in NOISY_SOURCES:
            logging.getLogger(source).setLevel(logging.DEBUG)
    else:
        for source in NOISY_SOURCES:
            logging.getLogger(source).setLevel(logging.WARNING)


def set_debug_mode(enabled: bool) -> None:
    """Enable or disable debug mode for all loggers.
    
    Convenience wrapper around set_console_level().
    
    Args:
        enabled: True to enable DEBUG output, False for INFO only.
    """
    set_console_level(logging.DEBUG if enabled else logging.INFO)


# =============================================================================
# Event-Driven Logging (Async) - Unchanged
# =============================================================================

async def _handle_log_event(payload: dict) -> None:
    """Handle asynchronous log events from the EventBus.
    
    Payload Structure:
        {
            "level": "INFO" | "DEBUG" | "ERROR" | "WARNING",
            "source": "LoggerName",
            "message": "Log message content"
        }
    """
    try:
        # Extract payload
        level_str = payload.get("level", "INFO").upper()
        source = payload.get("source", "EVENT_LOG")
        message = payload.get("message", "")
        
        # Map string level to logging constant
        level = getattr(logging, level_str, logging.INFO)
        
        # Noise Filter: If NOT in debug mode, suppress INFO from noisy sources
        if not _debug_mode and level == logging.INFO and source in NOISY_SOURCES:
            return  # Silenced by policy
        
        # Get logger and log
        logger = logging.getLogger(source)
        logger.log(level, message)
        
    except Exception as e:
        # Failsafe: Don't crash the bus listener
        sys_logger = logging.getLogger("SYSTEM")
        sys_logger.error(f"Failed to process log event: {e}")


def start_log_listener() -> None:
    """Initialize the Event-Driven Logger Subscriber.
    
    Subscribes to 'log:entry' on the central EventBus.
    Must be called after EventBus is ready.
    """
    # Local import prevents circular dependency
    from core.event_bus import get_event_bus
    
    try:
        bus = get_event_bus()
        bus.subscribe("log:entry", _handle_log_event)
        
        # Announce subscription
        logger = logging.getLogger("SYSTEM")
        logger.info("✅ Event-Driven Logger subscribed to 'log:entry'")
        
    except ImportError:
        print("⚠️ EventBus not found. Async logging disabled.")
    except Exception as e:
        print(f"❌ Failed to start log listener: {e}")


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
    "is_initialized",
    "start_log_listener",
    "LOG_DIR",
    "LOG_FILE",
    "NOISY_SOURCES",
    "NOISY_LIBRARIES",
]
