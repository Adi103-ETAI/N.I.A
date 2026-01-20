"""N.O.L.A. - Neural Operator for Language & Audio.

A modular voice I/O system for NIA using Edge TTS and Vosk STT.

Package Structure:
    nola/
    ├── __init__.py      # This file - public API exports & dependency check
    ├── manager.py       # NOLAManager orchestrator with wake word state machine
    ├── security.py      # Input sanitization & command filtering
    └── io.py            # HybridTTS (Edge+Piper) & VoskSTT implementations

Quick Start:
    from nola import get_nola_manager
    
    manager = get_nola_manager()
    manager.start()
    
    while True:
        text = manager.get_input(timeout=0.5)
        if text:
            response = brain.process(text)
            manager.speak(response)
    
    manager.stop()

Components:
    NOLAManager: Main orchestrator with ASLEEP/AWAKE state machine
    NOLAConfig: Configuration dataclass for NOLAManager
    HybridTTS: Edge TTS (primary) + Piper (fallback)
    VoskSTT: Offline speech recognition
    get_nola_manager: Singleton accessor for NOLAManager

Version: 2.5.0 (Edge TTS + Vosk Stack)
"""
from __future__ import annotations

import sys
import importlib.util
from core.logger import setup_logger

logger = setup_logger("NOLA")

# =============================================================================
# Dependency Verification
# =============================================================================

REQUIRED_DEPS = ['vosk', 'sounddevice', 'numpy', 'requests']


def check_dependencies() -> bool:
    """Check if all required audio dependencies are installed.
    
    Returns:
        True if all dependencies are available, exits with error otherwise.
    """
    missing = []
    
    for dep in REQUIRED_DEPS:
        if importlib.util.find_spec(dep) is None:
            missing.append(dep)
    
    if missing:
        logger.error("Missing NOLA dependencies: %s", ", ".join(missing))
        logger.error("Fix with: pip install %s", " ".join(missing))
        sys.exit(1)
    
    return True


# Run dependency check on import (silent on success, exits on failure)
check_dependencies()


# =============================================================================
# Package Exports
# =============================================================================

# Manager and config
from .manager import (
    NOLAManager,
    NOLAConfig,
    get_nola_manager,
)

# Security components
from .security import (
    SecurityLevel,
    SanitizedInput,
    InputSanitizer,
)

# I/O components
from .io import (
    RecognitionResult,
    HybridTTS,
    VoskSTT,
    AsyncEar,
    AsyncTTS,
    get_async_ear,
    get_async_tts,
)

# Package metadata
__version__ = "2.5.0"
__author__ = "NIA Team"
__all__ = [
    # Core
    "NOLAManager",
    "NOLAConfig",
    "get_nola_manager",
    
    # Security
    "SecurityLevel",
    "SanitizedInput", 
    "InputSanitizer",
    
    # I/O
    "RecognitionResult",
    "HybridTTS",
    "VoskSTT",
    "AsyncEar",
    "AsyncTTS",
    "get_async_ear",
    "get_async_tts",
    
    # Utilities
    "check_dependencies",
]
