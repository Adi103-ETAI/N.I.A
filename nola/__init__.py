"""N.O.L.A. - Neural Operator for Language & Audio.

A modular voice I/O system for NIA using Vosk (STT) and Piper (TTS).

Package Structure:
    nola/
    ├── __init__.py      # This file - public API exports & dependency check
    ├── manager.py       # NOLAManager orchestrator
    ├── security.py      # Input sanitization & command filtering
    ├── wakeword.py      # Wake word detection
    └── io.py            # AsyncEar (Vosk) & AsyncTTS (Piper) implementations

Quick Start:
    from nola import NOLAManager, NOLAConfig
    
    nola = NOLAManager()
    nola.start()
    
    while True:
        result = nola.get_input(timeout=0.5)
        if result:
            response = brain.process(result.text)
            nola.speak(response)
    
    nola.stop()

Components:
    NOLAManager: Main orchestrator that coordinates all components
    NOLAConfig: Configuration dataclass for NOLAManager
    AsyncEar: Non-blocking microphone listener (Vosk offline STT)
    AsyncTTS: Non-blocking text-to-speech engine (Piper binary)
    InputSanitizer: Security layer for dangerous command blocking
    WakeWordDetector: Voice activation detection
    SanitizedInput: Container for processed voice input
    SecurityLevel: Enum for input classification

Version: 2.0.0 (Vosk + Piper Stack)
"""
from __future__ import annotations

import sys
import importlib.util

# =============================================================================
# Dependency Verification (Vosk + Piper Stack)
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
        print("\n" + "=" * 60)
        print("[X] NOLA DEPENDENCY ERROR")
        print("=" * 60)
        print(f"\nMissing required packages: {', '.join(missing)}")
        print(f"\nFix with:\n  pip install {' '.join(missing)}")
        print("\n" + "=" * 60 + "\n")
        sys.exit(1)
    
    return True


# Run dependency check on import (silent on success, exits on failure)
check_dependencies()
# Note: Success message removed - status shown in main.py's print_system_status()


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

# I/O components (includes wake word detection via AsyncEar)
from .io import (
    RecognitionResult,
    AsyncEar,
    AsyncTTS,
    get_async_ear,
    get_async_tts,
)

# Package metadata
__version__ = "2.0.0"
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
    "AsyncEar",
    "AsyncTTS",
    "get_async_ear",
    "get_async_tts",
    
    # Utilities
    "check_dependencies",
]


def demo():
    """Run a quick demo of NOLA functionality.
    
    This function starts NOLA with wake word detection and 
    echoes back any recognized speech.
    """
    from .manager import demo as _demo
    _demo()
