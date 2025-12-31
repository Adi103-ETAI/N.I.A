"""N.O.L.A. - Neural Operator for Language & Audio.

A modular voice I/O system for NIA that separates audio handling
from the reasoning brain.

Package Structure:
    nola/
    ├── __init__.py      # This file - public API exports
    ├── manager.py       # NOLAManager orchestrator
    ├── security.py      # Input sanitization & command filtering
    ├── wakeword.py      # Wake word detection
    └── io.py            # AsyncEar & AsyncTTS implementations

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
    AsyncEar: Non-blocking microphone listener
    AsyncTTS: Non-blocking text-to-speech engine
    InputSanitizer: Security layer for dangerous command blocking
    WakeWordDetector: Voice activation detection
    SanitizedInput: Container for processed voice input
    SecurityLevel: Enum for input classification

Version: 1.0.0
"""
from __future__ import annotations

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

# Wake word detection
from .wakeword import (
    WakeWordDetector,
)

# I/O components
from .io import (
    RecognitionResult,
    AsyncEar,
    AsyncTTS,
    get_async_ear,
    get_async_tts,
)

# Package metadata
__version__ = "1.0.0"
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
    
    # Wake Word
    "WakeWordDetector",
    
    # I/O
    "RecognitionResult",
    "AsyncEar",
    "AsyncTTS",
    "get_async_ear",
    "get_async_tts",
]


def demo():
    """Run a quick demo of NOLA functionality.
    
    This function starts NOLA with wake word detection and 
    echoes back any recognized speech.
    """
    from .manager import demo as _demo
    _demo()


# Convenience check for package integrity
def _check_dependencies() -> dict:
    """Check availability of optional dependencies.
    
    Returns:
        Dict mapping dependency names to availability status.
    """
    deps = {}
    
    try:
        import speech_recognition
        deps["speech_recognition"] = True
    except ImportError:
        deps["speech_recognition"] = False
    
    try:
        import pyttsx3
        deps["pyttsx3"] = True
    except ImportError:
        deps["pyttsx3"] = False
    
    try:
        import pyaudio
        deps["pyaudio"] = True
    except ImportError:
        deps["pyaudio"] = False
    
    return deps


def check_system() -> None:
    """Print system dependency status."""
    deps = _check_dependencies()
    print("N.O.L.A. System Check")
    print("=" * 40)
    for name, available in deps.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"  {name}: {status}")
    print()
    
    if all(deps.values()):
        print("All dependencies installed. NOLA is ready!")
    else:
        missing = [k for k, v in deps.items() if not v]
        print(f"Missing: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
