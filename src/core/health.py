"""N.I.A. Health Check Module.

Provides dependency verification and system status reporting for the N.I.A. system.

Components:
    - check_dependencies(): Returns dict of all module availability
    - print_system_status(): Prints formatted status with banner

Usage:
    from src.core.health import check_dependencies, print_system_status
    
    # Quick check
    deps = check_dependencies()
    if deps["nia"] and deps["nola"]:
        print("Core systems ready!")
    
    # Full status display
    print_system_status()  # Shows banner + detailed status

Version: 4.0.0
"""
from __future__ import annotations

import os


def check_dependencies() -> dict:
    """Check all required dependencies.
    
    Returns:
        Dictionary with dependency availability status.
    """
    deps = {}
    
    # NIA brain (Core LLM Agent)
    try:
        import src.agents.nia as nia
        deps["nia"] = True
    except ImportError as e:
        deps["nia"] = False
        deps["nia_error"] = str(e)
    
    # NOLA voice I/O (TTS + STT)
    try:
        import src.agents.nola as nola
        deps["nola"] = True
    except ImportError as e:
        deps["nola"] = False
        deps["nola_error"] = str(e)
    
    # IRIS vision agent
    try:
        import src.agents.iris as iris
        deps["iris"] = True
    except ImportError as e:
        deps["iris"] = False
        deps["iris_error"] = str(e)
    
    # TARA tool agent
    try:
        import src.agents.tara as tara
        deps["tara"] = True
    except ImportError as e:
        deps["tara"] = False
        deps["tara_error"] = str(e)
    
    # LangGraph (workflow engine)
    try:
        import langgraph
        deps["langgraph"] = True
    except ImportError:
        deps["langgraph"] = False
    
    # Vosk STT
    try:
        import vosk
        deps["vosk"] = True
    except ImportError:
        deps["vosk"] = False
    
    # Sounddevice (audio I/O)
    try:
        import sounddevice
        deps["sounddevice"] = True
    except ImportError:
        deps["sounddevice"] = False
    
    # psutil (system monitoring)
    try:
        import psutil
        deps["psutil"] = True
    except ImportError:
        deps["psutil"] = False
    
    return deps


def print_system_status() -> None:
    """Print detailed system status with banner."""
    # Import banner
    try:
        from src.interface.cli.banner import BANNER
    except ImportError:
        BANNER = "N.I.A. - Neural Intelligence Assistant"
    
    deps = check_dependencies()
    
    print(BANNER)
    print("⚙️  System Status")
    print("=" * 50)
    
    # Core components
    print("\n📦 Core Components:")
    for name in ["nia", "nola", "langgraph"]:
        status = "✅" if deps.get(name) else "❌"
        print(f"   {status} {name}")
        if not deps.get(name) and deps.get(f"{name}_error"):
            print(f"      ⚠️  Error: {deps.get(f'{name}_error')}")
    
    # Voice stack
    print("\n🎤 Voice Stack (Vosk + Piper):")
    for name in ["vosk", "sounddevice"]:
        status = "✅" if deps.get(name) else "❌"
        print(f"   {status} {name}")
    
    # API Keys
    print("\n⚙️  Configuration:")
    env_vars = [
        ("NVIDIA_API_KEY", True),
        ("OPENAI_API_KEY", False),
        ("HUGGINGFACE_API_KEY", False),
        ("OLLAMA_HOST", False),
    ]
    
    for key, required in env_vars:
        value = os.environ.get(key)
        is_set = bool(value and value.strip())
        
        if is_set:
            if key == "OLLAMA_HOST":
                print(f"   ✅ {key}: {value}")
            else:
                print(f"   ✅ {key}: Set")
        else:
            if required:
                print(f"   ❌ {key}: Not set (Required!)")
            else:
                print(f"   ⚠️  {key}: Not set (Optional)")
    
    # Summary
    print("\n" + "=" * 50)
    
    has_primary_key = bool(os.environ.get("NVIDIA_API_KEY"))
    all_core = deps.get("nia") and deps.get("nola") and has_primary_key
    voice_ready = deps.get("vosk") and deps.get("sounddevice")
    
    if all_core and voice_ready:
        print("🚀 All systems nominal! Voice mode available.")
    elif all_core:
        print("⚠️  Core ready, but voice dependencies missing.")
        print("    Install: pip install vosk sounddevice")
    else:
        print("❌ System not ready. Check missing components above.")
        if not has_primary_key:
            print("    Set NVIDIA_API_KEY in .env file")
    
    print()
