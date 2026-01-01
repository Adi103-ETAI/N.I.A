"""N.I.A. Health Check Module.

Handles dependency checking and system status reporting.
"""
from __future__ import annotations

import os


def check_dependencies() -> dict:
    """Check all required dependencies.
    
    Returns:
        Dictionary with dependency availability status.
    """
    deps = {}
    
    # NIA brain
    try:
        deps["nia"] = True
    except ImportError as e:
        deps["nia"] = False
        deps["nia_error"] = str(e)
    
    # NOLA voice I/O
    try:
        deps["nola"] = True
    except ImportError as e:
        deps["nola"] = False
        deps["nola_error"] = str(e)
    
    # LangGraph
    try:
        deps["langgraph"] = True
    except ImportError:
        deps["langgraph"] = False
    
    # Vosk STT
    try:
        deps["vosk"] = True
    except ImportError:
        deps["vosk"] = False
    
    # Sounddevice
    try:
        deps["sounddevice"] = True
    except ImportError:
        deps["sounddevice"] = False
    
    # psutil
    try:
        deps["psutil"] = True
    except ImportError:
        deps["psutil"] = False
    
    return deps


def print_system_status() -> None:
    """Print detailed system status with banner."""
    # Import banner
    try:
        from interface.banner import BANNER
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
