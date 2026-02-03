"""N.I.A. Engine Commands Module.

Contains all CLI command handlers for the NIAAssistant.
Each function takes `engine` as the first parameter to access state.

Design Pattern:
    - Commands are pure functions, not methods
    - They receive the engine instance for state access
    - This allows easy testing and reduces coupling
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .system import NIAAssistant

# Lazy imports for modules needed by specific commands
# Avoids circular imports and speeds up module load


# =============================================================================
# COMMAND HANDLERS (Priority 0 - Strict Dispatch)
# =============================================================================

def cmd_toggle_debug(engine: 'NIAAssistant', text: str) -> bool:
    """Handle debug on/off."""
    if "on" in text:
        from src.core.logger import set_console_level
        import logging
        set_console_level(logging.DEBUG)
        engine.debug = True
        print("🐞 Debug Mode: ON")
    elif "off" in text:
        from src.core.logger import set_console_level
        import logging
        set_console_level(logging.WARNING)
        engine.debug = False
        print("🐞 Debug Mode: OFF")
    else:
        print(f"Debug is {'ON' if engine.debug else 'OFF'}")
    return True


def cmd_reload(engine: 'NIAAssistant', text: str) -> bool:
    """Handle system reload (restart script)."""
    print("[SYSTEM] 🔄 Reloading...")
    engine.stop()
    
    # Mark reload in env to detect on startup
    os.environ["NIA_RELOADED"] = "1"
    
    # Re-execute the current script
    os.execv(sys.executable, ['python'] + sys.argv)
    return True


def cmd_ghost_control(engine: 'NIAAssistant', text: str) -> bool:
    """Handle ghost mode on/off."""
    from src.core.config import settings
    state_file = settings.GHOST_STATE_FILE
    try:
        # Default to False
        new_state = False
        
        if "on" in text:
            new_state = True
        elif "off" in text:
            new_state = False
        else:
            # Just show status
            active, layer = engine._check_ghost_state()
            print(f"👻 Ghost Mode: {'ON' if active else 'OFF'} (Layer {layer})")
            return True

        # Write state
        layer_val = 3 if new_state else 0
        with open(state_file, "w") as f:
            json.dump({"active": new_state, "layer": layer_val}, f)
        
        # Update cache immediately to reflect change
        engine._update_ghost_cache((new_state, layer_val))
        
        if new_state:
            print("👻 [Ghost Mode ENABLED] Audio Suppressed. Stealth Sentry Active.")
        else:
            print("👻 [Ghost Mode DISABLED] Systems Normal.")
            
    except Exception as e:
        engine.logger.error(f"Ghost toggle failed: {e}")
        print(f"⚠️ Failed to toggle Ghost Mode: {e}")
        
    return True


def cmd_mic_control(engine: 'NIAAssistant', text: str) -> bool:
    """Handle mic on/off."""
    from src.core.registry import ServiceRegistry
    
    if "off" in text or "mute" in text:
        nola = ServiceRegistry.get("voice")
        if nola:
            nola.pause_listening()
        try:
            from main import print_mic_off
            print_mic_off()
        except ImportError:
            print("🔇 Microphone Paused")
        engine.speak("Voice system offline.")
        return True
        
    if "on" in text or "unmute" in text:
        if not engine.voice_mode:
            engine.voice_mode = True
            if engine.voice_mode:
                print("[SYSTEM] 🎙️ Voice System ONLINE")
                engine.speak("Voice system online.")
        else:
            nola = ServiceRegistry.get("voice")
            if nola:
                nola.resume_listening()
            print("[SYSTEM] 🎙️ Voice System ONLINE")
            engine.speak("Voice system online.")
        return True
        
    print("Usage: mic [on|off]")
    return True


def cmd_shutdown(engine: 'NIAAssistant', text: str) -> bool:
    """Handle exit/quit."""
    if engine.voice_mode:
        engine.speak("Goodbye!")
    print("👋 Goodbye!")
    engine._running = False
    return True


def cmd_standby(engine: 'NIAAssistant', text: str) -> bool:
    """
    Enter Standby Mode (End current conversation, keep system running).
    Acts as a 'Soft Stop' - the system stops processing the current turn 
    and returns to the main loop to await the next wake word.
    """
    print("\n[SYSTEM] 🌙 Entering Standby Mode...")
    engine.speak("Standing by, Director.")
    return True


# =============================================================================
# HELPER COMMAND HANDLERS
# =============================================================================

def handle_help(engine: 'NIAAssistant', text: str) -> bool:
    """Print help information."""
    print_help(engine)
    return True


def handle_clear(engine: 'NIAAssistant', text: str) -> bool:
    """Clear the terminal screen."""
    subprocess.run(['cls' if os.name == 'nt' else 'clear'], shell=True, check=False)
    return True


def handle_status(engine: 'NIAAssistant', text: str) -> bool:
    """Display system status dashboard."""
    print_status(engine)
    return True


def handle_prefs(engine: 'NIAAssistant', text: str) -> bool:
    """Display user preferences."""
    try:
        prefs = engine.memory.get_all_preferences() if engine.memory else {}
        if prefs:
            print(f"\n📋 [User Preferences]:\n{json.dumps(prefs, indent=2)}\n")
        else:
            print("\n📋 No preferences saved yet.\n")
    except Exception as exc:
        print(f"⚠️  Could not retrieve preferences: {exc}")
    return True


def handle_history(engine: 'NIAAssistant', text: str) -> bool:
    """Handle history commands (view/clear)."""
    # "history clear" or "clear history"
    if "clear" in text:
        try:
            from nia import clear_conversation
            if clear_conversation(engine.thread_id):
                print("✅ History cleared")
            else:
                print("⚠️  Could not clear history")
        except Exception as exc:
            print(f"❌ Error: {exc}")
        return True
        
    # Default: View history
    try:
        from nia import get_conversation_history
        history = get_conversation_history(engine.thread_id)
        if history:
            print(f"📜 History ({len(history)} messages):")
            for msg in history[-10:]:
                role = getattr(msg, 'type', 'unknown')
                content = getattr(msg, 'content', str(msg))[:100]
                content = content.replace('\n', ' ')
                print(f"   [{role}]: {content}")
        else:
            print("📜 History empty")
    except Exception as exc:
        print(f"⚠️  Could not retrieve history: {exc}")
    return True


def handle_reset(engine: 'NIAAssistant', text: str) -> bool:
    """Handle component resets."""
    if "audio" in text:
        print("⚙️  Resetting audio engine...")
        nola = engine._nola if hasattr(engine, '_nola') else None
        if nola:
            nola.stop_speaking()
            nola.resume_listening()
        print("✅ Audio reset complete")
        return True
    return False


# =============================================================================
# REFLEX HANDLERS (Priority 1 - Fuzzy Matching)
# =============================================================================

def handle_reflex(engine: 'NIAAssistant', text: str) -> bool:
    """Handle hardware reflex commands (mic, speaker, sentry)."""
    cmd = text.lower()
    
    if "sentry on" in cmd or "activate sentry" in cmd:
        if not engine.sentry_thread:
            engine._init_sentry()
            print("👁️ ✅ Sentry: ONLINE")
        else:
            print("⚠️  Sentry is already running")
        return True
    
    if "sentry off" in cmd or "deactivate sentry" in cmd:
        if engine.sentry_thread:
            engine.sentry_thread.stop()
            engine.sentry_thread = None
            print("👁️ ❌ Sentry: OFFLINE")
        else:
            print("⚠️  Sentry is not active")
        return True

    # TTS STOP
    if "shut up" in cmd or "stop talking" in cmd or "silence" in cmd:
        nola = engine._nola if hasattr(engine, '_nola') else None
        if nola:
            nola.stop_speaking()
        print("🔇 Silenced")
        return True

    return False


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_help(engine: 'NIAAssistant') -> None:
    """Print help information from external file."""
    from .system import _HELP_TEXT
    print(f"\n{_HELP_TEXT}\n")


def draw_bar(percent: float) -> str:
    """Returns a strict 17-character progress bar string."""
    bar_len = 10
    filled = int((percent / 100) * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    return f"[{bar}] {percent:>3.0f}%"


def print_status(engine: 'NIAAssistant') -> None:
    """Displays the Precision Aligned Dashboard (Strict Grid Layout)."""
    import psutil
    from datetime import datetime
    from src.core.registry import ServiceRegistry
    
    # 1. Gather Data
    cpu_p = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    dsk = psutil.disk_usage('/')
    
    # 2. Prepare Strings
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Status Flags (Strict 3 chars: "ON ", "OFF")
    nola_manager = ServiceRegistry.get("voice")
    iris_manager = ServiceRegistry.get("iris")
    
    s_nia  = "ON " 
    s_nola = "ON " if (nola_manager and getattr(nola_manager, 'is_active', lambda: False)()) else "OFF"
    s_iris = "ON " if iris_manager else "OFF"
    s_tara = "ON "
    
    # API Keys (Strict 7 chars)
    k_nv = "LINKED " if os.environ.get("NVIDIA_API_KEY") else "MISSING"
    k_oa = "LINKED " if os.environ.get("OPENAI_API_KEY") else "MISSING"
    
    # Resource Bars (Strict 17 chars)
    bar_cpu = draw_bar(cpu_p)
    bar_ram = draw_bar(mem.percent)
    bar_dsk = draw_bar(dsk.percent)
    
    # Memory Strings
    mem_used = f"{mem.used / (1024**3):.1f}"
    mem_tot  = f"{mem.total / (1024**3):.1f}"
    mem_str  = f"{mem_used}/{mem_tot} GB"
    dsk_free = f"{dsk.free / (1024**3):.1f} GB Free"
    
    # 3. Render Dashboard (Grid: Left=29, Right=36)
    print("\n┌" + "─"*29 + "┬" + "─"*36 + "┐")
    print(f"│ N.I.A. SYSTEM DASHBOARD     │ {time_str:>34} │")
    print("├─────────────────────────────┼────────────────────────────────────┤")
    print(f"│ 🧠 SUBSYSTEMS               │ 📊 RESOURCES                       │")
    print(f"│ • BRAIN (NIA) : [{s_nia}]       │  CPU: {bar_cpu:<25}    │")
    print(f"│ • VOICE (NOLA): [{s_nola}]       │  RAM: {bar_ram:<25}    │")
    print(f"│ • SENTRY(IRIS): [{s_iris}]       │  DSK: {bar_dsk:<25}    │")
    print(f"│ • TOOLS (TARA): [{s_tara}]       │                                    │")
    print("├─────────────────────────────┼────────────────────────────────────┤")
    print(f"│ 💾 MEMORY                   │ 🔐 SECURITY KEYS                   │")
    print(f"│  RAM : {mem_str:<21}│  NVIDIA API: [{k_nv:<7}]             │")
    print(f"│  DISK: {dsk_free:<21}│  OPENAI API: [{k_oa:<7}]             │")
    print("└─────────────────────────────┴────────────────────────────────────┘\n")


# =============================================================================
# COMMAND REGISTRY BUILDER
# =============================================================================

def build_command_registry(engine: 'NIAAssistant') -> dict:
    """Build the command dispatch registry.
    
    Returns a dictionary mapping command names to handler functions.
    Each handler is a lambda that binds the engine instance.
    """
    return {
        "reload": lambda text: cmd_reload(engine, text),
        "debug": lambda text: cmd_toggle_debug(engine, text),
        "mic": lambda text: cmd_mic_control(engine, text),
        "ghost": lambda text: cmd_ghost_control(engine, text),
        "exit": lambda text: cmd_shutdown(engine, text),
        "quit": lambda text: cmd_shutdown(engine, text),
        # Standby Triggers (Soft Stop)
        "bye": lambda text: cmd_standby(engine, text),
        "goodbye": lambda text: cmd_standby(engine, text),
        "goodnight": lambda text: cmd_standby(engine, text),
        "standby": lambda text: cmd_standby(engine, text),
        "sleep": lambda text: cmd_standby(engine, text),
        "rest": lambda text: cmd_standby(engine, text),
        # Helper commands
        "help": lambda text: handle_help(engine, text),
        "status": lambda text: handle_status(engine, text),
        "clear": lambda text: handle_clear(engine, text),
        "cls": lambda text: handle_clear(engine, text),
        "history": lambda text: handle_history(engine, text),
        "reset": lambda text: handle_reset(engine, text),
        "prefs": lambda text: handle_prefs(engine, text),
    }


def dispatch_reflex(engine: 'NIAAssistant', text: str) -> bool:
    """Dispatch to reflex handlers if no strict command matched."""
    return handle_reflex(engine, text)
