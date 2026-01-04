"""Ghost Protocol - Panic Button Module for N.I.A.

Implements a 3-layer security system for emergency concealment:
    Layer 1 (Concealment): Mute audio, disable TTS, minimize windows.
    Layer 2 (Sanitization): Layer 1 + Kill noisy apps (browsers, discord, etc).
    Layer 3 (Lockdown): Layer 2 + Lock Windows workstation + Sentry mode.

Usage:
    ghost_mode("activate", layer=1)  # Silent mode
    ghost_mode("activate", layer=2)  # Kill distractions
    ghost_mode("activate", layer=3)  # Full lockdown
    ghost_mode("deactivate")         # Return to normal
"""

import os
import json
import time
import ctypes
import subprocess
import pyautogui
import psutil

from tara.protocols import tara_tool


# =============================================================================
# Constants
# =============================================================================

GHOST_STATE_FILE = "data/ghost_state.json"

# Applications to kill in Layer 2 (noisy/distraction apps)
DISTRACTION_APPS = [
    'chrome.exe',
    'brave.exe',
    'firefox.exe',
    'msedge.exe',
    'discord.exe',
    'spotify.exe',
    'steam.exe',
    'vlc.exe',
]

# System-critical processes to NEVER kill
PROTECTED_PROCESSES = [
    'python.exe',
    'pythonw.exe',
    'python3.exe',
    'cmd.exe',
    'powershell.exe',
    'explorer.exe',
    'csrss.exe',
    'smss.exe',
    'services.exe',
    'svchost.exe',
    'lsass.exe',
    'winlogon.exe',
    'system',
]


# =============================================================================
# Helper Functions
# =============================================================================

def _set_ghost_state(active: bool, layer: int = 0) -> None:
    """Write ghost protocol state to JSON file.
    
    This file signals the main Engine to disable TTS while keeping
    the microphone loop active for continued listening.
    
    Args:
        active: Whether ghost mode is active.
        layer: Current security layer (0 = inactive, 1-3 = active layers).
    """
    state = {
        "active": active,
        "layer": layer,
        "timestamp": time.time(),
    }
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(GHOST_STATE_FILE), exist_ok=True)
    
    with open(GHOST_STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def _mute_system() -> None:
    """Mute system audio using pyautogui.
    
    Uses the volume mute key which toggles system mute state.
    This is more reliable than COM-based solutions and has no dependencies.
    """
    pyautogui.press('volumemute')


def _unmute_system() -> None:
    """Unmute system audio.
    
    Toggles mute again - assumes system was muted by ghost mode.
    In practice, this is the same as _mute_system() since it's a toggle.
    """
    pyautogui.press('volumemute')


def _minimize_all_windows() -> None:
    """Minimize all windows using Win+D hotkey.
    
    Shows desktop, effectively minimizing all visible windows.
    """
    pyautogui.hotkey('win', 'd')


def _kill_distractions() -> list:
    """Kill noisy/distraction applications (Layer 2).
    
    Terminates browsers, media players, and communication apps
    that could leak audio or notifications.
    
    Returns:
        List of terminated process names.
    """
    killed = []
    
    for proc in psutil.process_iter(['name', 'pid']):
        try:
            proc_name = proc.info['name']
            if proc_name is None:
                continue
                
            proc_name_lower = proc_name.lower()
            
            # Skip protected system processes
            if proc_name_lower in PROTECTED_PROCESSES:
                continue
            
            # Check if it's a distraction app
            if proc_name_lower in DISTRACTION_APPS:
                proc.terminate()
                killed.append(proc_name)
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Process already gone or access denied - continue silently
            continue
    
    return killed


def _lock_workstation() -> bool:
    """Lock the Windows workstation (Layer 3).
    
    Uses Windows API to lock the computer immediately.
    
    Returns:
        True if lock was successful, False otherwise.
    """
    try:
        ctypes.windll.user32.LockWorkStation()
        return True
    except Exception:
        return False


# =============================================================================
# Main Tool
# =============================================================================

@tara_tool(
    name="ghost_mode",
    category="security",
    description=(
        "Panic button for emergency concealment. "
        "Layer 1: Mute + minimize windows. "
        "Layer 2: + Kill browsers/Discord/Spotify. "
        "Layer 3: + Lock workstation. "
        "Use 'deactivate' to restore normal operation."
    )
)
def ghost_mode(action: str = "activate", layer: int = 1, **kwargs) -> str:
    """Activate or deactivate Ghost Protocol security layers.
    
    Args:
        action: "activate" to enable, "deactivate" to disable ghost mode.
        layer: Security layer (1-3). Only used when action is "activate".
               Layer 1: Mute + TTS disabled + windows minimized.
               Layer 2: Layer 1 + kill distraction apps.
               Layer 3: Layer 2 + lock workstation.
        **kwargs: Additional arguments (reserved for future use).
    
    Returns:
        Status message indicating what actions were taken.
    """
    
    # -------------------------------------------------------------------------
    # DEACTIVATE: Restore normal operation
    # -------------------------------------------------------------------------
    if action.lower() == "deactivate":
        _set_ghost_state(active=False, layer=0)
        _unmute_system()
        return "✅ Ghost Protocol Deactivated. Systems Normal."
    
    # -------------------------------------------------------------------------
    # ACTIVATE: Enable security layers
    # -------------------------------------------------------------------------
    
    # Clamp layer to valid range
    layer = max(1, min(3, layer))
    
    msg_parts = []
    
    # Layer 1: Concealment
    _set_ghost_state(active=True, layer=layer)
    _mute_system()
    _minimize_all_windows()
    msg_parts.append("👻 Layer 1: Silent & Concealed.")
    
    # Layer 2: Sanitization
    if layer >= 2:
        killed = _kill_distractions()
        kill_count = len(killed)
        msg_parts.append(f"💀 Layer 2: Apps Purged ({kill_count} terminated).")
    
    # Layer 3: Lockdown
    if layer >= 3:
        if _lock_workstation():
            msg_parts.append("🔒 Layer 3: System Locked.")
        else:
            msg_parts.append("⚠️ Layer 3: Lock failed.")
    
    return " ".join(msg_parts)


# =============================================================================
# Utility Functions (for external use)
# =============================================================================

def get_ghost_state() -> dict:
    """Read current ghost protocol state from file.
    
    Returns:
        State dict with 'active', 'layer', and 'timestamp' keys.
        Returns default inactive state if file doesn't exist.
    """
    if not os.path.exists(GHOST_STATE_FILE):
        return {"active": False, "layer": 0, "timestamp": 0}
    
    try:
        with open(GHOST_STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"active": False, "layer": 0, "timestamp": 0}


def is_ghost_active() -> bool:
    """Quick check if ghost mode is currently active.
    
    Returns:
        True if ghost mode is active, False otherwise.
    """
    state = get_ghost_state()
    return state.get("active", False)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ghost_mode",
    "get_ghost_state",
    "is_ghost_active",
    "GHOST_STATE_FILE",
]
