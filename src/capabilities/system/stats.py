"""
MODULE: System Operations (The Utilities)
STRICT SCOPE: OS Metadata, Clipboard, System Stats.
CONSTRAINTS: Read-only except for clipboard.

TARA 2.0 Atomic Tool Module.

Exports:
    - get_clipboard_text() -> str
    - set_clipboard_text(text: str) -> str
    - get_system_stats() -> str
    - get_battery_status() -> str
"""
from __future__ import annotations

import platform
from datetime import datetime
from typing import Dict, Optional

from src.core.logger import setup_logger

logger = setup_logger("TARA.Tools.SystemOps")

# =============================================================================
# Optional Dependencies
# =============================================================================

try:
    import pyperclip
    _HAS_CLIPBOARD = True
except ImportError:
    _HAS_CLIPBOARD = False
    pyperclip = None  # type: ignore
    logger.warning("pyperclip not available - clipboard disabled")

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    psutil = None  # type: ignore
    logger.warning("psutil not available - system stats disabled")


# =============================================================================
# Atomic Tool: get_clipboard_text
# =============================================================================

def get_clipboard_text() -> str:
    """
    Get text from the system clipboard.
    
    ONE ACTION: Read clipboard contents.
    
    Returns:
        Clipboard text or error message.
        
    Example:
        >>> get_clipboard_text()
        "📋 Clipboard: Hello, World!"
    """
    if not _HAS_CLIPBOARD:
        return "❌ pyperclip not installed. Run: uv add pyperclip"
    
    try:
        text = pyperclip.paste()
        
        if not text:
            return "📋 Clipboard is empty"
        
        # Truncate for display
        display = text[:200] + "..." if len(text) > 200 else text
        return f"📋 Clipboard: {display}"
        
    except Exception as e:
        return f"❌ Clipboard read failed: {e}"


# =============================================================================
# Atomic Tool: set_clipboard_text
# =============================================================================

def set_clipboard_text(text: str) -> str:
    """
    Copy text to the system clipboard.
    
    ONE ACTION: Write to clipboard.
    
    Args:
        text: Text to copy.
        
    Returns:
        Success or failure message.
    """
    if not _HAS_CLIPBOARD:
        return "❌ pyperclip not installed"
    
    if not text:
        return "❌ No text provided"
    
    try:
        pyperclip.copy(text)
        
        display = text[:50] + "..." if len(text) > 50 else text
        logger.info(f"Copied to clipboard: {len(text)} chars")
        return f"✅ Copied to clipboard: '{display}'"
        
    except Exception as e:
        return f"❌ Clipboard write failed: {e}"


# =============================================================================
# Atomic Tool: get_system_stats
# =============================================================================

def get_system_stats() -> str:
    """
    Get CPU and RAM usage statistics.
    
    ONE ACTION: Read system resource usage.
    
    Returns:
        Formatted system stats string.
        
    Example:
        >>> get_system_stats()
        "💻 System Stats: CPU: 25%, RAM: 60% (8.5 GB / 16 GB)"
    """
    if not _HAS_PSUTIL:
        return "❌ psutil not installed. Run: uv add psutil"
    
    try:
        # CPU usage (1 second sample)
        cpu_percent = psutil.cpu_percent(interval=0.5)
        
        # RAM usage
        memory = psutil.virtual_memory()
        ram_percent = memory.percent
        ram_used_gb = memory.used / (1024 ** 3)
        ram_total_gb = memory.total / (1024 ** 3)
        
        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = disk.percent
        
        stats = [
            f"💻 System Stats:",
            f"   CPU: {cpu_percent:.1f}%",
            f"   RAM: {ram_percent:.1f}% ({ram_used_gb:.1f} GB / {ram_total_gb:.1f} GB)",
            f"   Disk: {disk_percent:.1f}% used",
        ]
        
        return "\n".join(stats)
        
    except Exception as e:
        return f"❌ Failed to get stats: {e}"


# =============================================================================
# Atomic Tool: get_battery_status
# =============================================================================

def get_battery_status() -> str:
    """
    Get battery level and charging status.
    
    ONE ACTION: Read battery information.
    
    Returns:
        Battery status string.
    """
    if not _HAS_PSUTIL:
        return "❌ psutil not installed"
    
    try:
        battery = psutil.sensors_battery()
        
        if battery is None:
            return "🔌 No battery detected (desktop PC?)"
        
        percent = battery.percent
        plugged = "Charging" if battery.power_plugged else "On Battery"
        
        # Calculate time remaining
        if battery.secsleft > 0 and not battery.power_plugged:
            hours = battery.secsleft // 3600
            minutes = (battery.secsleft % 3600) // 60
            time_left = f", {hours}h {minutes}m remaining"
        else:
            time_left = ""
        
        icon = "🔋" if percent > 20 else "🪫"
        return f"{icon} Battery: {percent}% ({plugged}{time_left})"
        
    except Exception as e:
        return f"❌ Failed to get battery: {e}"


# =============================================================================
# Atomic Tool: get_system_info
# =============================================================================

def get_system_info() -> str:
    """
    Get basic system information.
    
    ONE ACTION: Read OS and hardware info.
    
    Returns:
        Formatted system info string.
    """
    try:
        info = [
            f"🖥️ System Info:",
            f"   OS: {platform.system()} {platform.release()}",
            f"   Version: {platform.version()}",
            f"   Machine: {platform.machine()}",
            f"   Processor: {platform.processor()}",
        ]
        
        # Add hostname
        try:
            import socket
            info.append(f"   Hostname: {socket.gethostname()}")
        except Exception:
            pass
        
        return "\n".join(info)
        
    except Exception as e:
        return f"❌ Failed to get info: {e}"


# =============================================================================
# Atomic Tool: get_current_time
# =============================================================================

def get_current_time() -> str:
    """
    Get current date and time.
    
    ONE ACTION: Read system clock.
    
    Returns:
        Formatted datetime string.
    """
    now = datetime.now()
    return f"🕐 Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# =============================================================================
# Atomic Tool: mute_volume / unmute_volume
# =============================================================================

def mute_volume(mute: bool = True) -> str:
    """
    Mute or unmute system audio.
    
    ONE ACTION: Toggle system volume mute state.
    
    Args:
        mute: True to mute, False to unmute.
        
    Returns:
        Success or failure message.
    """
    try:
        # Windows-specific: use pycaw or ctypes
        import ctypes
        from ctypes import wintypes, POINTER, cast
        
        # Try pycaw first (cleaner API)
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            from comtypes import CLSCTX_ALL
            
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            
            volume.SetMute(1 if mute else 0, None)
            return "🔇 System Muted" if mute else "🔊 System Unmuted"
            
        except ImportError:
            # Fallback: Use keyboard simulation
            import subprocess
            
            # Use NirCmd if available, otherwise powershell
            if mute:
                # PowerShell audio mute
                subprocess.run([
                    'powershell', '-Command',
                    '(New-Object -ComObject WScript.Shell).SendKeys([char]173)'
                ], capture_output=True, timeout=5)
                return "🔇 System Muted"
            else:
                subprocess.run([
                    'powershell', '-Command',
                    '(New-Object -ComObject WScript.Shell).SendKeys([char]173)'  # Toggle
                ], capture_output=True, timeout=5)
                return "🔊 System Unmuted"
                
    except Exception as e:
        return f"❌ Audio control failed: {e}"


__all__ = [
    "get_clipboard_text",
    "set_clipboard_text",
    "get_system_stats",
    "get_battery_status",
    "get_system_info",
    "get_current_time",
    "mute_volume",
]

