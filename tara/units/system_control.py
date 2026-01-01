"""TARA System Control Unit - System Monitoring & Hardware Control.

Provides tools for:
- System monitoring (CPU, RAM, disk)
- Hardware control (volume, battery)
- Time/date queries

Dependencies:
    pip install psutil pycaw comtypes
"""
from tara.protocols import tara_tool
import platform
from datetime import datetime
from typing import Any, Optional

# Graceful imports
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    psutil = None  # type: ignore

try:
    import comtypes
    _HAS_COMTYPES = True
except ImportError:
    _HAS_COMTYPES = False
    comtypes = None  # type: ignore

try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL
    from ctypes import cast, POINTER
    _HAS_PYCAW = True
except ImportError:
    _HAS_PYCAW = False
    AudioUtilities = None  # type: ignore
    IAudioEndpointVolume = None  # type: ignore


# =============================================================================
# Audio Helper (Thread-Safe)
# =============================================================================

def _get_audio_interface() -> Optional[Any]:
    """Get Windows audio endpoint interface (thread-safe).
    
    Returns:
        IAudioEndpointVolume interface or None if unavailable.
    """
    if not _HAS_PYCAW or not _HAS_COMTYPES:
        return None
    
    try:
        # CRITICAL: Initialize COM for this thread
        comtypes.CoInitialize()
        
        # Get speakers device
        devices = AudioUtilities.GetSpeakers()
        
        # Activate volume interface (legacy pycaw API)
        interface = devices.Activate(
            IAudioEndpointVolume._iid_,
            CLSCTX_ALL,
            None
        )
        
        # Cast to proper type
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        return volume
        
    except AttributeError:
        # New pycaw API (2024+) uses .EndpointVolume property
        try:
            comtypes.CoInitialize()
            device = AudioUtilities.GetSpeakers()
            return device.EndpointVolume
        except Exception:
            return None
    except Exception:
        return None


# =============================================================================
# System Monitoring Tools
# =============================================================================

@tara_tool(name="system_stats", category="system", description="Get current CPU and RAM usage.")
def system_stats() -> str:
    if not _HAS_PSUTIL:
        return "Error: psutil not installed."
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    return f"CPU Load: {cpu}%\nRAM Usage: {mem.percent}% ({round(mem.used/1024**3, 1)}GB used)"


@tara_tool(name="disk_stats", category="system", description="Get disk usage information.")
def disk_stats() -> str:
    if not _HAS_PSUTIL:
        return "Error: psutil not installed."
    disk = psutil.disk_usage('/')
    free_gb = round(disk.free / (1024**3), 1)
    return f"Disk Usage: {disk.percent}%\nFree Space: {free_gb} GB"


@tara_tool(name="current_time", category="system", description="Get current local time.")
def current_time() -> str:
    return datetime.now().strftime("%I:%M %p, %A %B %d")


@tara_tool(name="system_info", category="system", description="Get OS and Hardware details.")
def system_info() -> str:
    return f"System: {platform.system()} {platform.release()}\nMachine: {platform.machine()}"


# =============================================================================
# Hardware Control Tools
# =============================================================================

@tara_tool(name="get_battery", category="hardware", description="Get battery percentage and status.")
def get_battery() -> str:
    if not _HAS_PSUTIL:
        return "Error: psutil not installed."
    if not hasattr(psutil, "sensors_battery"):
        return "Battery info unavailable on this system."
    batt = psutil.sensors_battery()
    if not batt:
        return "No battery detected (desktop PC?)."
    status = "Charging" if batt.power_plugged else "Discharging"
    return f"Battery: {batt.percent}% ({status})"


@tara_tool(
    name="set_volume",
    category="hardware",
    description="Set system speaker volume (0-100). Also unmutes if muted."
)
def set_volume(level: int) -> str:
    """Set volume and ensure speakers are unmuted."""
    volume = _get_audio_interface()
    if not volume:
        return "Error: pycaw not installed or audio device unavailable."
    
    try:
        # Clamp level
        level = max(0, min(100, int(level)))
        
        # CRUCIAL: Unmute BEFORE setting volume
        volume.SetMute(0, None)
        
        # Set volume level
        volume.SetMasterVolumeLevelScalar(level / 100.0, None)
        
        return f"Volume set to {level}% (Unmuted)"
    except Exception as e:
        return f"Error setting volume: {str(e)}"


@tara_tool(
    name="get_volume",
    category="hardware",
    description="Get current system volume level and mute status."
)
def get_volume() -> str:
    """Get current volume level and mute state."""
    volume = _get_audio_interface()
    if not volume:
        return "Error: pycaw not installed or audio device unavailable."
    
    try:
        current_level = int(volume.GetMasterVolumeLevelScalar() * 100)
        is_muted = volume.GetMute()
        mute_status = "Yes (Muted)" if is_muted else "No (Active)"
        return f"Volume: {current_level}% | Muted: {mute_status}"
    except Exception as e:
        return f"Error getting volume: {str(e)}"


@tara_tool(
    name="mute_volume",
    category="hardware",
    description="Mute (silence) or Unmute (restore audio) the system speakers. Set 'mute' to True to silence, False to unmute."
)
def mute_volume(mute: bool = True) -> str:
    """Mute or unmute system speakers."""
    volume = _get_audio_interface()
    if not volume:
        return "Error: pycaw not installed or audio device unavailable."
    
    try:
        volume.SetMute(1 if mute else 0, None)
        return "System Muted" if mute else "System Unmuted"
    except Exception as e:
        return f"Error muting volume: {str(e)}"
