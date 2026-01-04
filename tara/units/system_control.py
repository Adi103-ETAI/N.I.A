"""TARA System Control Unit - Admin Layer for N.I.A.

Handles Hardware & Global OS State via PowerShell and Windows APIs.

Features:
    - Power & Maintenance (lock, sleep, shutdown, restart, recycle bin)
    - Audio Control (volume, mute via Pycaw)
    - System Stats (CPU, RAM, Disk, Battery via Psutil)

Dependencies:
    pip install psutil pycaw comtypes
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from typing import Any, Optional

from tara.protocols import tara_tool

# =============================================================================
# Optional Dependencies (Graceful Imports)
# =============================================================================

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    psutil = None  # type: ignore

try:
    import ctypes
    _HAS_CTYPES = True
except ImportError:
    _HAS_CTYPES = False
    ctypes = None  # type: ignore

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
# Helper: PowerShell Command Executor
# =============================================================================

def _run_powershell(cmd: str) -> str:
    """Execute a PowerShell command and return output.
    
    Args:
        cmd: PowerShell command string to execute.
        
    Returns:
        Command output as string.
        
    Raises:
        RuntimeError: If command execution fails.
    """
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0 and result.stderr:
            raise RuntimeError(f"PowerShell error: {result.stderr.strip()}")
        
        return result.stdout.strip()
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("PowerShell command timed out")
    except FileNotFoundError:
        raise RuntimeError("PowerShell not found on system")
    except Exception as e:
        raise RuntimeError(f"PowerShell execution failed: {e}")


# =============================================================================
# Helper: Audio Interface (Thread-Safe)
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
# TOOL GROUP 1: Power & Maintenance (The "Muscle")
# =============================================================================

@tara_tool(
    name="system_power",
    category="power",
    description="Control system power state. Actions: 'lock' (lock screen), 'sleep' (suspend), 'shutdown' (60s warning), 'restart' (60s warning), 'abort' (cancel shutdown/restart)."
)
def system_power(action: str) -> str:
    """Control system power state.
    
    Args:
        action: One of 'lock', 'sleep', 'shutdown', 'restart', 'abort'.
    """
    action = action.lower().strip()
    
    try:
        if action == "lock":
            # Lock workstation using rundll32
            result = subprocess.run(
                ["rundll32.exe", "user32.dll,LockWorkStation"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                return "🔒 Workstation locked."
            return "Error: Failed to lock workstation."
        
        elif action == "sleep":
            # Suspend system (0 = sleep, 1 = force, 0 = no hibernate)
            result = subprocess.run(
                ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
                capture_output=True,
                timeout=10,
            )
            return "😴 System entering sleep mode..."
        
        elif action == "shutdown":
            # Shutdown with 60 second warning
            _run_powershell("Stop-Computer -Force -Confirm:$false")
            return "⚠️ System will shut down in 60 seconds. Use 'abort' to cancel."
        
        elif action == "restart":
            # Restart with 60 second warning
            _run_powershell("shutdown /r /t 60 /c 'NIA: System restart in 60 seconds'")
            return "⚠️ System will restart in 60 seconds. Use 'abort' to cancel."
        
        elif action == "abort":
            # Cancel pending shutdown/restart
            _run_powershell("shutdown /a")
            return "✅ Shutdown/restart cancelled."
        
        else:
            return f"Error: Unknown action '{action}'. Valid: lock, sleep, shutdown, restart, abort."
            
    except RuntimeError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error executing power action: {e}"


@tara_tool(
    name="empty_recycle_bin",
    category="maintenance",
    description="Empty the Windows Recycle Bin permanently."
)
def empty_recycle_bin() -> str:
    """Empty the Windows Recycle Bin."""
    try:
        _run_powershell("Clear-RecycleBin -Force -ErrorAction SilentlyContinue")
        return "🗑️ Recycle Bin emptied."
    except RuntimeError as e:
        return f"Error emptying Recycle Bin: {e}"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# TOOL GROUP 2: Audio Control (The "Voice")
# =============================================================================

@tara_tool(
    name="set_volume",
    category="audio",
    description="Set system speaker volume (0-100). Also unmutes if muted."
)
def set_volume(level: int) -> str:
    """Set system master volume.
    
    Args:
        level: Volume level from 0 to 100.
    """
    volume = _get_audio_interface()
    if not volume:
        return "Error: Audio control unavailable (pycaw not installed or no audio device)."
    
    try:
        # Clamp level to valid range
        level = max(0, min(100, int(level)))
        
        # CRUCIAL: Unmute BEFORE setting volume
        volume.SetMute(0, None)
        
        # Set volume level (scalar 0.0 - 1.0)
        volume.SetMasterVolumeLevelScalar(level / 100.0, None)
        
        return f"🔊 Volume set to {level}% (Unmuted)"
        
    except Exception as e:
        return f"Error setting volume: {e}"


@tara_tool(
    name="mute_volume",
    category="audio",
    description="Mute or unmute system speakers. Set 'mute' to True to silence, False to unmute."
)
def mute_volume(mute: bool = True) -> str:
    """Mute or unmute system speakers.
    
    Args:
        mute: True to mute, False to unmute.
    """
    volume = _get_audio_interface()
    if not volume:
        return "Error: Audio control unavailable (pycaw not installed or no audio device)."
    
    try:
        volume.SetMute(1 if mute else 0, None)
        return "🔇 System Muted" if mute else "🔊 System Unmuted"
    except Exception as e:
        return f"Error toggling mute: {e}"


@tara_tool(
    name="get_volume",
    category="audio",
    description="Get current system volume level and mute status."
)
def get_volume() -> str:
    """Get current volume level and mute state."""
    volume = _get_audio_interface()
    if not volume:
        return "Error: Audio control unavailable (pycaw not installed or no audio device)."
    
    try:
        current_level = int(volume.GetMasterVolumeLevelScalar() * 100)
        is_muted = volume.GetMute()
        
        mute_status = "🔇 Muted" if is_muted else "🔊 Active"
        return f"Volume: {current_level}% | Status: {mute_status}"
        
    except Exception as e:
        return f"Error getting volume: {e}"


# =============================================================================
# TOOL GROUP 3: System Stats (The "Vitals")
# =============================================================================

@tara_tool(
    name="system_stats",
    category="stats",
    description="Get current CPU usage, RAM usage, and Disk usage."
)
def system_stats() -> str:
    """Get system resource statistics."""
    if not _HAS_PSUTIL:
        return "Error: psutil not installed. Run: pip install psutil"
    
    try:
        # CPU (short interval for responsiveness)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024 ** 3)
        mem_total_gb = mem.total / (1024 ** 3)
        
        # Disk (primary drive)
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024 ** 3)
        
        return (
            f"📊 System Stats\n"
            f"   CPU:  {cpu_percent:.1f}%\n"
            f"   RAM:  {mem.percent:.1f}% ({mem_used_gb:.1f}/{mem_total_gb:.1f} GB)\n"
            f"   Disk: {disk.percent:.1f}% ({disk_free_gb:.1f} GB free)"
        )
        
    except Exception as e:
        return f"Error getting system stats: {e}"


@tara_tool(
    name="battery_status",
    category="stats",
    description="Get battery percentage, charging status, and estimated time remaining (laptops only)."
)
def battery_status() -> str:
    """Get battery status (for laptops)."""
    if not _HAS_PSUTIL:
        return "Error: psutil not installed. Run: pip install psutil"
    
    try:
        battery = psutil.sensors_battery()
        
        if battery is None:
            return "🔌 No battery detected (desktop PC or virtual machine)."
        
        percent = battery.percent
        plugged = battery.power_plugged
        
        # Charging status
        if plugged:
            status = "⚡ Charging" if percent < 100 else "🔌 Fully Charged"
        else:
            status = "🔋 Discharging"
        
        # Time remaining (only meaningful when discharging)
        time_str = ""
        if battery.secsleft > 0 and not plugged:
            hours = battery.secsleft // 3600
            minutes = (battery.secsleft % 3600) // 60
            time_str = f" ({hours}h {minutes}m remaining)"
        elif battery.secsleft == psutil.POWER_TIME_UNLIMITED:
            time_str = " (calculating...)"
        
        return f"🔋 Battery: {percent:.0f}% | {status}{time_str}"
        
    except Exception as e:
        return f"Error getting battery status: {e}"
