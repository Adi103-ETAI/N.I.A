"""
Warden Service - Operation Iron Cage.

The "Smart Warden" responsible for handling high-risk tool escalations.
It subscribes to 'security:escalation' events and applies Smart Trust rules
before executing potentially dangerous operations.

Attributes:
    SAFE_ZONES: Directories where file operations are auto-approved (dynamic).
    SAFE_EXTENSIONS: File types that are auto-approved for deletion (temp/logs).

v3.1 - Operation Universal:
    SAFE_ZONES now uses OSContext for cross-platform compatibility.
"""
import asyncio
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Any, List

from core.logger import setup_logger
from core.event_bus import get_event_bus
from core.context import get_os_context

try:
    from send2trash import send2trash
    _HAS_TRASH = True
except ImportError:
    _HAS_TRASH = False

logger = setup_logger("WARDEN")

# =============================================================================
# Configuration
# =============================================================================

def _get_safe_zones() -> List[Path]:
    """Get dynamic SAFE_ZONES from OSContext (cross-platform)."""
    return get_os_context().get_safe_zones()

# Lazy-loaded at first use
SAFE_ZONES: List[Path] = []

SAFE_EXTENSIONS = {".tmp", ".log", ".bak", ".cache"}


# =============================================================================
# Warden Logic
# =============================================================================

class WardenService:
    """Security Interceptor using Smart Trust logic."""
    
    def __init__(self):
        self.bus = None
        
    def start(self):
        """Start the Warden listener."""
        global SAFE_ZONES
        try:
            # Lazy-load SAFE_ZONES from OSContext (cross-platform)
            SAFE_ZONES = _get_safe_zones()
            logger.debug(f"SAFE_ZONES loaded: {[str(z) for z in SAFE_ZONES]}")
            
            self.bus = get_event_bus()
            self.bus.subscribe("security:escalation", self._handle_escalation)
            logger.info("🛡️ Warden Service Active and Watching 'security:escalation'")
        except Exception as e:
            logger.error(f"❌ Failed to start Warden: {e}")

    async def _handle_escalation(self, payload: Dict[str, Any]) -> None:
        """Handle security escalation request."""
        tool_name = payload.get("tool")
        args = payload.get("args", {})
        
        logger.info(f"👮 Warden received request for: {tool_name}")
        
        try:
            if tool_name == "launch_app":
                await self._rule_app_launcher(args)
            elif tool_name == "delete_file":
                await self._rule_file_deletion(args)
            else:
                logger.warning(f"⛔ Unknown high-risk tool '{tool_name}' - BLOCKED by default.")
                
        except Exception as e:
            logger.error(f"❌ Warden Handler Failed: {e}")

    # --- Rule A: App Launcher (Shell=False) ---
    async def _rule_app_launcher(self, args: Dict[str, Any]) -> None:
        """Sanitize and run app launch requests without shell injection."""
        app_name = args.get("app_name", "")
        if not app_name:
            logger.warning("⛔ Blocked launch_app: No app_name provided.")
            return

        # Sanitize: Ensure we aren't passing command chains
        # Simple strategy: Treat the whole string as the executable/command
        # For deeper safety, we'd use shlex to parse arguments if they were separate
        
        logger.info(f"🕵️ Warden Inspecting Launch: '{app_name}'")
        
        # Safe Execution: Shell=False
        # This prevents "notepad && del *.*" type injections
        try:
            # Check if it's a known system executable or path
            # For now, we allow the run, but enforced through subprocess directly
            
            def _run_safe():
                # We use shell=True ONLY if strictly necessary, but user requested shell=False enforcement.
                # However, launching 'notepad' without path on Windows often needs shell=True or finding executable.
                # User instructions: "run with subprocess.Popen(..., shell=False). Never use shell=True."
                # We will attempt to run it directly. If it fails (not in PATH), we log failure.
                
                # Split args if present (naive split for this level of detail)
                # If app_name contains space and args, shlex might help
                cmd_parts = shlex.split(app_name, posix=False)
                
                subprocess.Popen(
                    cmd_parts,
                    shell=False,  # <--- THE IRON BAR
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
            await asyncio.to_thread(_run_safe)
            logger.info(f"✅ Warden Approved & Launched: {app_name}")
            
        except FileNotFoundError:
            logger.warning(f"⚠️ Launch failed: Executable not found '{app_name}'")
        except Exception as e:
            logger.error(f"⛔ Warden Execution Error: {e}")

    # --- Rule B: File Deletion (Smart Recycle) ---
    async def _rule_file_deletion(self, args: Dict[str, Any]) -> None:
        """Smart deletion logic: Recycle Bin only, and Safe Zones only."""
        path_str = args.get("path", "")
        if not path_str:
            return
            
        path = Path(path_str).resolve()
        
        logger.info(f"🕵️ Warden Inspecting Deletion: '{path}'")
        
        # Check 1: Is it in a Safe Zone? (Cross-platform Path comparison)
        def _is_in_zone(target: Path, zone: Path) -> bool:
            try:
                target.relative_to(zone)
                return True
            except ValueError:
                return False
        
        is_safe_zone = any(_is_in_zone(path, zone) for zone in SAFE_ZONES)
        
        # Check 2: Safe Extension?
        is_safe_ext = path.suffix.lower() in SAFE_EXTENSIONS
        
        if is_safe_zone or is_safe_ext:
            # APPROVED -> Smart Delete (Recycle Bin)
            if _HAS_TRASH:
                try:
                    await asyncio.to_thread(send2trash, str(path))
                    logger.info(f"✅ Warden Approved & Recycled: {path.name}")
                except Exception as e:
                    logger.error(f"❌ Recycle failed: {e}")
            else:
                logger.warning("⚠️ send2trash missing. Warden blocks permanent delete.")
        else:
            # DENIED
            logger.warning(f"⛔ Warden BLOCKED deletion: {path} (Outside Safe Zone)")


# Global Singleton Helper
_warden = None

def start_warden_service():
    """Initialize and start the global Warden service."""
    global _warden
    if _warden is None:
        _warden = WardenService()
        _warden.start()
