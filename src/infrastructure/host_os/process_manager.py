"""
MODULE: Host-Native Process Manager
VERSION: 1.0.0
SCOPE: System-wide process discovery, fuzzy matching, and safe termination.
RUNS ON: Host OS (NOT Docker).

Uses `psutil` for cross-platform process control. Implements Smart Lookup
(alias map + fuzzy match) and Safety Blocklist to prevent killing critical
system processes.
"""
from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import psutil

from src.core.logger import setup_logger

logger = setup_logger("TARA.ProcessManager")

# =============================================================================
# Constants
# =============================================================================

_OS = platform.system().lower()  # "windows", "linux", "darwin"

# NOTE: COMMON_ALIASES was removed in v1.1.0.
# App name resolution is now handled by AppIndex (src/infrastructure/host_os/app_index.py)
# which dynamically discovers ALL installed apps via Get-StartApps.
# Process name resolution for kill operations uses psutil name matching directly.

# Processes that must NEVER be killed (OS stability)
BLOCKLIST: Dict[str, Set[str]] = {
    "windows": {
        "system", "registry", "smss.exe", "csrss.exe", "wininit.exe",
        "services.exe", "lsass.exe", "lsaiso.exe", "svchost.exe",
        "winlogon.exe", "fontdrvhost.exe", "dwm.exe", "sihost.exe",
        "taskhostw.exe", "explorer.exe",  # Explorer is special - killing it removes the shell
        "runtimebroker.exe", "searchhost.exe", "startmenuexperiencehost.exe",
        "shellexperiencehost.exe", "textinputhost.exe", "ctfmon.exe",
        "conhost.exe", "dashost.exe", "audiodg.exe",
        "securityhealthservice.exe", "sgrmbroker.exe", "spoolsv.exe",
        "searchindexer.exe", "msdtc.exe", "dllhost.exe",
    },
    "linux": {
        "init", "systemd", "kthreadd", "ksoftirqd", "rcu_sched",
        "migration", "watchdog", "dbus-daemon", "networkmanager",
        "gdm", "sddm", "lightdm", "xorg", "xwayland", "wayland",
        "pulseaudio", "pipewire", "sshd", "cron", "rsyslogd",
    },
    "darwin": {
        "kernel_task", "launchd", "syslogd", "configd", "powerd",
        "diskarbitrationd", "logd", "opendirectoryd", "mds",
        "windowserver", "dock", "finder", "systemuiserver",
        "loginwindow", "coreaudiod", "audiomxd",
    },
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ProcessInfo:
    """Lightweight snapshot of a running process."""
    pid: int
    name: str
    exe: Optional[str] = None
    username: Optional[str] = None
    cmdline: Optional[str] = None
    status: Optional[str] = None

    def display(self) -> str:
        return f"{self.name} (PID: {self.pid})"


# =============================================================================
# HostProcessManager
# =============================================================================

class HostProcessManager:
    """
    Cross-platform process manager that runs on the Host OS.

    Provides:
    - Smart Lookup: alias map + exact match + fuzzy scan.
    - Safe Kill: blocklist enforcement + permission checks.
    """

    def __init__(self) -> None:
        self._os = _OS
        self._blocklist: Set[str] = {
            name.lower() for name in BLOCKLIST.get(self._os, set())
        }
        logger.debug(
            f"HostProcessManager initialized (OS={self._os}, "
            f"blocklist={len(self._blocklist)} entries)"
        )

    # =========================================================================
    # Smart Lookup
    # =========================================================================

    def find_process_by_name(
        self, query: str, fuzzy: bool = True
    ) -> List[ProcessInfo]:
        """
        Find running processes matching a user query.

        Strict 'Index-First' Strategy (Shotgun-Kill-Proof):
        1. AppIndex Oracle: Resolve human name -> EXACT exe filename only.
        2. Fallback Scan: Match process names that START WITH or
           exactly contain the user's original query.

        SAFETY: Never splits display names into words. Never uses
        generic fragments ('settings', 'host', 'service') as targets.

        Args:
            query: Human-friendly name (e.g., "Chrome", "Docker", "File Explorer").
            fuzzy: Enable starts-with / contains matching as fallback.

        Returns:
            List of matching ProcessInfo objects.
        """
        query_lower = query.lower().strip()
        results: List[ProcessInfo] = []
        seen_pids: Set[int] = set()

        # === EXACT TARGETS: only exe filenames from AppIndex ===
        # These are trusted — exact process name match only.
        exact_targets: Set[str] = set()

        # === QUERY TARGETS: the user's raw query for fuzzy matching ===
        # These use starts-with / contains logic.
        query_core = query_lower.replace(".exe", "")

        # =====================================================================
        # Step 1: AppIndex Oracle — resolve to exe filename ONLY
        # =====================================================================
        try:
            from src.infrastructure.host_os.app_index import get_app_index
            app_index = get_app_index()
            entry = app_index.search(query_lower)

            if entry:
                app_id = entry.app_id

                # Win32: extract the exe filename (the ONLY reliable target)
                if entry.app_type == "win32" or (
                    os.path.sep in app_id or "/" in app_id
                ):
                    exe_name = os.path.basename(app_id).lower()
                    exact_targets.add(exe_name)
                    logger.debug(
                        f"AppIndex Oracle: '{query}' -> exe='{exe_name}'"
                    )

                elif entry.app_type == "shell":
                    # Shell apps: extract last segment of dotted ID
                    # e.g., "Microsoft.Windows.Explorer" -> "explorer.exe"
                    id_parts = app_id.split(".")
                    if id_parts:
                        last_part = id_parts[-1].lower()
                        exact_targets.add(f"{last_part}.exe")
                        exact_targets.add(last_part)
                    logger.debug(
                        f"AppIndex Oracle: '{query}' -> shell "
                        f"hint='{id_parts[-1] if id_parts else 'none'}'"
                    )

                # UWP: no exe to extract — rely on query-based fuzzy matching

        except Exception as e:
            logger.debug(f"AppIndex Oracle skipped: {e}")

        # Also add the raw query as an exact target (e.g., "notepad.exe")
        if self._os == "windows" and not query_lower.endswith(".exe"):
            exact_targets.add(f"{query_lower}.exe")
        else:
            exact_targets.add(query_lower)

        # =====================================================================
        # Scan running processes
        # =====================================================================
        logger.debug(
            f"Scanning: exact_targets={exact_targets}, query_core='{query_core}'"
        )

        for proc in psutil.process_iter(
            ['pid', 'name', 'exe', 'username', 'status']
        ):
            try:
                info = proc.info
                proc_name = (info.get('name') or '').lower()
                proc_name_no_ext = proc_name.replace(".exe", "")

                matched = False

                # Priority 1: Exact match against AppIndex-resolved targets
                if proc_name in exact_targets:
                    matched = True

                # Priority 2: Fuzzy — process name STARTS WITH
                # the user's original query (forward direction only)
                elif fuzzy and len(query_core) > 2 and len(proc_name_no_ext) > 2:
                    if proc_name_no_ext.startswith(query_core):
                        matched = True

                if matched and info['pid'] not in seen_pids:
                    seen_pids.add(info['pid'])
                    results.append(ProcessInfo(
                        pid=info['pid'],
                        name=info.get('name', 'unknown'),
                        exe=info.get('exe'),
                        username=info.get('username'),
                        status=info.get('status'),
                    ))

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        logger.info(
            f"🔍 find_process_by_name('{query}'): "
            f"exact_targets={exact_targets}, found {len(results)} match(es)"
        )
        return results

    # =========================================================================
    # Safe Kill
    # =========================================================================

    def is_blocked(self, process_name: str) -> bool:
        """Check if a process is on the safety blocklist."""
        return process_name.lower() in self._blocklist

    def kill_process(self, pid: int, force: bool = False) -> str:
        """
        Terminate a process by PID with safety checks.

        Args:
            pid: Process ID to kill.
            force: If True, use SIGKILL / proc.kill(). Otherwise SIGTERM / proc.terminate().

        Returns:
            Status message string.
        """
        try:
            proc = psutil.Process(pid)
            proc_name = proc.name()
        except psutil.NoSuchProcess:
            return f"⚠️ Process PID {pid} does not exist (already dead?)"
        except psutil.AccessDenied:
            return f"❌ Access denied to PID {pid}"

        # Safety Check
        if self.is_blocked(proc_name):
            logger.warning(f"🛡️ BLOCKED: Attempted to kill protected process '{proc_name}' (PID {pid})")
            return f"🛡️ BLOCKED: '{proc_name}' is a protected system process and cannot be killed."

        # Permission Check: Don't kill other users' processes
        try:
            proc_user = proc.username()
            current_user = os.getlogin()
            # On Windows, username is DOMAIN\user
            if proc_user and current_user:
                proc_user_short = proc_user.split("\\")[-1].lower()
                current_user_short = current_user.lower()
                if proc_user_short != current_user_short:
                    return (
                        f"❌ Permission denied: '{proc_name}' belongs to "
                        f"user '{proc_user}', not '{current_user}'"
                    )
        except Exception:
            pass  # If we can't check, proceed with caution

        # Kill
        try:
            if force:
                proc.kill()
                action = "Force-killed"
            else:
                proc.terminate()
                action = "Terminated"

            # Wait for process to die (up to 5 seconds)
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                if not force:
                    # Escalate to force kill
                    proc.kill()
                    proc.wait(timeout=3)
                    action = "Force-killed (escalated)"
                else:
                    return f"❌ '{proc_name}' (PID {pid}) refused to die even with force kill"

            logger.info(f"💀 {action}: {proc_name} (PID {pid})")
            return f"💀 {action}: {proc_name} (PID {pid})"

        except psutil.NoSuchProcess:
            return f"✅ '{proc_name}' (PID {pid}) already terminated"
        except psutil.AccessDenied:
            return f"❌ Access denied killing '{proc_name}' (PID {pid}). Try running as admin."
        except Exception as e:
            return f"❌ Failed to kill '{proc_name}' (PID {pid}): {e}"

    def kill_by_name(self, query: str, force: bool = False) -> str:
        """
        High-level kill: Find + Kill all matching processes.

        Args:
            query: Human-friendly name.
            force: Use force kill.

        Returns:
            Summary of actions taken.
        """
        matches = self.find_process_by_name(query)

        if not matches:
            return f"⚠️ No running process found matching '{query}'"

        # Filter out blocked processes
        killable = [m for m in matches if not self.is_blocked(m.name)]
        blocked = [m for m in matches if self.is_blocked(m.name)]

        if not killable:
            names = ", ".join(m.name for m in blocked)
            return f"🛡️ All matches are protected system processes: {names}"

        results = []
        for proc_info in killable:
            result = self.kill_process(proc_info.pid, force=force)
            results.append(result)

        if blocked:
            results.append(
                f"🛡️ Skipped {len(blocked)} protected process(es): "
                + ", ".join(m.display() for m in blocked)
            )

        return "\n".join(results)

    def list_processes(self, filter_name: Optional[str] = None, limit: int = 30) -> str:
        """
        List running processes, optionally filtered.

        Args:
            filter_name: Optional name substring filter.
            limit: Max processes to return.

        Returns:
            Formatted process list.
        """
        procs: List[str] = []

        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            try:
                info = proc.info
                name = info.get('name', '')
                pid = info.get('pid', 0)

                if filter_name and filter_name.lower() not in name.lower():
                    continue

                mem = info.get('memory_info')
                mem_mb = f"{mem.rss / (1024*1024):.1f}MB" if mem else "N/A"
                procs.append(f"  {name:<30} PID: {pid:<8} MEM: {mem_mb}")

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

            if len(procs) >= limit:
                break

        if not procs:
            msg = f"No processes found matching '{filter_name}'" if filter_name else "No processes found"
            return msg

        header = f"Running Processes ({len(procs)}):"
        if filter_name:
            header = f"Processes matching '{filter_name}' ({len(procs)}):"

        return header + "\n" + "\n".join(procs)


# =============================================================================
# Singleton
# =============================================================================

_manager: Optional[HostProcessManager] = None


def get_process_manager() -> HostProcessManager:
    """Get or create the global HostProcessManager singleton."""
    global _manager
    if _manager is None:
        _manager = HostProcessManager()
    return _manager


__all__ = [
    "HostProcessManager",
    "ProcessInfo",
    "get_process_manager",
    "BLOCKLIST",
]
