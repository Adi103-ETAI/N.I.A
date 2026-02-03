"""
OS Context - Cross-Platform Abstraction Layer.

Provides a singleton OSContext class that abstracts OS-specific logic,
enabling N.I.A. to run on Windows, Linux, and macOS.

Usage:
    from src.core.context import get_os_context
    
    ctx = get_os_context()
    print(ctx.os_name)           # "windows", "linux", or "darwin"
    print(ctx.downloads_dir)     # Platform-appropriate Downloads path
    ctx.open_file("document.pdf") # Uses correct OS command
"""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import List, Optional

from src.core.logger import setup_logger

logger = setup_logger("OS_CONTEXT")


# =============================================================================
# OSContext Singleton
# =============================================================================

class OSContext:
    """
    Singleton class for OS-specific context and utilities.
    
    Attributes:
        os_name: Normalized OS name ("windows", "linux", or "darwin").
        is_windows: True if running on Windows.
        is_linux: True if running on Linux.
        is_macos: True if running on macOS.
        home_dir: User's home directory as Path.
        desktop_dir: User's Desktop directory as Path.
        downloads_dir: User's Downloads directory as Path.
        temp_dir: System temp directory as Path.
    """
    
    _instance: Optional["OSContext"] = None
    
    def __new__(cls) -> "OSContext":
        """Ensure only one instance exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize OS context (called once on first instantiation)."""
        # Detect OS
        system = platform.system().lower()
        
        if system == "windows":
            self.os_name = "windows"
        elif system == "linux":
            self.os_name = "linux"
        elif system == "darwin":
            self.os_name = "darwin"
        else:
            self.os_name = system  # Fallback for unknown OS
            logger.warning(f"Unknown OS detected: {system}")
        
        # Boolean shortcuts
        self.is_windows: bool = self.os_name == "windows"
        self.is_linux: bool = self.os_name == "linux"
        self.is_macos: bool = self.os_name == "darwin"
        
        # Resolve standard directories
        self.home_dir: Path = Path.home()
        self.desktop_dir: Path = self._resolve_desktop()
        self.downloads_dir: Path = self._resolve_downloads()
        self.temp_dir: Path = self._resolve_temp()
        
        logger.debug(f"OSContext initialized: {self.os_name}")
        logger.debug(f"  Home: {self.home_dir}")
        logger.debug(f"  Desktop: {self.desktop_dir}")
        logger.debug(f"  Downloads: {self.downloads_dir}")
    
    def _resolve_desktop(self) -> Path:
        """Resolve the user's Desktop directory."""
        if self.is_windows:
            # Windows: Check USERPROFILE environment or fallback
            desktop = self.home_dir / "Desktop"
        else:
            # Linux/macOS: XDG Desktop or ~/Desktop
            xdg_desktop = os.environ.get("XDG_DESKTOP_DIR")
            if xdg_desktop:
                desktop = Path(xdg_desktop)
            else:
                desktop = self.home_dir / "Desktop"
        
        # Ensure it exists (create if not)
        desktop.mkdir(parents=True, exist_ok=True)
        return desktop
    
    def _resolve_downloads(self) -> Path:
        """Resolve the user's Downloads directory."""
        if self.is_windows:
            downloads = self.home_dir / "Downloads"
        else:
            # Linux/macOS: XDG Download or ~/Downloads
            xdg_download = os.environ.get("XDG_DOWNLOAD_DIR")
            if xdg_download:
                downloads = Path(xdg_download)
            else:
                downloads = self.home_dir / "Downloads"
        
        # Ensure it exists
        downloads.mkdir(parents=True, exist_ok=True)
        return downloads
    
    def _resolve_temp(self) -> Path:
        """Resolve the system temp directory."""
        import tempfile
        return Path(tempfile.gettempdir())
    
    # =========================================================================
    # Cross-Platform Methods
    # =========================================================================
    
    def get_shell_command(self, cmd: List[str]) -> List[str]:
        """
        Prepare a command list for subprocess execution.
        
        On most platforms, the command list can be passed directly to subprocess.
        This method exists for future extensibility (e.g., shell wrappers).
        
        Args:
            cmd: Command as a list of strings (e.g., ["ls", "-la"]).
            
        Returns:
            Command list ready for subprocess.Popen/run.
        """
        # Currently, direct list works on all platforms with shell=False
        # Future: Could wrap with shell-specific logic if needed
        return cmd
    
    def open_file(self, path: str | Path) -> bool:
        """
        Open a file or URL with the system's default application.
        
        Uses:
            - Windows: start
            - Linux: xdg-open
            - macOS: open
        
        Args:
            path: File path or URL to open.
            
        Returns:
            True if command was launched successfully, False otherwise.
        """
        path_str = str(path)
        
        try:
            if self.is_windows:
                # Windows: Use 'start' command via cmd
                # Note: start requires shell=True or os.startfile
                os.startfile(path_str)  # type: ignore[attr-defined]
            elif self.is_macos:
                subprocess.Popen(
                    ["open", path_str],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:  # Linux and others
                subprocess.Popen(
                    ["xdg-open", path_str],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            logger.debug(f"Opened: {path_str}")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Failed to open '{path_str}': Command not found - {e}")
            return False
        except OSError as e:
            logger.error(f"Failed to open '{path_str}': {e}")
            return False
    
    def get_safe_zones(self) -> List[Path]:
        """
        Get the default safe zones for file operations.
        
        These are directories where file deletion/modification is auto-approved
        by the Warden security system.
        
        Returns:
            List of Path objects representing safe zones.
        """
        safe_zones = [
            self.downloads_dir,
            self.temp_dir,
        ]
        
        # Add OS-specific safe zones
        if self.is_windows:
            # Windows: Public Downloads
            public_downloads = Path("C:/Users/Public/Downloads")
            if public_downloads.exists():
                safe_zones.append(public_downloads)
        else:
            # Linux/macOS: /tmp is already covered by temp_dir
            pass
        
        return safe_zones


# =============================================================================
# Module-Level Accessor
# =============================================================================

_os_context: Optional[OSContext] = None


def get_os_context() -> OSContext:
    """
    Get the global OSContext singleton.
    
    Returns:
        The OSContext instance.
    """
    global _os_context
    if _os_context is None:
        _os_context = OSContext()
    return _os_context


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    "OSContext",
    "get_os_context",
]
