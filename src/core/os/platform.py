"""OS Context — Cross-Platform Abstraction Layer.

Provides a singleton ``OSContext`` class that abstracts all OS-specific logic,
enabling N.I.A. to run on Windows, Linux, and macOS without platform guards
scattered throughout the codebase.

This module is the **canonical implementation**. The public API is exposed
through ``src.core.os`` (package ``__init__.py``) and a backward-compat shim
at ``src.core.context``.

Usage::

    from src.core.os import get_os_context        # preferred (new)
    from src.core.context import get_os_context   # also works (shim)

    ctx = get_os_context()
    print(ctx.os_name)            # "windows", "linux", or "darwin"
    print(ctx.downloads_dir)      # platform-appropriate Downloads path
    ctx.open_file("document.pdf") # uses the correct OS command
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
    """Singleton class for OS-specific context and cross-platform utilities.

    Instantiated once per process and cached.  All subsequent calls to
    ``get_os_context()`` return the same instance.

    Attributes:
        os_name (str): Normalised OS name — ``"windows"``, ``"linux"``, or
            ``"darwin"``.
        is_windows (bool): True when running on Windows.
        is_linux   (bool): True when running on Linux.
        is_macos   (bool): True when running on macOS.
        home_dir      (Path): User's home directory.
        desktop_dir   (Path): User's Desktop directory (created if absent).
        downloads_dir (Path): User's Downloads directory (created if absent).
        temp_dir      (Path): System temp directory.
    """

    _instance: Optional["OSContext"] = None

    def __new__(cls) -> "OSContext":
        """Enforce singleton — only one OSContext per process."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialization (called once)
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Detect OS and resolve standard directory paths."""
        system = platform.system().lower()

        if system == "windows":
            self.os_name = "windows"
        elif system == "linux":
            self.os_name = "linux"
        elif system == "darwin":
            self.os_name = "darwin"
        else:
            self.os_name = system
            logger.warning(f"Unknown OS detected: {system}")

        self.is_windows: bool = self.os_name == "windows"
        self.is_linux:   bool = self.os_name == "linux"
        self.is_macos:   bool = self.os_name == "darwin"

        self.home_dir:      Path = Path.home()
        self.desktop_dir:   Path = self._resolve_desktop()
        self.downloads_dir: Path = self._resolve_downloads()
        self.temp_dir:      Path = self._resolve_temp()

        logger.debug(f"OSContext initialised: {self.os_name}")
        logger.debug(f"  Home:      {self.home_dir}")
        logger.debug(f"  Desktop:   {self.desktop_dir}")
        logger.debug(f"  Downloads: {self.downloads_dir}")

    # ------------------------------------------------------------------
    # Directory Resolvers
    # ------------------------------------------------------------------

    def _resolve_desktop(self) -> Path:
        """Resolve the user's Desktop directory (created if missing)."""
        if self.is_windows:
            desktop = self.home_dir / "Desktop"
        else:
            xdg = os.environ.get("XDG_DESKTOP_DIR")
            desktop = Path(xdg) if xdg else self.home_dir / "Desktop"

        desktop.mkdir(parents=True, exist_ok=True)
        return desktop

    def _resolve_downloads(self) -> Path:
        """Resolve the user's Downloads directory (created if missing)."""
        if self.is_windows:
            downloads = self.home_dir / "Downloads"
        else:
            xdg = os.environ.get("XDG_DOWNLOAD_DIR")
            downloads = Path(xdg) if xdg else self.home_dir / "Downloads"

        downloads.mkdir(parents=True, exist_ok=True)
        return downloads

    def _resolve_temp(self) -> Path:
        """Resolve the system temp directory."""
        import tempfile
        return Path(tempfile.gettempdir())

    # ------------------------------------------------------------------
    # Cross-Platform Methods
    # ------------------------------------------------------------------

    def get_shell_command(self, cmd: List[str]) -> List[str]:
        """Prepare a command list for subprocess execution.

        Currently returns the list unchanged; exists for future extensibility
        (e.g., wrapping in a shell-specific launcher on exotic platforms).

        Args:
            cmd: Command as a list of strings, e.g. ``["ls", "-la"]``.

        Returns:
            Command list ready for ``subprocess.Popen`` / ``subprocess.run``.
        """
        return cmd

    def open_file(self, path: str | Path) -> bool:
        """Open a file or URL with the system's default application.

        Dispatch table:
            - Windows → ``os.startfile``
            - macOS   → ``open``
            - Linux   → ``xdg-open``

        Args:
            path: File path or URL to open.

        Returns:
            ``True`` if the command was launched successfully, ``False``
            otherwise.
        """
        path_str = str(path)
        try:
            if self.is_windows:
                os.startfile(path_str)  # type: ignore[attr-defined]
            elif self.is_macos:
                subprocess.Popen(
                    ["open", path_str],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(
                    ["xdg-open", path_str],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            logger.debug(f"Opened: {path_str}")
            return True
        except FileNotFoundError as e:
            logger.error(f"Failed to open '{path_str}': {e}")
            return False
        except OSError as e:
            logger.error(f"Failed to open '{path_str}': {e}")
            return False

    def get_safe_zones(self) -> List[Path]:
        """Return directories where file mutations are auto-approved by the Warden.

        By default includes:
            - User Downloads directory
            - System temp directory
            - Project ``BASE_DIR`` (the sandboxed workspace)
            - Windows: ``C:/Users/Public/Downloads`` (if it exists)

        Returns:
            List of ``Path`` objects representing approved safe zones.
        """
        from src.core.config import settings  # local import — avoid circular dep

        safe_zones: List[Path] = [
            self.downloads_dir,
            self.temp_dir,
            settings.BASE_DIR,
        ]

        if self.is_windows:
            public = Path("C:/Users/Public/Downloads")
            if public.exists():
                safe_zones.append(public)

        return safe_zones


# =============================================================================
# Module-Level Accessor
# =============================================================================

_os_context: Optional[OSContext] = None


def get_os_context() -> OSContext:
    """Return the global ``OSContext`` singleton.

    Thread-safe after first instantiation (Python GIL protects the check).

    Returns:
        The single ``OSContext`` instance for this process.
    """
    global _os_context
    if _os_context is None:
        _os_context = OSContext()
    return _os_context


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "OSContext",
    "get_os_context",
]
