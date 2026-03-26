"""Feature Availability Detection.

Detects which optional features are available based on installed packages
and current platform. Enables graceful degradation on unsupported platforms.
"""
from __future__ import annotations

import platform
import sys
from typing import Dict

from src.core.logger import setup_logger

logger = setup_logger("FEATURES")


class PlatformFeatures:
    """Singleton for feature availability across platforms."""

    _instance: PlatformFeatures | None = None

    def __new__(cls) -> PlatformFeatures:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Detect available features based on platform and installed packages."""
        system = platform.system().lower()
        self.is_windows = system == "windows"
        self.is_linux = system == "linux"
        self.is_macos = system == "darwin"

        self._features: Dict[str, bool] = {}
        self._detect_desktop_automation()
        self._detect_audio_control()
        self._detect_gpu_support()

        logger.debug(f"Features initialized on {system}: {self._features}")

    def _detect_desktop_automation(self) -> None:
        """Check for desktop automation capabilities."""
        # UIAutomation - Windows only
        self._features["windows_uiautomation"] = False
        if self.is_windows:
            try:
                import pywinauto  # noqa: F401
                self._features["windows_uiautomation"] = True
                logger.debug("✓ Windows UIAutomation available")
            except ImportError:
                logger.debug("✗ Windows UIAutomation not available")

        # PyAutoGUI - Cross-platform but optional
        self._features["pyautogui"] = False
        try:
            import pyautogui  # noqa: F401
            self._features["pyautogui"] = True
            logger.debug("✓ PyAutoGUI available")
        except ImportError:
            logger.debug("✗ PyAutoGUI not available")

        # PyGetWindow - Windows only
        self._features["pygetwindow"] = False
        if self.is_windows:
            try:
                import pygetwindow  # noqa: F401
                self._features["pygetwindow"] = True
                logger.debug("✓ PyGetWindow available")
            except ImportError:
                logger.debug("✗ PyGetWindow not available")

        # xdotool - Linux only
        self._features["xdotool"] = False
        if self.is_linux:
            try:
                import subprocess
                result = subprocess.run(
                    ["which", "xdotool"],
                    capture_output=True,
                    timeout=2,
                )
                self._features["xdotool"] = result.returncode == 0
                if self._features["xdotool"]:
                    logger.debug("✓ xdotool available")
                else:
                    logger.debug("✗ xdotool not available")
            except Exception as e:
                logger.debug(f"✗ xdotool check failed: {e}")

    def _detect_audio_control(self) -> None:
        """Check for audio control capabilities."""
        # PyCaw - Windows audio control
        self._features["pycaw"] = False
        if self.is_windows:
            try:
                import pycaw  # noqa: F401
                self._features["pycaw"] = True
                logger.debug("✓ PyCaw available")
            except ImportError:
                logger.debug("✗ PyCaw not available")

        # SoundDevice - Cross-platform
        self._features["sounddevice"] = False
        try:
            import sounddevice  # noqa: F401
            self._features["sounddevice"] = True
            logger.debug("✓ SoundDevice available")
        except (ImportError, OSError) as e:
            # OSError for missing PortAudio library (expected in headless envs)
            logger.debug(f"✗ SoundDevice not available: {type(e).__name__}")

        # Core Audio - macOS only
        self._features["core_audio"] = False
        if self.is_macos:
            try:
                import AVFoundation  # noqa: F401
                self._features["core_audio"] = True
                logger.debug("✓ Core Audio (AVFoundation) available")
            except ImportError:
                logger.debug("✗ Core Audio not available")

    def _detect_gpu_support(self) -> None:
        """Check for GPU/accelerator support."""
        # CUDA
        self._features["cuda"] = False
        try:
            import torch
            self._features["cuda"] = torch.cuda.is_available()
            if self._features["cuda"]:
                logger.debug(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.debug("✗ CUDA not available")
        except ImportError:
            logger.debug("✗ PyTorch/CUDA not available")

    def has(self, feature: str) -> bool:
        """Check if a feature is available.

        Args:
            feature: Feature name (e.g., "windows_uiautomation", "pyautogui")

        Returns:
            True if the feature is available, False otherwise.
        """
        available = self._features.get(feature, False)
        if not available and feature in self._features:
            logger.debug(f"Feature '{feature}' requested but not available")
        return available

    def get_all(self) -> Dict[str, bool]:
        """Get all detected features.

        Returns:
            Dict mapping feature names to availability.
        """
        return self._features.copy()

    def summary(self) -> str:
        """Get human-readable feature summary.

        Returns:
            Multi-line string describing available features.
        """
        lines = [
            f"Platform: {platform.system()} {platform.release()}",
            f"Python: {sys.version.split()[0]}",
            "Features:",
        ]

        for feature, available in sorted(self._features.items()):
            status = "✓" if available else "✗"
            lines.append(f"  {status} {feature}")

        return "\n".join(lines)


def get_features() -> PlatformFeatures:
    """Get the global PlatformFeatures singleton.

    Returns:
        PlatformFeatures instance.
    """
    return PlatformFeatures()


__all__ = ["PlatformFeatures", "get_features"]
