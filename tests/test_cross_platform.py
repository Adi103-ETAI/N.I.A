"""Cross-Platform Compatibility Tests.

Tests for OS detection, feature availability, and graceful degradation.
"""
import pytest
from src.core.os import get_os_context
from src.core.features import get_features


class TestOSContext:
    """Test OSContext platform detection."""

    def test_os_context_singleton(self):
        """OSContext should be a singleton."""
        ctx1 = get_os_context()
        ctx2 = get_os_context()
        assert ctx1 is ctx2

    def test_os_detection(self):
        """OS should be detected as one of the supported types."""
        ctx = get_os_context()
        assert ctx.os_name in ("windows", "linux", "darwin")

    def test_platform_flags(self):
        """Platform flags should be mutually exclusive."""
        ctx = get_os_context()
        flags = [ctx.is_windows, ctx.is_linux, ctx.is_macos]
        assert sum(flags) == 1  # Exactly one should be True

    def test_path_attributes(self):
        """Path attributes should exist and be valid."""
        ctx = get_os_context()
        assert ctx.home_dir.exists()
        assert ctx.desktop_dir.exists()
        assert ctx.downloads_dir.exists()
        assert ctx.temp_dir.exists()

    def test_open_file_signature(self):
        """open_file should accept path and return bool."""
        ctx = get_os_context()
        # Test with invalid path - should return False
        result = ctx.open_file("/nonexistent/path.txt")
        assert isinstance(result, bool)

    def test_safe_zones(self):
        """Safe zones should always include downloads."""
        ctx = get_os_context()
        safe_zones = ctx.get_safe_zones()
        assert len(safe_zones) > 0
        assert ctx.downloads_dir in safe_zones


class TestPlatformFeatures:
    """Test PlatformFeatures detection."""

    def test_features_singleton(self):
        """PlatformFeatures should be a singleton."""
        feat1 = get_features()
        feat2 = get_features()
        assert feat1 is feat2

    def test_features_dict(self):
        """get_all should return a dict of features."""
        feat = get_features()
        all_features = feat.get_all()
        assert isinstance(all_features, dict)
        assert len(all_features) > 0

    def test_has_method(self):
        """has() should return bool for all features."""
        feat = get_features()
        for feature_name in feat.get_all():
            result = feat.has(feature_name)
            assert isinstance(result, bool)

    def test_platform_specific_features(self):
        """Platform-specific features should be detected correctly."""
        feat = get_features()
        ctx = get_os_context()

        if ctx.is_windows:
            # Windows may have UIAutomation/pygetwindow
            assert isinstance(feat.has("windows_uiautomation"), bool)
            assert isinstance(feat.has("pygetwindow"), bool)
        elif ctx.is_linux:
            # Linux may have xdotool
            assert isinstance(feat.has("xdotool"), bool)
        elif ctx.is_macos:
            # macOS may have Core Audio
            assert isinstance(feat.has("core_audio"), bool)

    def test_cross_platform_features(self):
        """Cross-platform features should be checked on all OSes."""
        feat = get_features()
        # These should exist on all platforms (though may not be available)
        assert "pyautogui" in feat.get_all()
        assert "sounddevice" in feat.get_all()

    def test_summary(self):
        """summary() should return a readable string."""
        feat = get_features()
        summary = feat.summary()
        assert isinstance(summary, str)
        assert "Platform:" in summary
        assert "Python:" in summary
        assert "Features:" in summary


class TestCrossPlatformConsistency:
    """Test that APIs are consistent across platforms."""

    def test_os_context_methods_available(self):
        """All OSContext methods should be available."""
        ctx = get_os_context()
        assert hasattr(ctx, "get_shell_command")
        assert hasattr(ctx, "open_file")
        assert hasattr(ctx, "get_safe_zones")
        assert callable(ctx.get_shell_command)
        assert callable(ctx.open_file)
        assert callable(ctx.get_safe_zones)

    def test_features_methods_available(self):
        """All PlatformFeatures methods should be available."""
        feat = get_features()
        assert hasattr(feat, "has")
        assert hasattr(feat, "get_all")
        assert hasattr(feat, "summary")
        assert callable(feat.has)
        assert callable(feat.get_all)
        assert callable(feat.summary)

    def test_path_operations(self):
        """Basic path operations should work."""
        from pathlib import Path
        ctx = get_os_context()

        # Should be able to resolve paths
        test_path = ctx.downloads_dir / "test.txt"
        assert isinstance(test_path, Path)

        # Should be able to resolve relative to base
        from src.core.config import settings
        base_path = settings.BASE_DIR
        assert base_path.exists()
        assert base_path.is_dir()
