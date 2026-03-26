"""Warden security interceptor (Operation Iron Cage)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.bus import get_event_bus
from src.core.logger import setup_logger
from src.core.os import get_os_context
from src.core.security.policies import (
    is_allowed_host_process_tool,
    is_auto_approved_tool,
    is_blocked_host_file_tool,
    is_restricted_file_tool,
)
from src.core.security.validation import (
    SecurityError,
    validate_app_launcher,
    validate_file_operation_paths,
)

try:
    import send2trash as _send2trash

    _HAS_TRASH = _send2trash is not None
except ImportError:
    _HAS_TRASH = False

logger = setup_logger("WARDEN")


class WardenService:
    """Security interceptor using Smart Trust logic."""

    def __init__(self) -> None:
        self.bus = None
        self.safe_zones: list[Path] = []
        self.safe_extensions = {
            ".tmp",
            ".log",
            ".bak",
            ".cache",
            ".txt",
            ".md",
            ".json",
            ".csv",
        }

    def start(self) -> None:
        """Initialize the Warden service."""
        try:
            self.safe_zones = get_os_context().get_safe_zones()
            logger.debug("SAFE_ZONES loaded: %s", [str(z) for z in self.safe_zones])

            try:
                self.bus = get_event_bus()
            except Exception:
                pass

            logger.info("🛡️ Warden Service Active (Blocking Mode)")
        except Exception as e:
            logger.error("❌ Failed to start Warden: %s", e)

    def check_permission(self, tool_name: str, args: dict[str, Any]) -> None:
        """Check if a high-risk operation is allowed."""
        logger.info("👮 Warden Check: %s", tool_name)

        if not self.safe_zones:
            self.safe_zones = get_os_context().get_safe_zones()

        if is_auto_approved_tool(tool_name):
            pass
        elif is_blocked_host_file_tool(tool_name):
            raise SecurityError(
                "Security Restriction: You must use the sandboxed_shell tool for file operations."
            )
        elif tool_name == "launch_app":
            validate_app_launcher(args)
        elif is_allowed_host_process_tool(tool_name):
            pass
        elif is_restricted_file_tool(tool_name):
            validate_file_operation_paths(args, self.safe_zones)
        else:
            raise SecurityError(f"Unknown high-risk tool '{tool_name}' blocked by default.")

        if tool_name == "delete_file" and not _HAS_TRASH:
            logger.warning("⚠️ send2trash missing. Permanent delete allowed inside Safe Zone.")

        logger.info("✅ Warden Approved: %s", tool_name)


_warden_instance: WardenService | None = None


def get_warden() -> WardenService:
    """Get the global Warden service instance."""
    global _warden_instance
    if _warden_instance is None:
        _warden_instance = WardenService()
        _warden_instance.start()
    return _warden_instance


def start_warden_service() -> None:
    """Legacy boot helper for Warden."""
    get_warden()


__all__ = ["SecurityError", "WardenService", "get_warden", "start_warden_service"]
