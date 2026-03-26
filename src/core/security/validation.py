"""Validation helpers shared by security enforcement layers."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class SecurityError(Exception):
    """Raised when a security check fails."""


def validate_app_launcher(args: dict[str, Any]) -> None:
    """Verify app launch request contains required fields."""
    app_name = args.get("app_name", "")
    if not app_name:
        raise SecurityError("No app_name provided.")


def validate_file_operation_paths(
    args: dict[str, Any],
    safe_zones: list[Path],
) -> None:
    """Enforce that file paths stay inside configured safe zones."""
    paths_to_check: list[str] = []
    if "path" in args:
        paths_to_check.append(args["path"])
    if "src" in args:
        paths_to_check.append(args["src"])
    if "dst" in args:
        paths_to_check.append(args["dst"])

    for raw_path in paths_to_check:
        path = Path(raw_path).resolve()

        is_safe_zone = False
        for zone in safe_zones:
            try:
                path.relative_to(zone)
                is_safe_zone = True
                break
            except ValueError:
                continue

        if not is_safe_zone:
            raise SecurityError(
                f"Access Denied: Path '{path}' is outside Allowed Workspace (Safe Zones)."
            )


__all__ = ["SecurityError", "validate_app_launcher", "validate_file_operation_paths"]
