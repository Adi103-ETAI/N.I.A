"""Top-level tools package exports and legacy adapter.

This module exports `HelloTool` and `EchoTool` for convenience, and it
provides a legacy `ToolManager` adapter that wraps the consolidated
`core.tool_manager.ToolManager` but preserves the older dict-wrapping
API used by existing code and tests.
"""


from typing import Any, Dict, List, Optional, Callable
import logging

from ..tool_manager import ToolManager as RawToolManager
from .hello_tool import HelloTool
from .echo_tool import EchoTool
import os

__all__ = ["ToolManager", "HelloTool", "EchoTool", "register_dev_tools"]


class ToolManager:
    """Legacy adapter over `core.tool_manager.ToolManager`.

    This adapter keeps the old behavior where `execute` and `aexecute`
    return a dict: {"success": bool, "output": ..., "error": ...}.
    Internally it delegates to the raw ToolManager which raises on error
    and returns raw results.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._inner = RawToolManager(logger=self.logger)

    # Registration passthroughs
    def register_tool(self, tool: Any) -> None:
        return self._inner.register_tool(tool)

    def register(self, name: str, func: Callable) -> None:
        return self._inner.register(name, func)

    def register_decorator(self, name: Optional[str] = None):
        return self._inner.register_decorator(name)

    def discover_and_register(self, package_name: str = "core.tools") -> None:
        return self._inner.discover_and_register(package_name)

    def load_plugins_from_directory(self, directory: str) -> int:
        return self._inner.load_plugins_from_directory(directory)

    def unload_plugin(self, tool_name: str) -> bool:
        return self._inner.unload_plugin(tool_name)

    def unload_all_plugins(self) -> int:
        return self._inner.unload_all_plugins()

    def plugin_tools(self) -> List[str]:
        return self._inner.plugin_tools()

    def list_tools(self) -> List[str]:
        return self._inner.list_tools()

    def has_tool(self, tool_name: str) -> bool:
        return self._inner.has_tool(tool_name)

    def set_permission(self, tool_name: str, check: Callable[[str, Dict], bool]) -> None:
        return self._inner.set_permission(tool_name, check)

    # Legacy-wrapped execute
    def execute(self, tool_name: str, params: Dict[str, Any], user: str = "system") -> Dict[str, Any]:
        try:
            result = self._inner.execute(tool_name, params, timeout=params.get("_timeout") if isinstance(params, dict) else None, user=user)
            # Normalize into dict
            if isinstance(result, dict) and result.get("success") is not None:
                return result
            return {"success": True, "output": result}
        except Exception as exc:
            self.logger.exception("Tool execution failed")
            return {"success": False, "error": str(exc)}

    async def aexecute(self, tool_name: str, params: Dict[str, Any], user: str = "system") -> Dict[str, Any]:
        try:
            result = await self._inner.execute_async(tool_name, params, timeout=params.get("_timeout") if isinstance(params, dict) else None, user=user)
            if isinstance(result, dict) and result.get("success") is not None:
                return result
            return {"success": True, "output": result}
        except Exception as exc:
            self.logger.exception("Async tool execution failed")
            return {"success": False, "error": str(exc)}


def register_dev_tools(mgr: Any = None) -> Any:
    """Register built-in development tools (HelloTool, EchoTool).

    If `mgr` is None a new legacy `ToolManager` adapter instance is created and
    returned with the dev tools registered. If a manager is provided the
    function will attempt to register the tools on it. This is a best-effort
    helper and will ignore registration errors to remain safe during imports.
    """
    if mgr is None:
        mgr = ToolManager()

    for tool_cls in (HelloTool, EchoTool):
        try:
            mgr.register_tool(tool_cls)
        except Exception:
            # best-effort: skip failing registrations
            continue

    return mgr

# Optional environment-gated auto-registration to avoid hidden side effects.
# If NIA_AUTO_REGISTER_DEV_TOOLS is truthy ("1", "true", "yes"), register dev tools
# at import-time. Default behavior is no auto-registration.
if os.environ.get("NIA_AUTO_REGISTER_DEV_TOOLS", "").lower() in {"1", "true", "yes"}:
    try:
        register_dev_tools()
    except Exception:
        pass
