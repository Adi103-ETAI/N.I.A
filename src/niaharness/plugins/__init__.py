"""Plugin discovery + command dispatch.

Focused extraction from hermes-agent/hermes_cli/plugins.py (2,333 LOC).

The full Hermes module has a PluginManager with lifecycle hooks, middleware,
tool registration, auxiliary tasks, toolsets, and more. This port covers the
essential APIs that the tui_gateway deep-ports depend on:
  - get_plugin_command_handler (slash command lookup)
  - resolve_plugin_command_result (async handler awaiting)
  - invoke_hook (lifecycle hook invocation)
  - get_plugin_commands (full command registry)

Plugin discovery scans ~/.nia/plugins/ for plugin manifests. When no plugins
are installed, all functions return empty/None — the TUI gateway's
command.dispatch falls through to built-in commands.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin discovery (simplified — scans ~/.nia/plugins/ for manifests)
# ---------------------------------------------------------------------------


class LoadedPlugin:
    """A loaded plugin with its manifest + registered commands."""

    def __init__(self, name: str, manifest: dict, path: Path):
        self.name = name
        self.manifest = manifest
        self.path = path
        self.commands: Dict[str, dict] = {}
        self.hooks: Dict[str, Callable] = {}


_plugin_manager: Optional["_PluginManager"] = None
_plugin_manager_lock = threading.Lock()


class _PluginManager:
    """Simplified plugin manager.

    Ported from hermes-agent/hermes_cli/plugins.py (PluginManager class).
    """

    def __init__(self):
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._plugin_commands: Dict[str, dict] = {}
        self._discovered = False

    def discover(self) -> None:
        """Discover plugins from ~/.nia/plugins/."""
        if self._discovered:
            return
        self._discovered = True
        try:
            plugins_dir = self._get_plugins_dir()
            if not plugins_dir.exists():
                return
            for entry in plugins_dir.iterdir():
                if not entry.is_dir():
                    continue
                manifest_path = entry / "plugin.json"
                if not manifest_path.exists():
                    continue
                try:
                    import json
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    name = manifest.get("name", entry.name)
                    plugin = LoadedPlugin(name, manifest, entry)
                    self._plugins[name] = plugin
                    # Register commands from manifest.
                    commands = manifest.get("commands", {})
                    if isinstance(commands, dict):
                        for cmd_name, cmd_info in commands.items():
                            if isinstance(cmd_info, dict):
                                self._plugin_commands[cmd_name] = {
                                    "handler": self._load_command_handler(entry, cmd_info),
                                    "description": cmd_info.get("description", ""),
                                    "plugin": name,
                                }
                except Exception as e:
                    logger.debug("Failed to load plugin %s: %s", entry, e)
        except Exception as e:
            logger.debug("Plugin discovery failed: %s", e)

    def _get_plugins_dir(self) -> Path:
        try:
            from niaharness.prompts.soul import get_nia_home
            return get_nia_home() / "plugins"
        except Exception:
            return Path(os.path.expanduser("~/.nia/plugins"))

    def _load_command_handler(self, plugin_dir: Path, cmd_info: dict) -> Optional[Callable]:
        """Load a command handler from a plugin's Python module."""
        module_path = cmd_info.get("module")
        function_name = cmd_info.get("function")
        if not module_path or not function_name:
            return None
        try:
            import importlib.util
            py_file = plugin_dir / f"{module_path.replace('.', '/')}.py"
            if not py_file.exists():
                return None
            spec = importlib.util.spec_from_file_location(
                f"nia_plugin_{plugin_dir.name}_{module_path}", py_file,
            )
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, function_name, None)
        except Exception as e:
            logger.debug("Failed to load command handler %s.%s: %s", module_path, function_name, e)
            return None

    def invoke_hook(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """Invoke a lifecycle hook on all loaded plugins.

        Returns a list of non-None return values from plugin callbacks.
        """
        self.discover()
        results: List[Any] = []
        for plugin in self._plugins.values():
            handler = plugin.hooks.get(hook_name)
            if handler is None:
                continue
            try:
                result = handler(**kwargs)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.debug("Plugin %s hook %s failed: %s", plugin.name, hook_name, e)
        return results

    def has_hook(self, hook_name: str) -> bool:
        """Return True when a hook has registered callbacks."""
        self.discover()
        return any(hook_name in p.hooks for p in self._plugins.values())


def get_plugin_manager() -> _PluginManager:
    """Return the singleton plugin manager."""
    global _plugin_manager
    if _plugin_manager is None:
        with _plugin_manager_lock:
            if _plugin_manager is None:
                _plugin_manager = _PluginManager()
    return _plugin_manager


def _ensure_plugins_discovered() -> _PluginManager:
    manager = get_plugin_manager()
    manager.discover()
    return manager


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Invoke a lifecycle hook on all loaded plugins.

    Ported from hermes-agent/hermes_cli/plugins.py line 2051.
    Returns a list of non-None return values from plugin callbacks.
    """
    return get_plugin_manager().invoke_hook(hook_name, **kwargs)


def has_hook(hook_name: str) -> bool:
    """Return True when a hook has registered callbacks."""
    return get_plugin_manager().has_hook(hook_name)


def get_plugin_command_handler(name: str) -> Optional[Callable]:
    """Return the handler for a plugin-registered slash command, or None.

    Ported from hermes-agent/hermes_cli/plugins.py line 2212.
    """
    entry = _ensure_plugins_discovered()._plugin_commands.get(name)
    return entry["handler"] if entry else None


_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS = 30.0


def resolve_plugin_command_result(result: Any) -> Any:
    """Resolve a plugin command return value, awaiting async handlers.

    Ported from hermes-agent/hermes_cli/plugins.py line 2221.

    Sync CLI/TUI dispatch sites call plugin handlers from plain functions.
    If a handler is async, await it directly when no loop is running; if
    we're already inside an active loop, run it in a helper thread with its
    own loop so the caller still gets a concrete result synchronously.
    """
    if not inspect.isawaitable(result):
        return result

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)

    outcome: Dict[str, Any] = {}
    failure: Dict[str, BaseException] = {}
    done = threading.Event()

    def _runner() -> None:
        try:
            outcome["value"] = asyncio.run(result)
        except BaseException as exc:
            failure["exc"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_runner, name="nia-plugin-command-await", daemon=True)
    thread.start()
    if not done.wait(timeout=_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS):
        raise TimeoutError(
            "Plugin command async handler did not complete within "
            f"{_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS:.0f}s"
        )
    if "exc" in failure:
        raise failure["exc"]
    return outcome.get("value")


def get_plugin_commands() -> Dict[str, dict]:
    """Return the full plugin commands dict.

    Ported from hermes-agent/hermes_cli/plugins.py line 2267.
    """
    return _ensure_plugins_discovered()._plugin_commands


def discover_plugins() -> _PluginManager:
    """Trigger idempotent plugin discovery."""
    return _ensure_plugins_discovered()


__all__ = [
    "invoke_hook",
    "has_hook",
    "get_plugin_command_handler",
    "resolve_plugin_command_result",
    "get_plugin_commands",
    "discover_plugins",
    "get_plugin_manager",
    "LoadedPlugin",
]
