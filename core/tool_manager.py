"""Consolidated ToolManager implementation.

This module provides a single authoritative ToolManager with a low-level
API that returns raw tool results and raises exceptions on error. A
compatibility adapter in `core.tools` exposes the older dict-wrapping
API used by legacy code and tests.
"""

import importlib
import inspect
import importlib.util
import logging
import os
import pkgutil
import asyncio
from typing import Any, Callable, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError


class ToolManager:
    """Consolidated ToolManager (raw-return API).

    - register(name, func): register a plain callable tool
    - register_tool(tool): register a class or instance with `name` and `run`
    - register_decorator(name=None): decorator for registering functions
    - discover_and_register(package_name='core.tools')
    - execute(tool_name, params, timeout=None, user='system'): sync execution
    - execute_async(tool_name, params, timeout=None, user='system'): async execution
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._registry: Dict[str, Any] = {}
        self._plugin_sources: Dict[str, str] = {}
        self._permissions: Dict[str, Callable[[str, Dict], bool]] = {}

    # ---------------- Registration APIs -----------------
    def register(self, name: str, func: Callable) -> None:
        """Register a plain function as a tool (legacy-style)."""
        if not callable(func):
            raise ValueError("func must be callable")
        self._registry[name] = func
        self.logger.info(f"Registered legacy tool: {name}")

    def register_tool(self, tool: Any) -> None:
        """Register a tool class, instance, or named callable.

        If `tool` is a class it will be instantiated. The resulting instance
        must have a `name` attribute and a callable `run` method.
        """
        # plain callable without name must use register(name, func)
        if callable(tool) and not hasattr(tool, "name") and not inspect.isclass(tool):
            raise ValueError("Use register(name, func) for plain callables")

        instance = tool() if inspect.isclass(tool) else tool

        # instance should have name and run
        name = getattr(instance, "name", None)
        run_callable = getattr(instance, "run", None)
        if name and callable(run_callable):
            self._registry[name] = instance
            self.logger.info(f"Registered tool: {name}")
            return

        # fallback: named callable
        if callable(instance):
            fallback_name = getattr(instance, "tool_name", None) or getattr(instance, "__name__", None)
            if fallback_name:
                self._registry[fallback_name] = instance
                self.logger.info(f"Registered callable tool: {fallback_name}")
                return

        raise ValueError("Tool must have a 'name' attribute and a callable 'run' method (or be a named callable)")

    def register_decorator(self, name: Optional[str] = None):
        def decorator(func: Callable):
            tool_name = name or getattr(func, "__name__", None)
            if not tool_name:
                raise ValueError("Tool must have a name")
            self.register(tool_name, func)
            return func

        return decorator

    # ---------------- Discovery / Plugins -----------------
    def discover_and_register(self, package_name: str = "core.tools") -> None:
        try:
            package = importlib.import_module(package_name)
        except ModuleNotFoundError:
            self.logger.warning(f"Package '{package_name}' not found for discovery")
            return

        package_path = getattr(package, "__path__", None)
        if not package_path:
            self.logger.warning(f"Package '{package_name}' has no __path__; cannot iterate modules")
            return

        for _, module_name, _ in pkgutil.iter_modules(package_path):
            if module_name == "__init__":
                continue
            full_name = f"{package_name}.{module_name}"
            try:
                module = importlib.import_module(full_name)
            except Exception as exc:
                self.logger.error(f"Failed to import module '{full_name}': {exc}", exc_info=True)
                continue

            # register classes with name/run
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if getattr(cls, "name", None) and hasattr(cls, "run"):
                    try:
                        self.register_tool(cls)
                    except Exception:
                        self.logger.exception(f"Failed registering tool class {cls} from {full_name}")

            # register functions
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                tool_name = getattr(obj, "tool_name", None) or name
                if getattr(obj, "_is_tool", False) or tool_name:
                    try:
                        self.register(tool_name, obj)
                    except Exception:
                        self.logger.exception(f"Failed registering function tool {tool_name} from {full_name}")

    def load_plugins_from_directory(self, directory: str) -> int:
        loaded = 0
        if not os.path.isdir(directory):
            self.logger.warning("Plugin directory not found: %s", directory)
            return loaded
        safe_mode = os.environ.get("NIA_PLUGIN_SAFE_MODE", "").lower() in {"1", "true", "yes"}
        allowlist: Optional[set[str]] = None
        if safe_mode:
            allow_path = os.path.join(directory, "ALLOWLIST.txt")
            if os.path.exists(allow_path):
                try:
                    with open(allow_path, "r", encoding="utf-8") as f:
                        allowlist = {line.strip() for line in f if line.strip()}
                except Exception:
                    allowlist = set()
            else:
                self.logger.warning("Safe mode enabled but no ALLOWLIST.txt found in %s", directory)
                allowlist = set()
        for fname in os.listdir(directory):
            if not fname.endswith(".py") or fname.startswith("__"):
                continue
            base = os.path.splitext(fname)[0]
            if safe_mode and allowlist is not None and base not in allowlist:
                self.logger.info("Skipping plugin %s due to safe mode allowlist", fname)
                continue
            path = os.path.join(directory, fname)
            mod_name = f"nia_plugin_{base}"
            try:
                spec = importlib.util.spec_from_file_location(mod_name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore
                for symbol in dir(module):
                    obj = getattr(module, symbol)
                    if hasattr(obj, "name") and hasattr(obj, "run"):
                        self.register_tool(obj)
                        self._plugin_sources[getattr(obj, "name")] = path
                        loaded += 1
            except Exception as exc:
                self.logger.warning("Failed to load plugin %s: %s", fname, exc)
        self.logger.info("Loaded %d plugin tools from %s", loaded, directory)
        return loaded

    def unload_plugin(self, tool_name: str) -> bool:
        if tool_name in self._plugin_sources:
            self._registry.pop(tool_name, None)
            self._plugin_sources.pop(tool_name, None)
            self.logger.info(f"Unloaded plugin tool: {tool_name}")
            return True
        self.logger.info(f"Cannot unload '{tool_name}': not a plugin tool.")
        return False

    def unload_all_plugins(self) -> int:
        count = 0
        for t in list(self._plugin_sources):
            self.unload_plugin(t)
            count += 1
        return count

    def reload_plugins(self, directory: str) -> int:
        """Reload plugins from `directory`.

        This unloads any currently loaded plugin tools and attempts to load
        plugins from the provided directory. It returns the number of plugin
        tools successfully loaded. Errors during unload/load are logged but
        do not raise.
        """
        try:
            self.unload_all_plugins()
        except Exception:
            self.logger.exception("Failed to unload plugins during reload")
        try:
            return self.load_plugins_from_directory(directory)
        except Exception:
            self.logger.exception("Failed to load plugins during reload")
            return 0

    def plugin_tools(self) -> List[str]:
        return list(self._plugin_sources)

    def list_tools(self) -> List[str]:
        return list(self._registry.keys())

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._registry

    # ---------------- Permissions -----------------
    def set_permission(self, tool_name: str, check: Callable[[str, Dict], bool]) -> None:
        self._permissions[tool_name] = check

    def _check_permission(self, tool_name: str, user: str, params: Dict) -> bool:
        checker = self._permissions.get(tool_name)
        if checker:
            return checker(user, params)
        return True

    # ---------------- Execution -----------------
    def _invoke(self, func: Callable, params: dict):
        """Call func with params trying keyword args first, then single dict, then no-arg."""
        # Try keyword call
        try:
            return func(**params)
        except TypeError:
            pass
        # Try single dict parameter
        try:
            return func(params)
        except TypeError:
            pass
        # Try no-arg
        return func()

    def _run_sync_with_timeout(self, func: Callable, params: dict, timeout: Optional[float]):
        if timeout is None:
            return self._invoke(func, params)
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(lambda: self._invoke(func, params))
            try:
                return future.result(timeout=timeout)
            except FutureTimeoutError:
                future.cancel()
                raise TimeoutError(f"Tool execution exceeded timeout of {timeout} seconds")

    async def execute_async(self, tool_name: str, params: dict, timeout: Optional[float] = None, user: str = "system") -> Any:
        if tool_name not in self._registry:
            raise ValueError(f"Tool '{tool_name}' not found")
        if not self._check_permission(tool_name, user, params):
            raise PermissionError("Permission denied")

        tool = self._registry[tool_name]

        # If tool is a plain coroutine function registered via register(name, func)
        if callable(tool) and asyncio.iscoroutinefunction(tool):
            try:
                coro = tool(**params)
                return await asyncio.wait_for(coro, timeout=timeout) if timeout is not None else await coro
            except asyncio.TimeoutError:
                raise RuntimeError(f"Tool execution exceeded timeout of {timeout} seconds")
            except Exception as exc:
                raise RuntimeError(f"Tool '{tool_name}' failed: {exc}") from exc

        # If tool is an instance with arun/run
        run_coro = None
        if hasattr(tool, "arun") and callable(getattr(tool, "arun")):
            run_coro = getattr(tool, "arun")
        elif hasattr(tool, "run") and asyncio.iscoroutinefunction(getattr(tool, "run")):
            run_coro = getattr(tool, "run")

        if run_coro is not None and asyncio.iscoroutinefunction(run_coro):
            try:
                # try kwargs then single dict
                try:
                    coro = run_coro(**params)
                except TypeError:
                    coro = run_coro(params)
                return await asyncio.wait_for(coro, timeout=timeout) if timeout is not None else await coro
            except asyncio.TimeoutError:
                raise RuntimeError(f"Tool execution exceeded timeout of {timeout} seconds")
            except Exception as exc:
                raise RuntimeError(f"Tool '{tool_name}' failed: {exc}") from exc

        # sync fallback: run in executor
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, lambda: self._run_tool_sync(tool, params, timeout))
        except TimeoutError:
            raise RuntimeError(f"Tool execution exceeded timeout of {timeout} seconds")
        except Exception as exc:
            raise RuntimeError(f"Tool '{tool_name}' failed: {exc}") from exc

    def _run_tool_sync(self, tool: Any, params: dict, timeout: Optional[float]):
        # tool can be an instance with run(kwargs) or a plain callable
        if hasattr(tool, "run") and callable(getattr(tool, "run")):
            run = getattr(tool, "run")
            return self._run_sync_with_timeout(lambda *a, **k: run(*a, **k), params, timeout)
        if callable(tool):
            return self._run_sync_with_timeout(lambda *a, **k: tool(*a, **k), params, timeout)
        raise RuntimeError("Tool is not callable")

    def execute(self, tool_name: str, params: dict, timeout: Optional[float] = None, user: str = "system") -> Any:
        if tool_name not in self._registry:
            raise ValueError(f"Tool '{tool_name}' not found")
        if not self._check_permission(tool_name, user, params):
            raise PermissionError("Permission denied")
        tool = self._registry[tool_name]
        # if tool.run is coroutine function, run with asyncio.run if no loop
        run_func = None
        if hasattr(tool, "run"):
            run_func = getattr(tool, "run")
        # If the registered tool is a plain coroutine function, run it properly
        if callable(tool) and asyncio.iscoroutinefunction(tool):
            try:
                coro = tool(**params)
                return asyncio.run(asyncio.wait_for(coro, timeout=timeout) if timeout is not None else coro)
            except asyncio.TimeoutError:
                raise RuntimeError(f"Tool execution exceeded timeout of {timeout} seconds")
            except Exception as exc:
                raise RuntimeError(f"Tool '{tool_name}' failed: {exc}") from exc

        if asyncio.iscoroutinefunction(run_func):
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    raise RuntimeError("Event loop running; use execute_async")
            except RuntimeError:
                # no running loop
                return asyncio.run(self.execute_async(tool_name, params, timeout, user=user))
        try:
            return self._run_tool_sync(tool, params, timeout)
        except TimeoutError:
            raise RuntimeError(f"Tool execution exceeded timeout of {timeout} seconds")
        except Exception as exc:
            raise RuntimeError(f"Tool '{tool_name}' failed: {exc}") from exc
