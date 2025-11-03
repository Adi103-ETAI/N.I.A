"""Registry-based Tool Manager for NIA: Handles discovery, registration, execution of tools, with hooks for permissions and voice features."""
from typing import Any, Dict, List, Type, Optional, Callable
import importlib
import inspect
import logging
import pkgutil
import asyncio
from .hello_tool import HelloTool
import os
import importlib.util

__all__ = ["ToolManager"]

class ToolManager:
    """NIA ToolManager: central registry and orchestrator for tools.

    - register_tool: add tool (class or instance)
    - list_tools: enumerate all tools
    - execute: sync execution
    - aexecute: async execution (if tool supports it)
    - discover_and_register: dynamic module import
    - TODO: permissions, sandbox, timeout, cancellation
    """
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._registry: Dict[str, Any] = {}
        self._plugin_sources: Dict[str, str] = {}  # map tool_name -> file_path
        self._permissions: Dict[str, Callable[[str, Dict], bool]] = {}  # Tool perms stub

    def register_tool(self, tool: Any) -> None:
        if hasattr(tool, "run") and hasattr(tool, "name") and not inspect.isclass(tool):
            tool_instance = tool
        else:
            try:
                tool_instance = tool()
            except Exception as exc:
                self.logger.error(f"Failed to instantiate tool {tool}: {exc}")
                raise
        name = getattr(tool_instance, "name", None)
        if not name or not hasattr(tool_instance, "run"):
            raise ValueError("Tool must have a 'name' and a callable 'run' method")
        self._registry[name] = tool_instance
        self.logger.info(f"Registered tool: {name}")

    def list_tools(self) -> List[str]:
        return list(self._registry.keys())

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._registry

    def set_permission(self, tool_name: str, check: Callable[[str, Dict], bool]) -> None:
        """Stub: set a permission check callable for a tool."""
        self._permissions[tool_name] = check

    def _check_permission(self, tool_name: str, user: str, params: Dict) -> bool:
        checker = self._permissions.get(tool_name)
        if checker:
            return checker(user, params)
        return True  # Default open policy for demo

    def execute(self, tool_name: str, params: Dict[str, Any], user: str = "system") -> Dict[str, Any]:
        self.logger.info(f"Executing tool '{tool_name}' with params: {params}")
        tool = self._registry.get(tool_name)
        if not tool:
            err = f"Tool '{tool_name}' is not registered"
            self.logger.error(err)
            return {"success": False, "error": err}
        if not self._check_permission(tool_name, user, params):
            err = f"Permission denied for tool '{tool_name}'"
            self.logger.warning(err)
            return {"success": False, "error": err}
        try:
            result = tool.run(params or {})
            if not isinstance(result, dict):
                result = {"output": result}
            return {"success": True, "output": result}
        except Exception as exc:
            self.logger.exception(f"Tool '{tool_name}' raised an exception")
            return {"success": False, "error": str(exc)}

    async def aexecute(self, tool_name: str, params: Dict[str, Any], user: str = "system") -> Dict[str, Any]:
        tool = self._registry.get(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool '{tool_name}' is not registered"}
        if not self._check_permission(tool_name, user, params):
            return {"success": False, "error": f"Permission denied for tool '{tool_name}'"}
        if hasattr(tool, "arun") and inspect.iscoroutinefunction(tool.arun):
            try:
                result = await tool.arun(params or {})
                if not isinstance(result, dict):
                    result = {"output": result}
                return {"success": True, "output": result}
            except Exception as exc:
                self.logger.exception(f"Tool '{tool_name}' async failed")
                return {"success": False, "error": str(exc)}
        else:
            # fallback to sync execution in ThreadPool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.execute(tool_name, params, user=user))

    def discover_and_register(self) -> None:
        """Discover tools in core.tools (excluding this file) and register classes/instances."""
        self.logger.info(f"Discovering tools in package '{__name__}'")
        import os
        pkg_path = os.path.dirname(__file__)
        for finder, mod_name, ispkg in pkgutil.iter_modules([pkg_path]):
            if mod_name == "__init__":
                continue
            try:
                full_name = f"core.tools.{mod_name}"
                module = importlib.import_module(full_name)
                self.logger.debug(f"Imported tool module: {full_name}")
                for _, cls in inspect.getmembers(module, inspect.isclass):
                    if getattr(cls, "name", None) and hasattr(cls, "run"):
                        try:
                            self.register_tool(cls)
                        except Exception as exc:
                            self.logger.warning(f"Failed to register {cls} from {full_name}: {exc}")
                for _, obj in inspect.getmembers(module):
                    if not inspect.isclass(obj) and hasattr(obj, "name") and hasattr(obj, "run"):
                        try:
                            self.register_tool(obj)
                        except Exception as exc:
                            self.logger.warning(f"Failed to register {obj} from {full_name}: {exc}")
            except Exception as exc:
                self.logger.warning(f"Failed to import tool mod {mod_name}: {exc}")

    def load_plugins_from_directory(self, directory: str) -> None:
        """Dynamically load and register Tool-compatible plugins from the given directory."""
        loaded = 0
        for fname in os.listdir(directory):
            if not fname.endswith('.py') or fname.startswith('__'):
                continue
            mod_name = fname[:-3]
            plugin_path = os.path.join(directory, fname)
            try:
                spec = importlib.util.spec_from_file_location(mod_name, plugin_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module) # type: ignore
                for symbol in dir(module):
                    obj = getattr(module, symbol)
                    if hasattr(obj, 'name') and hasattr(obj, 'run'):
                        self.register_tool(obj)
                        self._plugin_sources[getattr(obj, 'name')] = plugin_path
                        loaded += 1
            except Exception as exc:
                self.logger.warning(f"Failed to load plugin '{fname}': {exc}")
        self.logger.info(f"Loaded {loaded} plugin tools from: {directory}")

    def reload_plugins(self):
        """Unload, then reload all plugin tools from the default plugins directory."""
        plugin_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'plugins'))
        self.unload_all_plugins()
        self.load_plugins_from_directory(plugin_dir)

    def unload_plugin(self, tool_name: str) -> bool:
        """Unregister a plugin tool only (not core tools)."""
        if tool_name in self._plugin_sources:
            self._registry.pop(tool_name, None)
            self._plugin_sources.pop(tool_name, None)
            self.logger.info(f"Unloaded plugin tool: {tool_name}")
            return True
        self.logger.info(f"Cannot unload '{tool_name}': not a plugin tool.")
        return False

    def unload_all_plugins(self) -> int:
        """Unload all plugin tools that were sourced from plugins directory."""
        count = 0
        for tool in list(self._plugin_sources):
            self.unload_plugin(tool)
            count += 1
        return count

    def plugin_tools(self) -> List[str]:
        """Return a list of currently loaded plugin tool names."""
        return list(self._plugin_sources)

    # Voice/audio hooks (stubs): enable in assistant pipeline
    def speak(self, text: str) -> None:
        """Text-to-Speech output using pyttsx3 (if available)."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            # Fail silent, fallback to print only if needed
            pass

    def listen(self) -> Optional[str]:
        """Speech-to-Text input using speech_recognition (if available). Blocking, returns transcript, or None if unavailable."""
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Listening (speak now)...")
                audio = r.listen(source, timeout=5, phrase_time_limit=8)
            try:
                text = r.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except Exception:
                return None
        except Exception:
            return None

# Optionally auto-register hello tool for development
# In a full app, this could be driven by env/config
def _auto_register(tools: ToolManager):
    try:
        tools.register_tool(HelloTool)
    except Exception:
        pass
