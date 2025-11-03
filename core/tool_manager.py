import importlib
import inspect
import logging
import pkgutil
from typing import Any, Callable, Dict

class ToolManager:
    def __init__(self, logger=None):
        self.tools: Dict[str, Callable] = {}
        self.logger = logger or logging.getLogger(__name__)

    def register_tool(self, tool_cls):
        """Register a tool class that has a 'name' and 'run' method."""
        try:
            instance = tool_cls()
            if not hasattr(instance, "name") or not hasattr(instance, "run"):
                raise AttributeError(f"Tool {tool_cls.__name__} missing required attributes 'name' or 'run'")
            self.tools[instance.name] = instance.run
            self.logger.info(f"Registered tool: {instance.name}")
        except Exception as e:
            self.logger.error(f"Failed to register tool {tool_cls}: {e}")

    def discover_and_register(self, package_name="core.tools"):
        """Auto-discover and register tools from a given package."""
        try:
            package = importlib.import_module(package_name)
        except ModuleNotFoundError:
            self.logger.warning(f"Package '{package_name}' not found for tool discovery")
            return

            # note the indentation fix — this should not be inside the except
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package_name}.{module_name}")
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if hasattr(obj, "run") and hasattr(obj, "name"):
                    self.register_tool(obj)

    def execute(self, tool_name: str, params: dict, timeout=None) -> Any:
        """Execute a registered tool by name."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        try:
            func = self.tools[tool_name]
            if "text" in params and "message" not in params:
                params["message"] = params.pop("text")

            return func(**params)
        except Exception as e:
            raise RuntimeError(f"Tool '{tool_name}' failed: {e}")
    # Backward compatibility for older code that calls .register()
    def register(self, name: str, func):
        """Backward-compatible alias for old 'register' API."""
        self.tools[name] = func
        self.logger.info(f"Registered legacy tool: {name}")
