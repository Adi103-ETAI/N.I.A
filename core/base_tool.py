from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for tools.

    Tools should provide a module-level `name` attribute (string) and
    implement the `run(**params)` method. The concrete tools in
    `core.tools` are updated to subclass this for clarity and typing.
    """

    name: str = ""
    description: str = ""

    def __init__(self) -> None:
        if not getattr(self, "name", None):
            raise ValueError("Tool must define a `name` attribute")

    @abstractmethod
    def run(self, **params) -> Any:
        """Run the tool with keyword parameters and return a result.

        Implementations may be synchronous or async (coroutines).
        """
        raise NotImplementedError
