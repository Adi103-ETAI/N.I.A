"""Base class for N.I.A. extensions."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseExtension(ABC):
    """
    Base class for all N.I.A. extensions.
    
    Extensions can add new capabilities, modify agent behavior,
    or integrate with external services.
    
    Example:
        class MyExtension(BaseExtension):
            def __init__(self):
                super().__init__(
                    name="my_extension",
                    version="1.0.0",
                    description="My custom extension"
                )
            
            def initialize(self):
                # Register capabilities, subscribe to events, etc.
                pass
            
            def shutdown(self):
                # Cleanup resources
                pass
    """
    
    def __init__(
        self,
        name: str = "unnamed",
        version: str = "0.0.0",
        description: str = "",
        dependencies: Optional[List[str]] = None
    ):
        self.name = name
        self.version = version
        self.description = description
        self.dependencies = dependencies or []
        self._initialized = False
        self._logger = logging.getLogger(f"extension.{name}")
    
    @property
    def is_initialized(self) -> bool:
        """Check if extension has been initialized."""
        return self._initialized
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the extension.
        
        Override this to:
        - Register capabilities
        - Subscribe to events
        - Set up resources
        """
        pass
    
    def shutdown(self) -> None:
        """
        Shutdown the extension.
        
        Override this to cleanup resources.
        """
        pass
    
    def get_capabilities(self) -> List[Any]:
        """
        Get capabilities provided by this extension.
        
        Override to return custom capabilities.
        """
        return []
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get extension metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "dependencies": self.dependencies,
            "initialized": self._initialized,
        }
    
    def __repr__(self) -> str:
        return f"<Extension: {self.name} v{self.version}>"
