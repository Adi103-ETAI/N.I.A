"""Service Registry - Universal Socket for N.I.A. Components.

Acts as a central dictionary for loose coupling between the Core Engine
and its peripherals (Voice, Vision, Tools).

Pattern:
    - Main.py acts as the "Assembler", instantiating services and registering them.
    - Core Engine acts as the "Consumer", asking the registry for services.

Usage:
    from core.services import ServiceRegistry
    
    # Registration
    ServiceRegistry.register("voice", nola_manager)
    
    # Consumption
    voice = ServiceRegistry.get("voice")
    if voice:
        voice.speak("Hello")
"""
from typing import Any, Dict, Optional
from core.logger import setup_logger

logger = setup_logger("REGISTRY")

class ServiceRegistry:
    """Central service registry for dependency decoupling."""
    
    _services: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, name: str, service: Any) -> None:
        """Register a service instance."""
        cls._services[name] = service
        logger.debug(f"Service registered: '{name}' -> {type(service).__name__}")
        
    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        """Get a service instance by name."""
        return cls._services.get(name)
        
    @classmethod
    def list_services(cls) -> Dict[str, str]:
        """List all registered services."""
        return {k: type(v).__name__ for k, v in cls._services.items()}
    
    @classmethod
    def clear(cls) -> None:
        """Clear all services (for testing/shutdown)."""
        cls._services.clear()
        logger.debug("Registry cleared")
