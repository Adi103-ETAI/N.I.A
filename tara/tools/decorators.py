"""
TARA Tool Decorators.

Provides metadata tagging for TARA tools, specifically for security capability gating.
"""
from typing import Callable
import functools

def security_level(level: str) -> Callable:
    """
    Decorator to tag a tool with a security classification.
    
    Levels:
    - "read_only": Safe, idempotent operations (e.g., read_file, list_dir).
    - "host_standard": Default. Standard local execution (e.g., click, type).
    - "high_risk": Dangerous. Requires sandboxing or explicit approval (e.g., shell, delete).
    
    Usage:
        @security_level("high_risk")
        def delete_file(path): ...
    """
    valid_levels = {"read_only", "host_standard", "high_risk"}
    if level not in valid_levels:
        # Default to standard if invalid, but log warning in a real system
        level = "host_standard"
        
    def decorator(func: Callable) -> Callable:
        # Attach attribute to the function object itself
        setattr(func, "_security_level", level)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
