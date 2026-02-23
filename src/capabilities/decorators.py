"""Capabilities Security Decorators.

Provides the ``@security_level`` metadata decorator used to tag tool
functions with their risk classification.

Security Levels:
    ``"read_only"``
        Safe, idempotent operations — e.g. ``read_file``, ``list_dir``.
    ``"host_standard"``
        Default. Normal local execution — e.g. ``click``, ``type_text``.
    ``"high_risk"``
        Dangerous — e.g. ``delete_file``, shell commands.
        Requires Warden approval or sandboxed execution.

Usage::

    from src.capabilities.decorators import security_level

    @security_level("high_risk")
    def delete_file(path: str) -> str:
        ...
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
        
    import inspect
    
    def decorator(func: Callable) -> Callable:
        # Attach attribute to the function object itself
        # Note: We must attach it to the WRAPPER eventually, but we do it here
        # so it persists through inspection if needed.
        setattr(func, "_security_level", level)
        
        # Check if the wrapped function is a coroutine
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
        
        # Ensure the wrapper also carries the tag
        setattr(wrapper, "_security_level", level)
        return wrapper
    return decorator
