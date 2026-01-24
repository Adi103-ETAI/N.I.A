"""Example Plugin - Demonstrates the plugin system.

Drop this file in the plugins/ directory to see it automatically loaded.
Delete or rename to disable.

To create your own plugin:
1. Create a .py file in plugins/
2. Define public functions with docstrings
3. Restart N.I.A. or call plugin reload

Example tool below will be auto-registered as "example_plugin__greet"
"""


def greet(name: str) -> str:
    """Greet someone by name.
    
    Args:
        name: The person's name to greet.
        
    Returns:
        A friendly greeting message.
    """
    return f"Hello, {name}! Welcome to N.I.A. plugins."


def add_numbers(a: int, b: int) -> str:
    """Add two numbers together.
    
    Args:
        a: First number.
        b: Second number.
        
    Returns:
        The sum as a string.
    """
    return f"The sum of {a} and {b} is {a + b}"


# Private functions (prefixed with _) are not exposed as tools
def _helper():
    """This won't be registered as a tool."""
    pass
