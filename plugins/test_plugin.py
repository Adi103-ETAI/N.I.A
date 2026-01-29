"""
PLUGIN: System Test Module
VERSION: 1.0
AUTHOR: Director
"""

import time

def test_hello_world(name: str = "Director") -> str:
    """
    A diagnostic tool to verify that the external Plugin System is working correctly.
    Use this tool when the user asks for a 'system test', 'plugin check', or 'hello world'.

    Args:
        name: The name of the user to greet (default: Director).

    Returns:
        A verification message with a timestamp proving the plugin code was executed.
    """
    timestamp = time.strftime("%H:%M:%S")
    return f"✅ [PLUGIN SUCCESS] Time: {timestamp} | Message: Hello {name}! The External Plugin System is FULLY OPERATIONAL and Hot-Loaded."