"""Agent tool wrappers — Sprint 3.

Exposes TARA and IRIS as callable async tool functions that return
SubagentResult, for use by the Coordinator in Sprint 4.
"""
from src.capabilities.agents.invoke_tara import invoke_tara
from src.capabilities.agents.invoke_iris import invoke_iris

__all__ = ["invoke_tara", "invoke_iris"]
