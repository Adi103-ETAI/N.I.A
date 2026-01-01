"""N.I.A. Terminal UI Module - Abstracted CLI Interface.

This module encapsulates all terminal user interface logic including:
- PromptToolkit session management
- Styling and formatting
- Clean stdout context for background threads

Usage:
    from interface.chat import TerminalUI
    
    ui = TerminalUI()
    with ui.context():
        user_input = ui.get_input("You: ")
        ui.print("Response here")
"""
from __future__ import annotations

from typing import Any, Optional


# =============================================================================
# PromptToolkit Imports (with graceful fallback)
# =============================================================================

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.styles import Style
    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    _HAS_PROMPT_TOOLKIT = False
    PromptSession = None  # type: ignore
    patch_stdout = None  # type: ignore
    Style = None  # type: ignore


# =============================================================================
# Null Context Manager (fallback when prompt_toolkit unavailable)
# =============================================================================

class _NullContext:
    """Minimal context manager that does nothing."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


# =============================================================================
# Terminal UI Class
# =============================================================================

class TerminalUI:
    """Encapsulates all terminal UI logic for NIA.
    
    Provides a clean abstraction over prompt_toolkit with fallbacks
    for systems where it's not available.
    
    Example:
        ui = TerminalUI()
        with ui.context():
            while True:
                user_input = ui.get_input("You: ")
                ui.print(f"Echo: {user_input}")
    """
    
    def __init__(
        self,
        prompt_style: str = "ansigreen bold",
        enable_history: bool = True,
    ) -> None:
        """Initialize the terminal UI.
        
        Args:
            prompt_style: Style for the input prompt (prompt_toolkit format).
            enable_history: Enable command history navigation.
        """
        self._has_toolkit = _HAS_PROMPT_TOOLKIT
        self._session: Optional[Any] = None
        self._style: Optional[Any] = None
        
        if self._has_toolkit:
            self._session = PromptSession()
            self._style = Style.from_dict({'prompt': prompt_style})
    
    def get_input(self, prompt: str = "You: ") -> str:
        """Get user input with styled prompt.
        
        Args:
            prompt: Text to display as prompt.
            
        Returns:
            User's input string.
        """
        if self._has_toolkit and self._session:
            return self._session.prompt(
                [('class:prompt', prompt)],
                style=self._style,
            )
        else:
            return input(prompt)
    
    def context(self):
        """Get context manager for clean stdout handling.
        
        Use this to wrap your main loop so background thread output
        doesn't interfere with user input.
        
        Returns:
            Context manager (patch_stdout or null context).
        """
        if self._has_toolkit and patch_stdout:
            return patch_stdout()
        return _NullContext()
    
    def print(self, text: str, **kwargs) -> None:
        """Print text to terminal (wrapper for future rich text support).
        
        Args:
            text: Text to print.
            **kwargs: Additional print arguments.
        """
        print(text, **kwargs)
    
    def print_banner(self, text: str) -> None:
        """Print a banner/header text.
        
        Args:
            text: Banner content.
        """
        print(text)
    
    @property
    def is_available(self) -> bool:
        """Check if prompt_toolkit is available."""
        return self._has_toolkit


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TerminalUI",
]
