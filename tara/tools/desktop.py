"""Desktop automation tools for TARA.

Provides app control and clipboard management.
"""
from __future__ import annotations

from typing import Any, Type

# Try to import dependencies
try:
    from AppOpener import open as app_open, close as app_close
    _HAS_APPOPENER = True
except ImportError:
    _HAS_APPOPENER = False
    app_open = None  # type: ignore
    app_close = None  # type: ignore

try:
    import pyperclip
    _HAS_PYPERCLIP = True
except ImportError:
    _HAS_PYPERCLIP = False
    pyperclip = None  # type: ignore

# Try to import LangChain tools
try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    BaseTool = object  # type: ignore
    BaseModel = object  # type: ignore


# =============================================================================
# App Control Tools
# =============================================================================

class OpenAppInput(BaseModel):
    """Input for OpenAppTool."""
    app_name: str = Field(description="Name of the application to open (e.g., 'chrome', 'notepad', 'brave')")


class OpenAppTool(BaseTool):
    """Tool to open an application on the desktop."""
    
    name: str = "open_app"
    description: str = (
        "Open an application on the desktop. "
        "Examples: 'chrome', 'notepad', 'brave', 'spotify', 'discord'. "
        "For browsers, prefer 'brave' as the default."
    )
    args_schema: Type[BaseModel] = OpenAppInput
    
    def _run(self, app_name: str) -> str:
        """Open the specified application."""
        if not _HAS_APPOPENER:
            return "Error: AppOpener not installed. Run: pip install AppOpener"
        
        try:
            app_open(app_name, match_closest=True, output=False)
            return f"Opened {app_name}"
        except Exception as exc:
            return f"Error opening {app_name}: {exc}"
    
    async def _arun(self, app_name: str) -> str:
        """Async version - just calls sync."""
        return self._run(app_name)


class CloseAppInput(BaseModel):
    """Input for CloseAppTool."""
    app_name: str = Field(description="Name of the application to close")


class CloseAppTool(BaseTool):
    """Tool to close an application on the desktop."""
    
    name: str = "close_app"
    description: str = (
        "Close an application that is currently running. "
        "Examples: 'chrome', 'notepad', 'spotify'."
    )
    args_schema: Type[BaseModel] = CloseAppInput
    
    def _run(self, app_name: str) -> str:
        """Close the specified application."""
        if not _HAS_APPOPENER:
            return "Error: AppOpener not installed. Run: pip install AppOpener"
        
        try:
            app_close(app_name, match_closest=True, output=False)
            return f"Closed {app_name}"
        except Exception as exc:
            return f"Error closing {app_name}: {exc}"
    
    async def _arun(self, app_name: str) -> str:
        """Async version - just calls sync."""
        return self._run(app_name)


# =============================================================================
# URL/Web Tools
# =============================================================================

class OpenURLInput(BaseModel):
    """Input for OpenURLTool."""
    url: str = Field(description="The URL or web link to open (e.g., 'https://youtube.com/watch?v=...')")


class OpenURLTool(BaseTool):
    """Tool to open a specific URL in the default web browser."""
    
    name: str = "open_url"
    description: str = (
        "Open a specific URL or website link in the default browser. "
        "Use this for opening YouTube videos, specific websites, or any web link. "
        "Examples: 'https://youtube.com', 'https://google.com/search?q=...'."
    )
    args_schema: Type[BaseModel] = OpenURLInput
    
    def _run(self, url: str) -> str:
        """Open the specified URL in the default browser."""
        import webbrowser
        
        try:
            # Ensure URL has a scheme
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            
            webbrowser.open(url)
            return f"Opened {url}"
        except Exception as exc:
            return f"Error opening URL: {exc}"
    
    async def _arun(self, url: str) -> str:
        """Async version - just calls sync."""
        return self._run(url)


class PlayYouTubeInput(BaseModel):
    """Input for PlayYouTubeTool."""
    query: str = Field(description="The song name, video title, or search query to play on YouTube")


class PlayYouTubeTool(BaseTool):
    """Tool to play a song or video on YouTube using pywhatkit.
    
    This tool searches YouTube and automatically plays the first result.
    """
    
    name: str = "play_youtube"
    description: str = (
        "Use this tool ONLY when the user asks to 'play', 'stream', or 'listen to' "
        "a specific song or video on YouTube. Input should be the song name or search query. "
        "Examples: 'Bohemian Rhapsody', 'Shape of You Ed Sheeran', 'funny cat videos'."
    )
    args_schema: Type[BaseModel] = PlayYouTubeInput
    
    def _run(self, query: str) -> str:
        """Play the specified song/video on YouTube."""
        try:
            import pywhatkit
            pywhatkit.playonyt(query)
            return f"Playing '{query}' on YouTube"
        except ImportError:
            return "Error: pywhatkit not installed. Run: pip install pywhatkit"
        except Exception as exc:
            return f"Error playing on YouTube: {exc}"
    
    async def _arun(self, query: str) -> str:
        """Async version - just calls sync."""
        return self._run(query)


# =============================================================================
# Clipboard Tools
# =============================================================================

class CopyInput(BaseModel):
    """Input for CopyTool."""
    text: str = Field(description="Text to copy to the clipboard")


class CopyTool(BaseTool):
    """Tool to copy text to the system clipboard."""
    
    name: str = "copy_to_clipboard"
    description: str = (
        "Copy text to the system clipboard. "
        "The user can then paste it anywhere with Ctrl+V."
    )
    args_schema: Type[BaseModel] = CopyInput
    
    def _run(self, text: str) -> str:
        """Copy text to clipboard."""
        if not _HAS_PYPERCLIP:
            return "Error: pyperclip not installed. Run: pip install pyperclip"
        
        try:
            pyperclip.copy(text)
            return "Copied to clipboard."
        except Exception as exc:
            return f"Error copying to clipboard: {exc}"
    
    async def _arun(self, text: str) -> str:
        """Async version - just calls sync."""
        return self._run(text)


class PasteInput(BaseModel):
    """Input for PasteTool - no input required."""
    pass


class PasteTool(BaseTool):
    """Tool to read text from the system clipboard."""
    
    name: str = "paste_from_clipboard"
    description: str = (
        "Read and return the current contents of the system clipboard. "
        "Use when the user asks what's in the clipboard or wants to use clipboard content."
    )
    args_schema: Type[BaseModel] = PasteInput
    
    def _run(self, **kwargs: Any) -> str:
        """Get clipboard contents."""
        if not _HAS_PYPERCLIP:
            return "Error: pyperclip not installed. Run: pip install pyperclip"
        
        try:
            content = pyperclip.paste()
            if content:
                return f"Clipboard contains: {content}"
            else:
                return "Clipboard is empty."
        except Exception as exc:
            return f"Error reading clipboard: {exc}"
    
    async def _arun(self, **kwargs: Any) -> str:
        """Async version - just calls sync."""
        return self._run(**kwargs)
