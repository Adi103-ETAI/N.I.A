"""TARA Desktop Control Unit - Application & Clipboard Tools.

Provides tools for:
- Opening/closing applications
- YouTube playback
- Clipboard operations

Dependencies:
    pip install AppOpener pyperclip pywhatkit
"""
from __future__ import annotations

from tara.protocols import tara_tool

# =============================================================================
# Optional Dependencies (graceful fallback)
# =============================================================================

# AppOpener for app control
try:
    from AppOpener import open as app_open, close as app_close
    _HAS_APPOPENER = True
except ImportError:
    _HAS_APPOPENER = False

# pyperclip for clipboard
try:
    import pyperclip
    _HAS_PYPERCLIP = True
except ImportError:
    _HAS_PYPERCLIP = False

# pywhatkit for YouTube
try:
    import pywhatkit
    _HAS_PYWHATKIT = True
except ImportError:
    _HAS_PYWHATKIT = False

# webbrowser as fallback for URLs
import webbrowser


# =============================================================================
# Application Control Tools
# =============================================================================

@tara_tool(
    name="open_app",
    category="desktop",
    description="Open an application on the desktop. Examples: 'brave', 'chrome', 'notepad', 'spotify', 'discord'."
)
def open_application(app_name: str) -> str:
    """Open an application by name.
    
    Args:
        app_name: Name of the application to open (e.g., 'brave', 'notepad').
    """
    if not _HAS_APPOPENER:
        return "Error: AppOpener not installed. Run: pip install AppOpener"
    
    try:
        app_open(app_name, match_closest=True, throw_error=True)
        return f"Opened {app_name}"
    except Exception as e:
        return f"Error opening {app_name}: {e}"


@tara_tool(
    name="close_app",
    category="desktop",
    description="Close a running application. Examples: 'chrome', 'notepad', 'spotify'."
)
def close_application(app_name: str) -> str:
    """Close a running application by name.
    
    Args:
        app_name: Name of the application to close.
    """
    if not _HAS_APPOPENER:
        return "Error: AppOpener not installed. Run: pip install AppOpener"
    
    try:
        app_close(app_name, match_closest=True, throw_error=True)
        return f"Closed {app_name}"
    except Exception as e:
        return f"Error closing {app_name}: {e}"


# =============================================================================
# Media Tools
# =============================================================================

@tara_tool(
    name="play_youtube",
    category="desktop",
    description="Play a song or video on YouTube. Searches and auto-plays the first result."
)
def play_youtube(query: str) -> str:
    """Search and play a YouTube video.
    
    Args:
        query: Song name, video title, or search query.
    """
    if not _HAS_PYWHATKIT:
        return "Error: pywhatkit not installed. Run: pip install pywhatkit"
    
    try:
        pywhatkit.playonyt(query)
        return f"Playing '{query}' on YouTube"
    except Exception as e:
        return f"Error playing on YouTube: {e}"


@tara_tool(
    name="open_url",
    category="desktop",
    description="Open a specific URL or website in the default browser."
)
def open_url(url: str) -> str:
    """Open a URL in the default browser.
    
    Args:
        url: The web URL to open (e.g., 'https://github.com').
    """
    try:
        # Add protocol if missing
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        webbrowser.open(url)
        return f"Opened {url}"
    except Exception as e:
        return f"Error opening URL: {e}"


# =============================================================================
# Clipboard Tools
# =============================================================================

@tara_tool(
    name="get_clipboard",
    category="desktop",
    description="Get the current contents of the system clipboard."
)
def get_clipboard_content() -> str:
    """Get text from the system clipboard."""
    if not _HAS_PYPERCLIP:
        return "Error: pyperclip not installed. Run: pip install pyperclip"
    
    try:
        content = pyperclip.paste()
        if content:
            # Truncate if too long
            if len(content) > 500:
                return f"Clipboard ({len(content)} chars): {content[:500]}..."
            return f"Clipboard: {content}"
        return "Clipboard is empty"
    except Exception as e:
        return f"Error reading clipboard: {e}"


@tara_tool(
    name="set_clipboard",
    category="desktop",
    description="Copy text to the system clipboard."
)
def set_clipboard_content(text: str) -> str:
    """Copy text to the system clipboard.
    
    Args:
        text: Text to copy to clipboard.
    """
    if not _HAS_PYPERCLIP:
        return "Error: pyperclip not installed. Run: pip install pyperclip"
    
    try:
        pyperclip.copy(text)
        return f"Copied {len(text)} characters to clipboard"
    except Exception as e:
        return f"Error copying to clipboard: {e}"
