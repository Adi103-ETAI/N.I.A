"""IRIS Vision Tools - Screen Capture for NIA.

Provides screen capture functionality using MSS and processes images for AI analysis.
"""
from __future__ import annotations

import base64
import io
import logging

logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import mss
    _HAS_MSS = True
except ImportError:
    _HAS_MSS = False
    mss = None  # type: ignore

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    Image = None  # type: ignore

try:
    from langchain_core.tools import tool
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    # Fallback decorator
    def tool(func):
        return func


def _process_image(sct_img) -> str:
    """Convert raw MSS capture to a Base64 encoded string optimized for AI.
    
    - Converts BGRA to RGB.
    - Resizes to max 1024x1024 to save bandwidth/tokens.
    - Compresses to PNG.
    
    Args:
        sct_img: MSS screenshot object.
        
    Returns:
        Base64 encoded PNG string.
    """
    if not _HAS_PIL:
        raise ImportError("PIL not installed. Run: pip install pillow")
    
    # MSS returns BGRA, PIL needs RGB
    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    
    # Resize if larger than 1024px to save tokens
    max_size = 1024
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Save to buffer as PNG
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@tool
def capture_screen() -> str:
    """Captures the primary monitor and returns a Base64 string.
    
    Use this tool when the user asks you to 'look', 'see', or 'analyze' the screen.
    Returns a Base64-encoded PNG image of the current screen.
    """
    if not _HAS_MSS:
        return "Error: mss not installed. Run: pip install mss"
    
    if not _HAS_PIL:
        return "Error: pillow not installed. Run: pip install pillow"
    
    try:
        with mss.mss() as sct:
            # Capture Primary Monitor (index 1)
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            return _process_image(sct_img)
    except Exception as exc:
        logger.exception("Screen capture failed: %s", exc)
        return f"Error capturing screen: {exc}"


def capture_screen_raw() -> str:
    """Non-decorated version for direct use in IRIS agent."""
    if not _HAS_MSS:
        raise ImportError("mss not installed. Run: pip install mss")
    
    if not _HAS_PIL:
        raise ImportError("pillow not installed. Run: pip install pillow")
    
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        return _process_image(sct_img)
