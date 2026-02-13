"""IRIS Vision Tools - Hardware Layer for Image Capture.

Handles raw image capture from screen and webcam.
Does NOT include analysis, OCR, or face recognition - just capture and save.

Usage:
    from src.agents.iris.capture import capture_screen, capture_webcam
    
    # Take a screenshot
    path = capture_screen(delay=1.0)
    
    # Capture from webcam
    path = capture_webcam()
"""
from __future__ import annotations

import os
import time
from datetime import datetime

from src.agents.tara.protocols import tara_tool

# Optional imports with graceful fallback
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    cv2 = None  # type: ignore

try:
    import pyautogui
    _HAS_PYAUTOGUI = True
except (ImportError, KeyError, Exception):
    # KeyError: 'DISPLAY' can happen on headless systems
    _HAS_PYAUTOGUI = False
    pyautogui = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

VISION_CACHE = "data/vision_cache"

# Ensure cache directory exists
os.makedirs(VISION_CACHE, exist_ok=True)


# =============================================================================
# Helper Functions
# =============================================================================

def _generate_filename(prefix: str, extension: str) -> str:
    """Generate a timestamped filename.
    
    Args:
        prefix: Filename prefix (e.g., 'screen', 'webcam').
        extension: File extension (e.g., 'png', 'jpg').
        
    Returns:
        Absolute path for the new file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.{extension}"
    return os.path.abspath(os.path.join(VISION_CACHE, filename))


# =============================================================================
# Tool 1: Screen Capture
# =============================================================================

@tara_tool(
    name="capture_screen",
    category="vision",
    description="Take a screenshot of the entire screen and save it. Returns the file path."
)
def capture_screen(delay: float = 0.5, **kwargs) -> str:
    """Capture the screen and save as PNG.
    
    Args:
        delay: Seconds to wait before capturing (default 0.5).
        **kwargs: Ignored (catches LLM hallucinated args).
        
    Returns:
        Absolute path to the saved screenshot.
    """
    if not _HAS_PYAUTOGUI:
        return "Error: pyautogui not installed. Run: uv add pyautogui"
    
    try:
        # Wait for the specified delay
        if delay > 0:
            time.sleep(delay)
        
        # Take screenshot
        screenshot = pyautogui.screenshot()
        
        # Generate filename and save
        filepath = _generate_filename("screen", "png")
        screenshot.save(filepath)
        
        return filepath
        
    except Exception as e:
        return f"Error capturing screen: {e}"


# =============================================================================
# Tool 2: Webcam Capture
# =============================================================================

@tara_tool(
    name="capture_webcam",
    category="vision",
    description="Capture a single frame from the webcam and save it. Returns the file path."
)
def capture_webcam(**kwargs) -> str:
    """Capture a single frame from the webcam.
    
    Args:
        **kwargs: Ignored (catches LLM hallucinated args).
        
    Returns:
        Absolute path to the saved image, or error message.
    """
    if not _HAS_CV2:
        return "Error: OpenCV not installed. Run: uv add opencv-python"
    
    cap = None
    try:
        # Try to open the default camera (index 0)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows
        
        if not cap.isOpened():
            return "Error: Camera unavailable. It may be in use by Sentry or another app."
        
        # Allow camera to warm up
        time.sleep(0.3)
        
        # Read a single frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            return "Error: Failed to capture frame from webcam."
        
        # Generate filename and save
        filepath = _generate_filename("webcam", "jpg")
        cv2.imwrite(filepath, frame)
        
        return filepath
        
    except Exception as e:
        return f"Error capturing webcam: {e}"
        
    finally:
        # Always release the camera
        if cap is not None:
            cap.release()


# =============================================================================
# Tool 3: Cache Cleanup
# =============================================================================

@tara_tool(
    name="cleanup_vision_cache",
    category="vision",
    description="Delete all cached images from the vision cache folder."
)
def cleanup_vision_cache(**kwargs) -> str:
    """Delete all files in the vision cache directory.
    
    Args:
        **kwargs: Ignored (catches LLM hallucinated args).
        
    Returns:
        Status message with count of deleted files.
    """
    try:
        if not os.path.exists(VISION_CACHE):
            return "Vision cache directory does not exist."
        
        deleted_count = 0
        errors = []
        
        for filename in os.listdir(VISION_CACHE):
            filepath = os.path.join(VISION_CACHE, filename)
            
            # Only delete files, not subdirectories
            if os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except OSError as e:
                    errors.append(f"{filename}: {e}")
        
        if errors:
            return f"Deleted {deleted_count} files. Errors: {', '.join(errors)}"
        
        return f"✅ Deleted {deleted_count} files from vision cache."
        
    except Exception as e:
        return f"Error cleaning cache: {e}"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "capture_screen",
    "capture_webcam",
    "cleanup_vision_cache",
    "VISION_CACHE",
]
