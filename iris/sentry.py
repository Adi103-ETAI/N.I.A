"""IRIS Sentry Module - Background Screen Monitoring.

Uses Windows Native OCR to passively monitor screen for errors.
Runs in a background thread and alerts when danger keywords are detected.

Usage:
    from iris.sentry import start_sentry, stop_sentry
    
    # Start monitoring in background
    start_sentry(callback=lambda msg: print(msg))
    
    # Stop monitoring
    stop_sentry()
"""
from __future__ import annotations

import asyncio
import io
import logging
import threading
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try imports
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

# Windows OCR imports
try:
    from winsdk.windows.media.ocr import OcrEngine
    from winsdk.windows.graphics.imaging import (
        BitmapDecoder,
        SoftwareBitmap,
    )
    from winsdk.windows.storage.streams import (
        DataWriter,
        InMemoryRandomAccessStream,
    )
    _HAS_WINSDK = True
except ImportError:
    _HAS_WINSDK = False
    OcrEngine = None  # type: ignore
    logger.warning("winsdk not available - Sentry mode disabled")


# =============================================================================
# Configuration
# =============================================================================

# Keywords that trigger alerts (case-insensitive)
# Note: DANGER monitoring removed to prevent visual feedback loops
TRIGGERS = {
    "SECURITY": [
        "password", "credential", "unauthorized", "breach",
        "2fa", "verification code", "otp", "authentication",
    ],
    "COMMS": [
        "new message", "notification", "incoming call",
        "unread", "missed call",
    ],
}

# Anti-loop filter - ignore our own output
IGNORE_PATTERNS = [
    "sentry", "routing to iris", "nia:",
]

# Scan interval in seconds
SCAN_INTERVAL = 8

# Screen capture region (None = full screen)
CAPTURE_REGION = None


# =============================================================================
# Windows OCR Functions
# =============================================================================

async def _image_to_software_bitmap(pil_image: "Image.Image") -> Optional["SoftwareBitmap"]:
    """Convert PIL Image to Windows SoftwareBitmap for OCR.
    
    Args:
        pil_image: PIL Image in RGB format.
        
    Returns:
        SoftwareBitmap ready for OCR, or None on failure.
    """
    if not _HAS_WINSDK:
        return None
    
    try:
        # Convert to BGRA format (Windows expects BGRA)
        if pil_image.mode != "RGBA":
            pil_image = pil_image.convert("RGBA")
        
        # Save to PNG in memory
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        
        # Create Windows stream
        stream = InMemoryRandomAccessStream()
        writer = DataWriter(stream)
        writer.write_bytes(png_bytes)
        await writer.store_async()
        await writer.flush_async()
        stream.seek(0)
        
        # Decode to SoftwareBitmap
        decoder = await BitmapDecoder.create_async(stream)
        software_bitmap = await decoder.get_software_bitmap_async()
        
        return software_bitmap
        
    except Exception as exc:
        logger.debug("Failed to convert image: %s", exc)
        return None


async def _run_ocr(software_bitmap: "SoftwareBitmap") -> str:
    """Run Windows OCR on a SoftwareBitmap.
    
    Args:
        software_bitmap: Windows SoftwareBitmap.
        
    Returns:
        Extracted text from the image.
    """
    if not _HAS_WINSDK:
        return ""
    
    try:
        # Get OCR engine from user's language settings
        engine = OcrEngine.try_create_from_user_profile_languages()
        if not engine:
            logger.warning("No OCR engine available for user languages")
            return ""
        
        # Run OCR
        result = await engine.recognize_async(software_bitmap)
        
        # Extract text from all lines
        text_lines = []
        for line in result.lines:
            text_lines.append(line.text)
        
        return "\n".join(text_lines)
        
    except Exception as exc:
        logger.debug("OCR failed: %s", exc)
        return ""


async def scan_screen_for_text() -> str:
    """Capture screen and extract text using Windows OCR.
    
    Returns:
        Text found on screen.
    """
    if not _HAS_MSS or not _HAS_PIL or not _HAS_WINSDK:
        return ""
    
    try:
        with mss.mss() as sct:
            # Capture primary monitor
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            
            # Convert to PIL (smaller size for faster OCR)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
            # Resize to reduce OCR time (max 1280px wide)
            max_width = 1280
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to Windows bitmap and run OCR
            software_bitmap = await _image_to_software_bitmap(img)
            if software_bitmap:
                text = await _run_ocr(software_bitmap)
                
                # DEBUG: Print what OCR reads
                if DEBUG_MODE and text.strip():
                    preview = text.replace('\n', ' ')[:80]
                    print(f"[DEBUG OCR] Read: {preview}...")
                
                return text
            
        return ""
        
    except Exception as exc:
        logger.debug("Screen scan failed: %s", exc)
        return ""


def check_for_triggers(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Check text for any trigger keywords.
    
    Args:
        text: Text to scan.
        
    Returns:
        Tuple of (alert_type, keyword) or (None, None) if no match.
    """
    if not text:
        return None, None
    
    text_lower = text.lower()
    
    for alert_type, keywords in TRIGGERS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return alert_type, keyword
    
    return None, None


# =============================================================================
# Sentry Background Thread
# =============================================================================

class SentryThread(threading.Thread):
    """Background thread for Security & Comms monitoring."""
    
    def __init__(
        self,
        callback: Callable[[str, str], None],
        interval: float = SCAN_INTERVAL,
        keywords: List[str] = None,
    ) -> None:
        """Initialize sentry thread.
        
        Args:
            callback: Function(alert_type, keyword) to call on detection.
            interval: Seconds between scans.
            keywords: Optional custom keywords (uses TRIGGERS if None).
        """
        super().__init__(daemon=True, name="IRIS-Sentry")
        self.callback = callback
        self.interval = interval
        self.keywords = keywords or []
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def stop(self) -> None:
        """Stop the sentry thread."""
        self._stop_event.set()
    
    def run(self) -> None:
        """Main sentry loop - monitors for SECURITY and COMMS events."""
        print("👁️", end="", flush=True)
        logger.info("👁️ ✅ Sentry: ONLINE (Monitoring...)")
        
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        last_alert_keyword = ""
        
        try:
            while not self._stop_event.is_set():
                try:
                    text = self._loop.run_until_complete(scan_screen_for_text())
                    
                    if text:
                        text_lower = text.lower()
                        
                        # Anti-loop filter
                        if any(pattern in text_lower for pattern in IGNORE_PATTERNS):
                            self._stop_event.wait(self.interval)
                            continue
                        
                        # CHECK SECURITY (passwords, 2FA, etc.)
                        found_security = next(
                            (kw for kw in TRIGGERS.get("SECURITY", []) if kw in text_lower),
                            None
                        )
                        if found_security and found_security != last_alert_keyword:
                            last_alert_keyword = found_security
                            if self.callback:
                                try:
                                    self.callback("SECURITY", found_security)
                                except Exception:
                                    pass
                            self._stop_event.wait(10)
                            continue
                        
                        # CHECK COMMS (messages, notifications)
                        found_comms = next(
                            (kw for kw in TRIGGERS.get("COMMS", []) if kw in text_lower),
                            None
                        )
                        if found_comms and found_comms != last_alert_keyword:
                            last_alert_keyword = found_comms
                            if self.callback:
                                try:
                                    self.callback("COMMS", found_comms)
                                except Exception:
                                    pass
                            self._stop_event.wait(10)
                            continue
                        
                        # No alert - reset
                        if not found_security and not found_comms:
                            last_alert_keyword = ""
                            
                except Exception as exc:
                    logger.debug("Sentry scan error: %s", exc)
                
                self._stop_event.wait(self.interval)
                
        finally:
            self._loop.close()
            logger.info("👁️ Sentry thread stopped")


# =============================================================================
# Public API
# =============================================================================

_sentry_thread: Optional[SentryThread] = None


def start_sentry(
    callback: Callable[[str], None] = None,
    interval: float = SCAN_INTERVAL,
    keywords: List[str] = None,
) -> bool:
    """Start the background sentry.
    
    Args:
        callback: Function to call when danger is detected.
        interval: Seconds between scans.
        keywords: Keywords to watch for.
        
    Returns:
        True if started, False if already running or not available.
    """
    global _sentry_thread
    
    if not _HAS_WINSDK:
        logger.warning("Sentry not available (winsdk not installed)")
        return False
    
    if _sentry_thread and _sentry_thread.is_alive():
        logger.info("⚠️  Sentry already running")
        return _sentry_thread  # Return existing thread
    
    # Default callback
    if callback is None:
        callback = lambda msg: print(msg)
    
    _sentry_thread = SentryThread(
        callback=callback,
        interval=interval,
        keywords=keywords,
    )
    _sentry_thread.start()
    
    return _sentry_thread  # Return thread object for control


def stop_sentry() -> bool:
    """Stop the background sentry.
    
    Returns:
        True if stopped, False if not running.
    """
    global _sentry_thread
    
    if _sentry_thread and _sentry_thread.is_alive():
        _sentry_thread.stop()
        _sentry_thread.join(timeout=2)
        _sentry_thread = None
        print("👁️ ❌ Sentry: OFFLINE")
        return True
    
    return False


def is_sentry_running() -> bool:
    """Check if sentry is currently running."""
    return _sentry_thread is not None and _sentry_thread.is_alive()
