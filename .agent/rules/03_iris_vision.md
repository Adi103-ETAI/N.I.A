# 👁️ IRIS VISION SECURITY MANDATE

## 🎯 EXECUTION CONTEXT

IRIS handles sensitive visual data (screenshots, webcam). Privacy violations are **FATAL ERRORS**.

---

## 💀 FATAL ERRORS (Immediate Stop)

If you attempt any of the following, the task is considered **FAILED**:

1. **Disk Leak:** Saving screenshots/images to disk without checking `config.DEBUG_MODE`.
   * **CORRECT ACTION:** Hold images in `BytesIO` memory buffer, process, then discard.

2. **Cloud Vision:** Sending raw image bytes to external APIs without Ghost Mode check.
   * **CORRECT ACTION:** Check `config.GHOST_MODE` before ANY external transmission.

3. **Hardcoded Paths:** Using `C:\Users\...` or absolute paths for image storage.
   * **CORRECT ACTION:** Use `tempfile.TemporaryDirectory()` if disk is absolutely required.

---

## 🛡️ THE "TRANSIENT IMAGE" TEMPLATE

All image processing MUST follow this exact pattern:

```python
from io import BytesIO
import base64

def process_image(self, capture_func: Callable) -> str:
    """
    Capture and analyze image without disk persistence.
    """
    try:
        # 1. Ghost Mode Check
        if self.config.GHOST_MODE:
            return "Error: Vision disabled in Ghost Mode."
        
        # 2. Capture to Memory (NOT disk)
        buffer = BytesIO()
        image = capture_func()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        
        # 3. Encode for LLM
        b64_image = base64.b64encode(buffer.read()).decode("utf-8")
        
        # 4. Explicit Cleanup
        buffer.close()
        del image
        
        # 5. Send to Vision LLM
        return self._analyze_with_llm(b64_image)
        
    except Exception as e:
        self.logger.exception(f"Vision failed: {e}")
        return f"Error: Vision processing failed - {str(e)}"
```

---

## 🚫 STRICT PROHIBITIONS

- **NO `PIL.Image.save()` to disk:** Unless `DEBUG_MODE=True` AND path is in `tempfile`.

- **NO Webcam without consent:** Webcam capture requires explicit user trigger (not auto-detect).

- **NO Image caching:** Do not store Base64 strings in memory managers or state dicts.

---

## 📝 COMMITMENT PROTOCOL

Before generating ANY IRIS code, you must state:

> "I have verified this code against 03_iris_vision.md and it contains NO disk leaks or privacy violations."
