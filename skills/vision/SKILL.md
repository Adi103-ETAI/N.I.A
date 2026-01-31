---
name: Vision
description: Screen and webcam capture for visual analysis and monitoring
tools: ["capture_screen", "capture_webcam", "cleanup_vision_cache"]
platforms: ["windows", "linux", "darwin"]
---
To capture visual information:

**Screen Capture:**
- `capture_screen()` - Take screenshot of entire screen
- `capture_screen(delay=2.0)` - Wait 2 seconds before capture
- Returns: Absolute path to saved PNG file

**Webcam Capture:**
- `capture_webcam()` - Capture single frame from webcam
- Returns: Absolute path to saved JPG file

> [!NOTE]
> Webcam may fail if already in use by Sentry or another app.

**Cache Management:**
- `cleanup_vision_cache()` - Delete all cached images
- Use periodically to free disk space

**Use Cases:**
1. **UI Debugging** - Capture screen to see current state
2. **Coordinate Finding** - Screenshot → analyze → get click positions
3. **Visual Verification** - Confirm an action completed correctly
4. **Face/Object Detection** - Capture webcam for analysis

**Workflow Example:**
1. `capture_screen()` → Get screenshot path
2. Analyze image to find button location
3. Use `mouse_click(x, y)` to interact

> [!TIP]
> Images are saved to `data/vision_cache/` with timestamps.
