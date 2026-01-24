# ⛔ N.I.A. GLOBAL HARD RULES (DO NOT IGNORE)

## 💀 FATAL ERRORS (Immediate Stop)

If you attempt any of the following, the task is considered **FAILED**:

1. **Cloud Leak:** Importing `openai`, `google.cloud`, or `elevenlabs` directly. *ONLY* access these via `models.model_manager` or `nola.io`.

2. **Hard Deletion:** Using `os.remove()`, `shutil.rmtree()`, or `pathlib.Path.unlink()`.
   * **CORRECT ACTION:** You MUST use `tara.tools.file_ops.safe_delete` (Recycle Bin).

3. **Split-Brain:** Instantiating `MemoryManager()` or `NOLAManager()` manually.
   * **CORRECT ACTION:** Get them from `core.container.ServiceContainer`.

---

## 🔒 SECURITY ENFORCEMENT

- **Input Sanitization:** NEVER pass raw user input directly to a shell command (`subprocess.run`). Always sanitize via `shlex.quote`.

- **Ghost Mode:** Before sending any data to an external API (even LLMs), check `config.GHOST_MODE`. If True, raise `SecurityException`.

---

## 📝 COMMITMENT PROTOCOL

Before generating ANY code, you must state:

> "I have verified this code against 00_global_protocol.md and it contains NO forbidden imports or hard deletions."
