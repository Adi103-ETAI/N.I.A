"""ASR plugin using `speech_recognition` where available, else a typed-input fallback.

Provides a `listen` tool that captures audio and returns `{'text': ...}`.
If running in subprocess mode and `speech_recognition`/`pyaudio` are not available,
it will read from stdin for compatibility.
"""
from typing import Any, Dict


class AsrSpeechRecog:
    name = "listen"
    description = "ASR via speech_recognition (microphone) with stdin fallback"

    def __init__(self):
        try:
            import speech_recognition as sr  # type: ignore
            self._sr = sr
            self._has_sr = True
        except Exception:
            self._sr = None
            self._has_sr = False

    def run(self, **params) -> Dict[str, Any]:
        # If a typed param is provided, prefer that (non-interactive/test-friendly)
        if params.get("typed"):
            return {"ok": True, "text": params.get("typed")}

        if self._has_sr and self._sr:
            r = self._sr.Recognizer()
            with self._sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                return {"ok": True, "text": text}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        # fallback: use `typed` parameter if provided, else request user input
        if params.get("typed"):
            typed = params.get("typed")
        else:
            typed = input("[ASR fallback] Type your input: ")
        return {"ok": True, "text": typed}


if __name__ == "__main__":
    import sys
    import json

    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    plugin = AsrSpeechRecog()
    try:
        out = plugin.run(**payload)
    except Exception as exc:
        out = {"ok": False, "error": str(exc)}
    print(json.dumps(out))
