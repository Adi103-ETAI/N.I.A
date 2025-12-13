"""TTS plugin that uses `pyttsx3` when available, else falls back to stdout.

Provides a `speak` tool that accepts `{'text': '...'}` and returns
`{'ok': True}` when speech playback completes (or when fallback prints).

Supports subprocess mode via `__main__` which reads JSON from stdin and
prints JSON to stdout.
"""
from typing import Any, Dict


class TtsPyttsx3:
    name = "speak"
    description = "TTS via pyttsx3 (falls back to stdout)"

    def __init__(self):
        try:
            import pyttsx3  # type: ignore

            self._engine = pyttsx3.init()
            self._has_engine = True
        except Exception:
            self._engine = None
            self._has_engine = False

    def run(self, **params) -> Dict[str, Any]:
        text = params.get("text") or params.get("message") or ""
        if not text:
            return {"ok": False, "error": "No text provided"}

        if self._has_engine:
            # blocking speak
            self._engine.say(text)
            self._engine.runAndWait()
            return {"ok": True}

        # Fallback: print to stderr so stdout remains pure JSON
        import sys
        print(f"[TTS fallback] {text}", file=sys.stderr)
        return {"ok": True, "fallback": True}


if __name__ == "__main__":
    import sys
    import json

    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    plugin = TtsPyttsx3()
    try:
        out = plugin.run(**payload)
    except Exception as exc:
        out = {"ok": False, "error": str(exc)}
    print(json.dumps(out))
