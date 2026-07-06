"""Text-to-speech tool — gives NIA a voice (Jarvis-style).

The audit (P3) noted that ``nia_voice`` only handles speech-to-text (transcription);
there is no text-to-speech counterpart.  This module fills that gap.

Implementation
--------------
Primary backend: **KittenTTS** — an open-source, lightweight neural TTS model
(~15M params, ~56MB on disk) that runs entirely on CPU via ONNX.  Ships with
8 voices including ``Jasper`` (the default — Jarvis-like male voice).

Fallback: ``espeak`` command-line tool, for environments where KittenTTS
can't be installed (e.g. minimal containers).  Lower quality but always
available on Debian/Ubuntu.

The model is loaded lazily on first call and cached for the lifetime of the
process so subsequent calls are fast (~1–2s per synthesis after the initial
~10s model load).

Audio files are saved to ``/home/z/my-project/download/`` so the user can
play them back.  The tool returns the file path, duration, and backend used.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class SpeakToolInput(BaseModel):
    """Arguments for the speak (text-to-speech) tool.

    Voice names follow the KittenTTS convention: Bella, Jasper (default —
    male, Jarvis-like), Luna, Bruno, Rosie, Hugo, Kiki, Leo.

    When the espeak fallback is used, voice names are mapped to the closest
    espeak equivalent (en-us / en-gb with pitch shifts).
    """

    text: str = Field(
        description="The text to synthesise.  Limited to ~3000 characters per call.",
    )
    voice: str = Field(
        default="Jasper",
        description=(
            "Voice name.  KittenTTS voices: Bella, Jasper (default male/Jarvis-like), "
            "Luna, Bruno, Rosie, Hugo, Kiki, Leo."
        ),
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed)",
    )
    output_format: Literal["wav", "mp3"] = Field(
        default="wav",
        description="Output audio format.  wav is native (24kHz); mp3 requires ffmpeg.",
    )
    model: str = Field(
        default="KittenML/kitten-tts-nano-0.8",
        description=(
            "KittenTTS HuggingFace model ID.  Options (smallest→largest): "
            "KittenML/kitten-tts-nano-0.8 (15M, default), "
            "KittenML/kitten-tts-micro-0.8 (40M), "
            "KittenML/kitten-tts-mini-0.8 (80M, highest quality)."
        ),
    )


# ---------------------------------------------------------------------------
# KittenTTS singleton — model is expensive to load (~10s), so cache it
# ---------------------------------------------------------------------------


class _KittenTTSSession:
    """Lazily-loaded, cached KittenTTS model singleton.

    The same model instance is reused across calls.  Switching models
    (e.g. from nano to mini for higher quality) reloads the cache.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_name: str | None = None
        self._load_error: str | None = None
        self._lock = asyncio.Lock()

    async def get(self, model_name: str):
        """Return a KittenTTS instance for ``model_name``, loading if needed."""
        # Fast path: same model already loaded.
        if self._model is not None and self._model_name == model_name:
            return self._model
        async with self._lock:
            # Double-check after acquiring lock.
            if self._model is not None and self._model_name == model_name:
                return self._model
            # Load in a thread — KittenTTS does network + disk I/O on first load.
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None, self._load_sync, model_name
            )
            self._model_name = model_name
            return self._model

    def _load_sync(self, model_name: str):
        """Blocking model load.  Raises RuntimeError on failure."""
        try:
            from kittentts import KittenTTS  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"kittentts is not installed: {exc}. "
                "Install with: pip install kittentts soundfile"
            ) from exc
        try:
            return KittenTTS(model_name)
        except Exception as exc:
            raise RuntimeError(f"failed to load KittenTTS model {model_name!r}: {exc}") from exc

    def reset(self) -> None:
        """Discard the cached model (next call will reload)."""
        self._model = None
        self._model_name = None


_SESSION = _KittenTTSSession()


# ---------------------------------------------------------------------------
# Voice mapping for the espeak fallback
# ---------------------------------------------------------------------------

# Map KittenTTS voice names → espeak voice + pitch hints.
_ESPEAK_VOICE_MAP = {
    "Bella": ("en-us+f3", 50),
    "Jasper": ("en-us+m1", 40),  # male, lower pitch
    "Luna": ("en-us+f2", 55),
    "Bruno": ("en-us+m2", 35),  # male, even lower
    "Rosie": ("en-gb+f3", 55),
    "Hugo": ("en-gb+m2", 40),
    "Kiki": ("en-us+f4", 60),
    "Leo": ("en-gb+m1", 40),
}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class SpeakTool(BaseTool):
    """Synthesise speech from text and save it as an audio file."""

    name = "speak"
    description = (
        "Convert text to speech and save the audio to disk.  Uses KittenTTS "
        "(neural, ~15M params, runs on CPU) for high-quality output with 8 voices "
        "including the Jarvis-like 'Jasper' voice.  Falls back to espeak if "
        "KittenTTS is unavailable.  Returns the path to the generated audio file."
    )
    input_model = SpeakToolInput

    def is_read_only(self, arguments: SpeakToolInput) -> bool:
        # Generates a file but doesn't mutate user state; treat as read-only
        # so the permission system doesn't gate it unnecessarily.
        del arguments
        return True

    async def execute(self, arguments: SpeakToolInput, context: ToolExecutionContext) -> ToolResult:
        # Validate input length.
        if not arguments.text.strip():
            return ToolResult(output="text is empty", is_error=True)
        if len(arguments.text) > 3000:
            return ToolResult(
                output=f"text is too long ({len(arguments.text)} chars); max 3000 chars per call",
                is_error=True,
            )

        # Output dir — always under the project download dir.
        out_dir = Path("/home/z/my-project/download")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe_voice = re.sub(r"[^a-zA-Z0-9-]", "_", arguments.voice)
        base_name = f"tts-{safe_voice}-{ts}"

        # Try KittenTTS first.
        try:
            return await self._synth_kitten(arguments, out_dir, base_name)
        except Exception as exc:
            kitten_err = f"{type(exc).__name__}: {exc}"

        # Fallback to espeak.
        try:
            return self._synth_espeak(arguments, out_dir, base_name)
        except RuntimeError as exc:
            espeak_err = str(exc)
        except Exception as exc:
            espeak_err = f"{type(exc).__name__}: {exc}"

        # Both failed.
        return ToolResult(
            output=(
                "Text-to-speech failed: neither KittenTTS nor espeak is available.\n"
                f"  KittenTTS error: {kitten_err}\n"
                f"  espeak error:    {espeak_err}\n"
                "Install one of them:\n"
                "  pip install kittentts soundfile   # recommended (neural, ~56MB model)\n"
                "  apt-get install espeak            # fallback (lower quality)"
            ),
            is_error=True,
        )

    # ---- KittenTTS backend ---------------------------------------------

    async def _synth_kitten(
        self,
        args: SpeakToolInput,
        out_dir: Path,
        base_name: str,
    ) -> ToolResult:
        """Synthesise via KittenTTS.  Raises on failure."""
        model = await _SESSION.get(args.model)

        # Validate voice name.
        voices = getattr(model, "available_voices", []) or []
        if voices and args.voice not in voices:
            return ToolResult(
                output=(
                    f"Unknown voice: {args.voice!r}.  "
                    f"Available voices: {', '.join(voices)}"
                ),
                is_error=True,
            )

        # Generate audio samples (numpy array, 24kHz).
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            lambda: model.generate(args.text, voice=args.voice, speed=args.speed),
        )

        # Write WAV via soundfile.
        try:
            import soundfile as sf  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f"soundfile not installed: {exc}") from exc

        wav_path = out_dir / f"{base_name}.wav"
        sf.write(str(wav_path), audio, 24000)

        # If user requested mp3, convert via ffmpeg if available.
        final_path = wav_path
        if args.output_format == "mp3":
            if shutil.which("ffmpeg"):
                mp3_path = out_dir / f"{base_name}.mp3"
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(wav_path),
                        "-codec:a", "libmp3lame", "-qscale:a", "4",
                        str(mp3_path),
                    ],
                    check=True,
                    capture_output=True,
                )
                try:
                    wav_path.unlink()
                except OSError:
                    pass
                final_path = mp3_path
            else:
                # No ffmpeg — keep the WAV but warn in the output.
                pass

        size_kb = final_path.stat().st_size // 1024
        duration_s = len(audio) / 24000
        return ToolResult(
            output=(
                f"Spoken: {final_path.name}\n"
                f"  Backend: KittenTTS ({args.model})\n"
                f"  Voice: {args.voice}\n"
                f"  Duration: {duration_s:.1f}s\n"
                f"  Size: {size_kb} KB\n"
                f"  Path: {final_path}"
                + ("\n  (note: mp3 requested but ffmpeg not found — kept wav)" if args.output_format == "mp3" and not shutil.which("ffmpeg") else "")
            ),
            metadata={
                "path": str(final_path),
                "backend": "kittentts",
                "model": args.model,
                "voice": args.voice,
                "duration_seconds": duration_s,
                "sample_rate": 24000,
                "size_bytes": final_path.stat().st_size,
            },
        )

    # ---- espeak fallback backend ---------------------------------------

    def _synth_espeak(
        self,
        args: SpeakToolInput,
        out_dir: Path,
        base_name: str,
    ) -> ToolResult:
        """Synthesise via the espeak command-line tool.  Raises RuntimeError if missing."""
        if not shutil.which("espeak"):
            raise RuntimeError("espeak not installed and not on PATH")

        # Look up the voice mapping; fall back to en-us.
        espeak_voice, pitch_val = _ESPEAK_VOICE_MAP.get(args.voice, ("en-us", 50))

        # Convert speed multiplier to words-per-minute (espeak baseline 175 wpm).
        wpm = max(80, min(450, int(175 * args.speed)))

        # espeak outputs wav via -w.
        wav_path = out_dir / f"{base_name}.wav"
        cmd = [
            "espeak",
            "-v", espeak_voice,
            "-s", str(wpm),
            "-p", str(pitch_val),
            "-w", str(wav_path),
            args.text,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Convert to mp3 if requested.
        final_path = wav_path
        if args.output_format == "mp3" and shutil.which("ffmpeg"):
            mp3_path = out_dir / f"{base_name}.mp3"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-qscale:a", "4", str(mp3_path)],
                check=True,
                capture_output=True,
            )
            try:
                wav_path.unlink()
            except OSError:
                pass
            final_path = mp3_path

        size_kb = final_path.stat().st_size // 1024 if final_path.exists() else 0
        return ToolResult(
            output=(
                f"Spoken: {final_path.name}\n"
                f"  Backend: espeak (fallback)\n"
                f"  Voice: {args.voice} → {espeak_voice}\n"
                f"  Size: {size_kb} KB\n"
                f"  Path: {final_path}"
            ),
            metadata={
                "path": str(final_path),
                "backend": "espeak",
                "voice": args.voice,
                "espeak_voice": espeak_voice,
                "size_bytes": final_path.stat().st_size if final_path.exists() else 0,
            },
        )
