"""Tool for N.I.A voice integration.

Provides voice input transcription and voice mode control,
connecting NIA to niaharness's voice system.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class NiaVoiceInput(BaseModel):
    """Arguments for NIA voice operations."""

    action: str = Field(
        description="Operation: 'transcribe' to transcribe audio from a file, "
        "'status' to check voice capabilities, "
        "'keyterms' to extract key terms from text"
    )
    audio_path: str | None = Field(default=None, description="Path to audio file (for transcribe)")
    text: str | None = Field(default=None, description="Text to extract keyterms from (for keyterms)")


class NiaVoiceTool(BaseTool):
    """N.I.A voice integration tool.

    Connects to niaharness's voice system for speech-to-text transcription,
    voice mode control, and key term extraction.
    """

    name = "nia_voice"
    description = (
        "Voice integration for N.I.A. Actions: "
        "transcribe (transcribe audio file to text), "
        "status (check voice capabilities), "
        "keyterms (extract key terms from text)"
    )
    input_model = NiaVoiceInput

    async def execute(self, arguments: NiaVoiceInput, context: ToolExecutionContext) -> ToolResult:
        action = arguments.action

        if action == "transcribe":
            if not arguments.audio_path:
                return ToolResult(output="audio_path is required for transcribe", is_error=True)
            try:
                from niaharness.voice.stream_stt import transcribe_stream
                import aiofiles
                async with aiofiles.open(arguments.audio_path, "rb") as f:
                    audio_data = await f.read()
                result = await transcribe_stream(audio_data)
                return ToolResult(output=f"Transcription: {result}")
            except ImportError:
                return ToolResult(
                    output="Voice dependencies not installed. Install with: pip install niaharness[voice]",
                    is_error=True,
                )
            except Exception as e:
                return ToolResult(output=f"Transcription failed: {e}", is_error=True)

        elif action == "status":
            try:
                from niaharness.voice.voice_mode import inspect_voice_capabilities
                caps = inspect_voice_capabilities()
                return ToolResult(output=f"Voice capabilities: {caps}")
            except ImportError:
                return ToolResult(
                    output="Voice dependencies not installed. Install with: pip install niaharness[voice]",
                    is_error=True,
                )

        elif action == "keyterms":
            if not arguments.text:
                return ToolResult(output="text is required for keyterms", is_error=True)
            try:
                from niaharness.voice.keyterms import extract_keyterms
                terms = extract_keyterms(arguments.text)
                return ToolResult(output=f"Key terms: {', '.join(terms)}")
            except ImportError:
                return ToolResult(output="Voice dependencies not installed", is_error=True)
            except Exception as e:
                return ToolResult(output=f"Keyterm extraction failed: {e}", is_error=True)

        else:
            return ToolResult(
                output=f"Unknown action: {action}. Use: transcribe, status, keyterms",
                is_error=True,
            )

    def is_read_only(self, arguments: NiaVoiceInput) -> bool:
        return arguments.action in ("status", "keyterms")
