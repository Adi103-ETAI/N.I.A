"""VoiceManager helper for TTS/ASR provider management and runtime selection.

This helper centralizes provider selection, device & volume settings, and
fallback behaviors for voice in NIA. It works with `core.tool_manager.ToolManager`.

It supports both in-process and subprocess-based providers by using
`ToolManager.register_tool` and `ToolManager.register_subprocess_plugin`.

Usage:
    mgr = ToolManager()
    vm = VoiceManager(mgr)
    vm.set_tts_provider('speak', mode='subprocess', path='plugins/tts_pyttsx3.py')
    vm.set_asr_provider('listen', mode='subprocess', path='plugins/asr_speechrecog.py')

    vm.speak('hello world')
    text = vm.listen()

"""
from __future__ import annotations

from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging
import queue as _queue
import threading
import time


@dataclass
class ProviderConfig:
    name: str
    mode: str = "inprocess"  # 'inprocess' or 'subprocess'
    path: Optional[str] = None
    timeout: int = 5
    meta: Dict[str, Any] = field(default_factory=dict)


class VoiceManager:
    def __init__(self, tool_manager: "ToolManager", logger: Optional[logging.Logger] = None):
        # Import type hint at runtime to avoid circular import in top-level
        from core.tool_manager import ToolManager as TM

        if not isinstance(tool_manager, TM):
            raise TypeError("tool_manager must be a ToolManager instance")
        self._mgr = tool_manager
        self.logger = logger or logging.getLogger(__name__)
        self._tts_config: Optional[ProviderConfig] = None
        self._asr_config: Optional[ProviderConfig] = None
        # Basic device & voice settings
        self.device_index: Optional[int] = None
        self.volume: Optional[float] = None
        self.voice: Optional[str] = None

    # Provider registration APIs
    def set_tts_provider(self, name: str, mode: str = "inprocess", path: Optional[str] = None, timeout: int = 15, instance: Optional[Any] = None) -> None:
        """Register a TTS provider.

        mode: 'inprocess' or 'subprocess'. For 'subprocess', provide `path`.
        For 'inprocess', either provide `instance` or expect that the class/object
        is available and will be registered.
        """
        cfg = ProviderConfig(name=name, mode=mode, path=path, timeout=timeout)
        if mode == "subprocess":
            if path is None:
                raise ValueError("path is required for subprocess providers")
            self._mgr.register_subprocess_plugin(path, name=name, timeout=timeout)
            self.logger.info("Registered subprocess TTS provider %s -> %s", name, path)
        else:
            # in-process: instance or class
            if instance is None:
                raise ValueError("instance is required for inprocess TTS providers")
            self._mgr.register_tool(instance)
            self.logger.info("Registered in-process TTS provider %s", name)
        self._tts_config = cfg

    def set_asr_provider(self, name: str, mode: str = "inprocess", path: Optional[str] = None, timeout: int = 5, instance: Optional[Any] = None) -> None:
        cfg = ProviderConfig(name=name, mode=mode, path=path, timeout=timeout)
        if mode == "subprocess":
            if path is None:
                raise ValueError("path is required for subprocess providers")
            self._mgr.register_subprocess_plugin(path, name=name, timeout=timeout)
            self.logger.info("Registered subprocess ASR provider %s -> %s", name, path)
        else:
            if instance is None:
                raise ValueError("instance is required for inprocess ASR providers")
            self._mgr.register_tool(instance)
            self.logger.info("Registered in-process ASR provider %s", name)
        self._asr_config = cfg

    # Device & voice settings
    def set_device_index(self, idx: Optional[int]) -> None:
        self.device_index = idx

    def set_volume(self, vol: Optional[float]) -> None:
        self.volume = vol

    def set_voice(self, name: Optional[str]) -> None:
        self.voice = name

    # Runtime speak/listen wrappers
    def speak(self, text: str, **kwargs) -> Dict[str, Any]:
        if not self._tts_config or not self._mgr.has_tool(self._tts_config.name):
            return {"ok": False, "error": "No TTS provider registered"}
        params = {"text": text, **kwargs}
        # Inject simple device options where supported
        if self.device_index is not None:
            params.setdefault("device_index", self.device_index)
        if self.volume is not None:
            params.setdefault("volume", self.volume)
        if self.voice is not None:
            params.setdefault("voice", self.voice)
        try:
            return self._mgr.execute(self._tts_config.name, params)
        except Exception as exc:
            self.logger.exception("TTS speak failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    def listen(self, **kwargs) -> Dict[str, Any]:
        if not self._asr_config or not self._mgr.has_tool(self._asr_config.name):
            return {"ok": False, "error": "No ASR provider registered"}
        params = {**kwargs}
        if self.device_index is not None:
            params.setdefault("device_index", self.device_index)
        try:
            return self._mgr.execute(self._asr_config.name, params)
        except Exception as exc:
            self.logger.exception("ASR listen failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    def list_providers(self) -> Dict[str, Optional[ProviderConfig]]:
        return {"tts": self._tts_config, "asr": self._asr_config}

    def ensure_voice_mode(self) -> bool:
        # returns True if both tts and asr providers are present
        if not self._tts_config or not self._asr_config:
            return False
        return self._mgr.has_tool(self._tts_config.name) and self._mgr.has_tool(self._asr_config.name)


def normalize_listen_result(res: object) -> Optional[str]:
    """Normalize listen responses into plain text or None.

    Similar semantics to the helper used by the CLI.
    """
    if res is None:
        return None
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        if 'ok' in res and res.get('ok') is False:
            return None
        if 'success' in res and res.get('success') is False:
            return None
        for key in ('text', 'output', 'transcript', 'message'):
            if key in res and res.get(key):
                return res.get(key)
    return None


class BackgroundListener:
    """Background thread that polls an ASR provider and enqueues recognized text.

    The listener calls `voice_manager.listen()` repeatedly with a short timeout,
    normalizes results, and places recognized text onto `output_queue`.

    The `voice_manager` may be any object exposing a `listen(**kwargs)` method
    (e.g. `VoiceManager` or a test double).
    """
    def __init__(self, voice_manager: object, output_queue: _queue.Queue | None = None, poll_interval: float = 0.1, timeout: float = 2.0):
        self.voice_manager = voice_manager
        self.output_queue = output_queue or _queue.Queue()
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, join_timeout: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout=join_timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                try:
                    res = self.voice_manager.listen(_timeout=self.timeout)
                except Exception:
                    res = None
                text = normalize_listen_result(res)
                if text:
                    try:
                        self.output_queue.put_nowait(text)
                    except Exception:
                        pass
            except Exception:
                # swallow errors and continue polling
                pass
            time.sleep(self.poll_interval)
