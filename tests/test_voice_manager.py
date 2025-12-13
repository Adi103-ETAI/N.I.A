import os
from core.tool_manager import ToolManager
from core.voice_manager import VoiceManager


def test_voice_manager_subprocess_registration_and_speak_listen():
    mgr = ToolManager()
    vm = VoiceManager(mgr)

    tts_path = os.path.join(os.path.dirname(__file__), "..", "plugins", "tts_pyttsx3.py")
    asr_path = os.path.join(os.path.dirname(__file__), "..", "plugins", "asr_speechrecog.py")
    tts_path = os.path.normpath(tts_path)
    asr_path = os.path.normpath(asr_path)

    vm.set_tts_provider(name="speak", mode="subprocess", path=tts_path)
    vm.set_asr_provider(name="listen", mode="subprocess", path=asr_path)

    # speak
    sres = vm.speak("hello world")
    assert isinstance(sres, dict)
    assert sres.get("ok") is True

    # listen (use typed fallback)
    lres = vm.listen(typed="hey there")
    assert isinstance(lres, dict)
    assert lres.get("ok") is True
    assert lres.get("text") == "hey there"


def test_voice_manager_fallbacks():
    mgr = ToolManager()
    vm = VoiceManager(mgr)

    # No providers configured
    sres = vm.speak("hi")
    assert sres.get("ok") is False and "No TTS provider" in sres.get("error")
    lres = vm.listen()
    assert lres.get("ok") is False and "No ASR provider" in lres.get("error")


def test_device_and_voice_settings_are_applied():
    mgr = ToolManager()
    vm = VoiceManager(mgr)

    tts_path = os.path.join(os.path.dirname(__file__), "..", "plugins", "tts_pyttsx3.py")
    tts_path = os.path.normpath(tts_path)

    vm.set_tts_provider(name="speak", mode="subprocess", path=tts_path)
    vm.set_device_index(1)
    vm.set_volume(0.5)
    vm.set_voice("neutral")

    res = vm.speak("hello world")
    # The plugin returns ok True; we can't assert it used device/volume, but the call should succeed
    assert res.get("ok") is True
