import os
from core.tool_manager import ToolManager


def test_register_subprocess_tts_plugin():
    mgr = ToolManager()
    path = os.path.join(os.path.dirname(__file__), "..", "plugins", "tts_pyttsx3.py")
    path = os.path.normpath(path)
    mgr.register_subprocess_plugin(path, name="speak")
    assert mgr.has_tool("speak")
    res = mgr.execute("speak", {"text": "hello world"})
    assert isinstance(res, dict)
    assert res.get("ok") is True


def test_register_subprocess_asr_plugin():
    mgr = ToolManager()
    path = os.path.join(os.path.dirname(__file__), "..", "plugins", "asr_speechrecog.py")
    path = os.path.normpath(path)
    mgr.register_subprocess_plugin(path, name="listen")
    assert mgr.has_tool("listen")
    res = mgr.execute("listen", {"typed": "hi there"})
    assert isinstance(res, dict)
    assert res.get("ok") is True
    assert res.get("text") == "hi there"
