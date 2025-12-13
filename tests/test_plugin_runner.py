import json
import os
import tempfile
from core.plugin_runner import execute_plugin_file, PluginExecutionError
from core.tool_manager import ToolManager


def test_execute_plugin_file_plugins_my_plugin():
    path = os.path.join(os.path.dirname(__file__), "..", "plugins", "my_plugin.py")
    path = os.path.normpath(path)
    result = execute_plugin_file(path, params={"x": 1}, timeout=3)
    assert isinstance(result, dict)
    assert result.get("ok") is True
    assert "message" in result


def test_register_and_run_subprocess_plugin(tmp_path):
    # write a small plugin to the tmp_path
    plugin_file = tmp_path / "tmp_plugin.py"
    plugin_file.write_text(
        """class TmpPlugin:
    name = 'tmp_plugin'
    def run(self, params):
        return {'ok': True, 'params': params}

if __name__ == '__main__':
    import sys, json
    payload = json.loads(sys.stdin.read() or '{}')
    print(json.dumps({'ok': True, 'params': payload}))
"""
    )

    mgr = ToolManager()
    mgr.register_subprocess_plugin(str(plugin_file), name="tmp_plugin", timeout=2)
    assert mgr.has_tool("tmp_plugin")
    res = mgr.execute("tmp_plugin", {"a": 2})
    assert isinstance(res, dict)
    assert res.get("ok") is True
    assert res.get("params") == {"a": 2}
