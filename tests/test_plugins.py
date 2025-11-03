import tempfile, shutil, os
from core.tools import ToolManager
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Write out a test plugin
PLUGIN_CODE = """
class TempPlugin:
    name = 'temp_plugin'
    def run(self, params):
        return {'works': True, 'params': params}
"""
ERR_PLUGIN_CODE = """
raise Exception('fail! this plugin fails on import')
"""

def test_plugin_load_and_unload():
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, 'temp_plugin.py'), 'w') as f:
        f.write(PLUGIN_CODE)
    mgr = ToolManager()
    mgr.load_plugins_from_directory(tmpdir)
    assert 'temp_plugin' in mgr.plugin_tools()
    assert mgr.execute('temp_plugin', {'foo': 42})['output']['works']
    assert mgr.unload_plugin('temp_plugin') is True
    assert 'temp_plugin' not in mgr.plugin_tools()
    shutil.rmtree(tmpdir)

def test_plugin_reload_and_error_tolerant():
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, 'temp_plugin.py'), 'w') as f:
        f.write(PLUGIN_CODE)
    with open(os.path.join(tmpdir, 'err_plugin.py'), 'w') as f:
        f.write(ERR_PLUGIN_CODE)
    mgr = ToolManager()
    mgr.load_plugins_from_directory(tmpdir)
    assert 'temp_plugin' in mgr.plugin_tools()
    assert 'err_plugin' not in mgr.plugin_tools()
    # Should not crash and can reload (but err_plugin remains ignored)
    mgr.reload_plugins = lambda: mgr.load_plugins_from_directory(tmpdir)
    mgr.reload_plugins()
    assert 'temp_plugin' in mgr.plugin_tools()
    shutil.rmtree(tmpdir)
