import tempfile
import shutil
import os
from core.tool_manager import ToolManager

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
    assert mgr.execute('temp_plugin', {'foo': 42})['works']
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
    # Should not crash and can reload (err_plugin remains ignored)
    # Use the manager's reload_plugins helper which centralizes unload+load.
    mgr.reload_plugins(tmpdir)
    assert 'temp_plugin' in mgr.plugin_tools()
    shutil.rmtree(tmpdir)

def test_plugin_safe_mode_allowlist():
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, 'temp_plugin.py'), 'w') as f:
            f.write(PLUGIN_CODE)
        with open(os.path.join(tmpdir, 'other_plugin.py'), 'w') as f:
            f.write(PLUGIN_CODE.replace('temp_plugin', 'other_plugin'))
        with open(os.path.join(tmpdir, 'ALLOWLIST.txt'), 'w') as f:
            f.write('temp_plugin\n')
        os.environ['NIA_PLUGIN_SAFE_MODE'] = '1'
        mgr = ToolManager()
        mgr.load_plugins_from_directory(tmpdir)
        assert 'temp_plugin' in mgr.plugin_tools()
        assert 'other_plugin' not in mgr.plugin_tools()
    finally:
        os.environ.pop('NIA_PLUGIN_SAFE_MODE', None)
        shutil.rmtree(tmpdir)
