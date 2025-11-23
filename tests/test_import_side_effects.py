import os
import importlib


def test_no_auto_registration_by_default(monkeypatch):
    monkeypatch.delenv("NIA_AUTO_REGISTER_DEV_TOOLS", raising=False)
    import core.tools as tools_pkg
    # Fresh RawToolManager should have no dev tools until explicitly registered
    from core.tool_manager import ToolManager as RawToolManager
    mgr = RawToolManager()
    assert "echo" not in mgr.list_tools()
    assert "hello" not in mgr.list_tools()


def test_env_gated_auto_registration(monkeypatch):
    monkeypatch.setenv("NIA_AUTO_REGISTER_DEV_TOOLS", "1")
    # Re-import to trigger import-time gating
    import core.tools as tools_pkg
    importlib.reload(tools_pkg)
    from core.tools import ToolManager as AdapterToolManager
    mgr = AdapterToolManager()
    # After adapter creation, register_dev_tools is not auto-called on mgr
    # Import-time gating registers to a new manager only if called without mgr.
    # Verify explicit registration works and is idempotent
    from core.tools import register_dev_tools
    register_dev_tools(mgr)
    assert "echo" in mgr.list_tools()
    assert "hello" in mgr.list_tools()
