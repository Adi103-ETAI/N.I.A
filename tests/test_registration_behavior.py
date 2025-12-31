from core.tool_manager import ToolManager as RawToolManager
from core.tools import register_dev_tools


def test_register_dev_tools_idempotent():
    mgr = RawToolManager()
    register_dev_tools(mgr)
    # Re-register should not create duplicates or errors
    register_dev_tools(mgr)
    tools = mgr.list_tools()
    assert tools.count("echo") == 1
    assert tools.count("hello") == 1


# TODO: Refactor for NIA/TARA architecture
# def test_brain_uses_existing_tools():
#     # Ensure CognitiveLoop does not re-register when tools already exist
#     from core.brain import CognitiveLoop  # DEPRECATED - brain.py deleted
#     from core.tools.echo_tool import EchoTool
#     mgr = RawToolManager()
#     mgr.register_tool(EchoTool)
#     loop = CognitiveLoop(memory=object(), tool_manager=mgr, model_manager=object())
#     # echo already exists, run should succeed without duplicate registration
#     resp = loop.run("hello")
#     assert isinstance(resp, str)
