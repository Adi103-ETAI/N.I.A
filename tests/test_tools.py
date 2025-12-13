from core.tool_manager import ToolManager
import asyncio

class TestTool:
    name = 'hello'
    def run(self, params): return {'msg': 'hi '+str(params.get('x',''))}
    async def arun(self, params): return {'msg': 'async hi'}

def test_register_list_execute():
    mgr = ToolManager()
    mgr.register_tool(TestTool)
    assert 'hello' in mgr.list_tools()
    out = mgr.execute('hello', {'x': 'bob'})
    assert out['msg'] == 'hi bob'

def test_permission_denied():
    mgr = ToolManager()
    mgr.register_tool(TestTool)
    mgr.set_permission('hello', lambda user,params: False)
    import pytest
    with pytest.raises(PermissionError):
        mgr.execute('hello', {}, user='bob')

def test_async_tool():
    mgr = ToolManager()
    mgr.register_tool(TestTool)
    result = asyncio.run(mgr.execute_async('hello', {}))
    assert result['msg'] == 'async hi'

def test_error_returns():
    class Bad:
        name = 'fail'
        def run(self, p): raise Exception('oops')
    mgr = ToolManager()
    mgr.register_tool(Bad)
    import pytest
    with pytest.raises(RuntimeError):
        mgr.execute('fail', {})
