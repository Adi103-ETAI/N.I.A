from core.tools import ToolManager
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestTool:
    name = 'hello'
    def run(self, params): return {'msg': 'hi '+str(params.get('x',''))}
    async def arun(self, params): return {'msg': 'async hi'}

def test_register_list_execute():
    mgr = ToolManager()
    mgr.register_tool(TestTool)
    assert 'hello' in mgr.list_tools()
    out = mgr.execute('hello', {'x': 'bob'})
    assert out['success']
    assert out['output']['msg'] == 'hi bob'

def test_permission_denied():
    mgr = ToolManager()
    mgr.register_tool(TestTool)
    mgr.set_permission('hello', lambda user,params: False)
    out = mgr.execute('hello', {}, user='bob')
    assert not out['success']
    assert 'denied' in out['error']

def test_async_tool():
    mgr = ToolManager()
    mgr.register_tool(TestTool)
    result = asyncio.run(mgr.aexecute('hello', {}))
    assert result['success']
    assert result['output']['msg'] == 'async hi'

def test_error_returns():
    class Bad:
        name = 'fail'
        def run(self, p): raise Exception('oops')
    mgr = ToolManager()
    mgr.register_tool(Bad)
    out = mgr.execute('fail', {})
    assert not out['success']
    assert 'oops' in out['error']
