from core.tools.hello_tool import HelloTool
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_greet():
    tool = HelloTool()
    out = tool.run({'who': 'NIA'})
    assert out['greeting'] == 'Hello, NIA!'
    out2 = tool.run({})
    assert 'Hello, world' in out2['greeting']
