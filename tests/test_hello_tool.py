from core.tools.hello_tool import HelloTool

def test_greet():
    tool = HelloTool()
    out = tool.run({'who': 'NIA'})
    assert out['greeting'] == 'Hello, NIA!'
    out2 = tool.run({})
    assert 'Hello, world' in out2['greeting']
