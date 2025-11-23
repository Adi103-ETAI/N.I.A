import pytest
from core.brain import CognitiveLoop
from core.memory import InMemoryMemory

class DummyModel:
    def render_response(self, summary):
        return f"[render]{summary.get('goal','?')}"

class DummyTool:
    name = "dummy"
    def run(self, params):
        if params.get("fail"):
            raise ValueError("fail")
        return {"worked": True}

class ToolMgr:
    def __init__(self): self._r = {"dummy": DummyTool(), "echo": DummyTool()}
    def list_tools(self): return list(self._r)
    def execute(self, tool, params, **kw): return self._r[tool].run(params)
    def has_tool(self, t): return t in self._r

@pytest.fixture
def loop():
    return CognitiveLoop(memory=InMemoryMemory(), tool_manager=ToolMgr(), model_manager=DummyModel())

def test_cognitive_normal(loop):
    r = loop.run("run dummy test")
    assert "render" in r or "completed" in r

def test_cognitive_fallback(loop):
    # unknown intent falls back to echo
    r = loop.run("something with no plan")
    assert "render" in r or "completed" in r

def test_tool_error(loop):
    # force error in tool, should be error-handled and not crash
    class FailTool:
        name = "failtool"
        def run(self, p):
            raise ValueError("fail!!")
    loop.tool_manager._r["failtool"] = FailTool()
    r = loop.run("run failtool test")
    assert r # just needs to not crash

def test_reflection_memory(loop):
    loop.run("run dummy test")
    traces = [s for s in loop.memory._store if s["collection"]=="execution_traces"]
    assert len(traces) > 0
