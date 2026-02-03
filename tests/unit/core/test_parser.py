"""Test for the robust parser in TARA graph nodes."""
import pytest


@pytest.fixture
def parse_llama_tool_calls():
    """Import parser function with skip if deps missing."""
    try:
        from src.agents.tara.graph.nodes import _parse_llama_tool_calls
        return _parse_llama_tool_calls
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Skipping: Missing dependency - {e}")


class TestLlamaToolCallParser:
    """Test _parse_llama_tool_calls function."""
    
    def test_function_format(self, parse_llama_tool_calls):
        """Test <function> format parsing."""
        test_input = '<|python_tag|><function>launch_app</function>{"app_name": "notepad"}'
        result = parse_llama_tool_calls(test_input)
        
        assert len(result) == 1
        assert result[0]["name"] == "launch_app"
        assert result[0]["args"]["app_name"] == "notepad"
    
    def test_call_format(self, parse_llama_tool_calls):
        """Test .call() format parsing."""
        test_input = '<|python_tag|>browser_open_url.call({"url": "google.com"})'
        result = parse_llama_tool_calls(test_input)
        
        assert len(result) == 1
        assert result[0]["name"] == "browser_open_url"
        assert result[0]["args"]["url"] == "google.com"
    
    def test_no_tools(self, parse_llama_tool_calls):
        """Test text without tools returns empty list."""
        test_input = "Just some text without tools"
        result = parse_llama_tool_calls(test_input)
        
        assert len(result) == 0
    
    def test_multiline_format(self, parse_llama_tool_calls):
        """Test multiline with extra whitespace."""
        test_input = '''<|python_tag|>
<function>type_in_element</function>
{
    "window_alias": "notepad_1",
    "element_name": "Edit",
    "text": "Hello World"
}
'''
        result = parse_llama_tool_calls(test_input)
        
        assert len(result) == 1
        assert result[0]["name"] == "type_in_element"
        assert result[0]["args"]["window_alias"] == "notepad_1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
