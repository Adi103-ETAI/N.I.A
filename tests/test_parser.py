"""Quick test for the robust parser."""
from src.agents.tara.graph.nodes import _parse_llama_tool_calls

# Test 1: <function> format
test1 = '<|python_tag|><function>launch_app</function>{"app_name": "notepad"}'
result1 = _parse_llama_tool_calls(test1)
print(f"Test 1 (function format): {result1}")

# Test 2: .call() format  
test2 = '<|python_tag|>browser_open_url.call({"url": "google.com"})'
result2 = _parse_llama_tool_calls(test2)
print(f"Test 2 (.call format): {result2}")

# Test 3: No python_tag (should return empty)
test3 = "Just some text without tools"
result3 = _parse_llama_tool_calls(test3)
print(f"Test 3 (no tools): {result3}")

# Test 4: Multiline with extra whitespace
test4 = '''<|python_tag|>
<function>type_in_element</function>
{
    "window_alias": "notepad_1",
    "element_name": "Edit",
    "text": "Hello World"
}
'''
result4 = _parse_llama_tool_calls(test4)
print(f"Test 4 (multiline): {result4}")

print("\n✅ All tests passed!" if all([
    len(result1) == 1 and result1[0]["name"] == "launch_app",
    len(result2) == 1 and result2[0]["name"] == "browser_open_url", 
    len(result3) == 0,
    len(result4) == 1 and result4[0]["name"] == "type_in_element",
]) else "\n❌ Some tests failed!")
