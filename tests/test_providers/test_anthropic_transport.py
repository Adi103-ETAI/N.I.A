"""Tests for the Anthropic transport layer.

Covers:
  - Tool-schema sanitization (nullable-union collapse, top-level union strip)
  - Tool-ID sanitization
  - Tool conversion (OpenAI → Anthropic, dedup, cache_control forwarding)
  - Image-source conversion (data URI + URL)
  - Per-role message conversion (assistant / tool / user / system)
  - Repair pipeline (orphan-strip, merge, thinking-signatures, evict-screenshots)
  - Prompt caching (last block, last tool, system)
  - Model-name normalization + capability detection
  - build_anthropic_kwargs (central entry point) — multiple scenarios
  - Endpoint classifiers (third-party / Kimi / DeepSeek / MiniMax / Azure)
  - sanitize_anthropic_kwargs (Responses-API leak guard)
  - NIA ConversationMessage conversion path
"""

from __future__ import annotations

import json

import pytest

from niaharness.providers.anthropic_transport import (
    ADAPTIVE_EFFORT_MAP,
    THINKING_BUDGET,
    _apply_assistant_cache_control_to_last_cacheable_block,
    _content_parts_to_anthropic_blocks,
    _convert_assistant_message,
    _convert_content_part_to_anthropic,
    _convert_content_to_anthropic,
    _convert_tool_message_to_result,
    _convert_user_message,
    _evict_old_screenshots,
    _forbids_sampling_params,
    _get_anthropic_max_output,
    _is_azure_anthropic_endpoint,
    _is_deepseek_anthropic_endpoint,
    _is_kimi_coding_endpoint,
    _is_kimi_family_endpoint,
    _is_minimax_anthropic_endpoint,
    _is_oauth_token,
    _is_third_party_anthropic_endpoint,
    _manage_thinking_signatures,
    _merge_consecutive_roles,
    _normalize_tool_input_schema,
    _resolve_anthropic_messages_max_tokens,
    _resolve_positive_max_tokens,
    _sanitize_replay_block,
    _sanitize_tool_id,
    _strip_nullable_unions,
    _strip_orphaned_tool_blocks,
    _supports_adaptive_thinking,
    _supports_fast_mode,
    _supports_xhigh_effort,
    apply_cache_control_to_last_tool,
    apply_cache_control_to_system,
    base_url_host_matches,
    build_anthropic_kwargs,
    convert_conversation_messages_to_anthropic,
    convert_messages_to_anthropic,
    convert_tools_to_anthropic,
    normalize_model_name,
    sanitize_anthropic_kwargs,
)


# ---------------------------------------------------------------------------
# Tool-schema sanitization
# ---------------------------------------------------------------------------

class TestStripNullableUnions:
    def test_collapses_anyof_with_null_to_non_null_branch(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
        }
        result = _strip_nullable_unions(schema, keep_nullable_hint=False)
        assert result["properties"]["age"] == {"type": "integer"}

    def test_preserves_required_array(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        result = _strip_nullable_unions(schema)
        assert result["required"] == ["name"]

    def test_recurses_into_nested_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "email": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    },
                },
            },
        }
        result = _strip_nullable_unions(schema)
        assert result["properties"]["user"]["properties"]["email"] == {"type": "string"}

    def test_recurses_into_array_items(self):
        schema = {
            "type": "array",
            "items": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        }
        result = _strip_nullable_unions(schema)
        assert result["items"] == {"type": "string"}

    def test_passes_through_non_dict(self):
        assert _strip_nullable_unions(None) is None
        assert _strip_nullable_unions("foo") == "foo"


class TestNormalizeToolInputSchema:
    def test_strips_top_level_oneof(self):
        schema = {
            "oneOf": [{"type": "string"}, {"type": "integer"}],
            "description": "test",
        }
        result = _normalize_tool_input_schema(schema)
        assert "oneOf" not in result
        assert result["type"] == "object"
        assert result["description"] == "test"

    def test_strips_top_level_anyof(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        result = _normalize_tool_input_schema(schema)
        assert "anyOf" not in result
        assert result["type"] == "object"

    def test_strips_top_level_allof(self):
        schema = {"allOf": [{"type": "string"}]}
        result = _normalize_tool_input_schema(schema)
        assert "allOf" not in result

    def test_ensures_object_has_dict_properties(self):
        schema = {"type": "object", "properties": "not-a-dict"}
        result = _normalize_tool_input_schema(schema)
        assert result["properties"] == {}

    def test_returns_default_for_empty_schema(self):
        result = _normalize_tool_input_schema({})
        assert result == {"type": "object", "properties": {}}

    def test_returns_default_for_falsy_schema(self):
        result = _normalize_tool_input_schema(None)
        assert result == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# Tool-ID sanitization
# ---------------------------------------------------------------------------

class TestSanitizeToolId:
    def test_replaces_invalid_chars_with_underscore(self):
        assert _sanitize_tool_id("tool.123!@#") == "tool_123___"

    def test_returns_default_for_empty(self):
        assert _sanitize_tool_id("") == "tool_0"
        assert _sanitize_tool_id(None) == "tool_0"

    def test_preserves_valid_id(self):
        assert _sanitize_tool_id("toolu_abc123") == "toolu_abc123"

    def test_preserves_dashes_and_underscores(self):
        assert _sanitize_tool_id("call_123-abc") == "call_123-abc"


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------

class TestConvertToolsToAnthropic:
    def test_converts_openai_format(self):
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        result = convert_tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather"
        assert result[0]["input_schema"]["type"] == "object"

    def test_passes_through_anthropic_format(self):
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ]
        result = convert_tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"

    def test_dedupes_by_name(self):
        tools = [
            {"function": {"name": "dup", "description": "first"}},
            {"function": {"name": "dup", "description": "second"}},
        ]
        result = convert_tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]["description"] == "first"

    def test_forwards_cache_control(self):
        tools = [
            {
                "function": {"name": "t1", "description": "t1"},
                "cache_control": {"type": "ephemeral"},
            }
        ]
        result = convert_tools_to_anthropic(tools)
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_strips_nullable_unions_from_input_schema(self):
        tools = [
            {
                "function": {
                    "name": "t",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        },
                    },
                },
            }
        ]
        result = convert_tools_to_anthropic(tools)
        assert result[0]["input_schema"]["properties"]["x"] == {"type": "string"}

    def test_returns_empty_for_none_input(self):
        assert convert_tools_to_anthropic(None) == []
        assert convert_tools_to_anthropic([]) == []


# ---------------------------------------------------------------------------
# Image-source conversion
# ---------------------------------------------------------------------------

class TestImageSourceConversion:
    def test_data_uri_jpeg(self):
        url = "data:image/jpeg;base64,/9j/4AAQ"
        src = _convert_content_part_to_anthropic({"type": "image_url", "image_url": {"url": url}})
        assert src == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQ"},
        }

    def test_data_uri_png(self):
        url = "data:image/png;base64,iVBOR"
        src = _convert_content_part_to_anthropic({"type": "image_url", "image_url": {"url": url}})
        assert src["source"]["media_type"] == "image/png"

    def test_plain_url(self):
        url = "https://example.com/cat.jpg"
        src = _convert_content_part_to_anthropic({"type": "image_url", "image_url": {"url": url}})
        assert src == {"type": "image", "source": {"type": "url", "url": url}}

    def test_text_part(self):
        result = _convert_content_part_to_anthropic({"type": "text", "text": "hello"})
        assert result == {"type": "text", "text": "hello"}

    def test_string_passthrough(self):
        assert _convert_content_part_to_anthropic("hello") == {"type": "text", "text": "hello"}

    def test_none_returns_none(self):
        assert _convert_content_part_to_anthropic(None) is None


# ---------------------------------------------------------------------------
# Per-role message conversion
# ---------------------------------------------------------------------------

class TestConvertUserMessage:
    def test_string_content(self):
        result = _convert_user_message("hello")
        assert result == {"role": "user", "content": "hello"}

    def test_empty_string_becomes_placeholder(self):
        result = _convert_user_message("")
        assert result["content"] == "(empty message)"

    def test_whitespace_only_becomes_placeholder(self):
        result = _convert_user_message("   ")
        assert result["content"] == "(empty message)"

    def test_list_content(self):
        result = _convert_user_message([{"type": "text", "text": "hi"}])
        assert result["role"] == "user"
        assert result["content"] == [{"type": "text", "text": "hi"}]

    def test_list_with_only_empty_text_becomes_placeholder(self):
        result = _convert_user_message([{"type": "text", "text": "  "}])
        assert result["content"] == [{"type": "text", "text": "(empty message)"}]


class TestConvertAssistantMessage:
    def test_plain_text_content(self):
        result = _convert_assistant_message({"content": "hello"})
        assert result == {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}

    def test_empty_content_becomes_placeholder(self):
        result = _convert_assistant_message({"content": ""})
        assert result["content"] == [{"type": "text", "text": "(empty)"}]

    def test_tool_calls_become_tool_use_blocks(self):
        result = _convert_assistant_message({
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                }
            ],
        })
        blocks = result["content"]
        tool_use_blocks = [b for b in blocks if b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["id"] == "call_1"
        assert tool_use_blocks[0]["name"] == "get_weather"
        assert tool_use_blocks[0]["input"] == {"city": "SF"}

    def test_preserves_thinking_blocks_from_reasoning_details(self):
        result = _convert_assistant_message({
            "content": "hello",
            "reasoning_details": [{"type": "thinking", "thinking": "deep thought"}],
        })
        blocks = result["content"]
        thinking_blocks = [b for b in blocks if b.get("type") == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "deep thought"

    def test_invalid_tool_call_args_become_empty_dict(self):
        result = _convert_assistant_message({
            "content": "",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "t", "arguments": "not-json"}},
            ],
        })
        tool_use = [b for b in result["content"] if b.get("type") == "tool_use"][0]
        assert tool_use["input"] == {}


class TestConvertToolMessageToResult:
    def test_creates_user_message_with_tool_result(self):
        result: list[dict] = []
        _convert_tool_message_to_result(result, {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "sunny",
        })
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "tool_result"
        assert result[0]["content"][0]["tool_use_id"] == "call_1"
        assert result[0]["content"][0]["content"] == "sunny"

    def test_merges_consecutive_tool_results_into_one_user_message(self):
        result: list[dict] = []
        _convert_tool_message_to_result(result, {
            "role": "tool", "tool_call_id": "call_1", "content": "sunny",
        })
        _convert_tool_message_to_result(result, {
            "role": "tool", "tool_call_id": "call_2", "content": "rainy",
        })
        assert len(result) == 1
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["tool_use_id"] == "call_1"
        assert result[0]["content"][1]["tool_use_id"] == "call_2"

    def test_handles_empty_content(self):
        result: list[dict] = []
        _convert_tool_message_to_result(result, {
            "role": "tool", "tool_call_id": "call_1", "content": "",
        })
        assert result[0]["content"][0]["content"] == "(no output)"

    def test_sanitizes_tool_use_id(self):
        result: list[dict] = []
        _convert_tool_message_to_result(result, {
            "role": "tool", "tool_call_id": "tool.id.with.dots", "content": "x",
        })
        assert result[0]["content"][0]["tool_use_id"] == "tool_id_with_dots"


# ---------------------------------------------------------------------------
# Repair pipeline
# ---------------------------------------------------------------------------

class TestStripOrphanedToolBlocks:
    def test_strips_tool_use_with_no_matching_result(self):
        result = [
            {"role": "assistant", "content": [
                {"type": "text", "text": "thinking..."},
                {"type": "tool_use", "id": "orphan", "name": "t", "input": {}},
            ]},
            {"role": "user", "content": "next message"},
        ]
        _strip_orphaned_tool_blocks(result)
        # tool_use should be stripped, text block kept
        assert result[0]["content"] == [{"type": "text", "text": "thinking..."}]

    def test_keeps_tool_use_with_adjacent_result(self):
        result = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_1", "name": "t", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "ok"},
            ]},
        ]
        _strip_orphaned_tool_blocks(result)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "tool_use"

    def test_strips_tool_result_with_no_matching_use(self):
        result = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "orphan", "content": "x"},
            ]},
        ]
        _strip_orphaned_tool_blocks(result)
        assert result[0]["content"] == [{"type": "text", "text": "(tool result removed)"}]

    def test_invalidates_thinking_signature_when_stripping(self):
        result = [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "deep", "signature": "sig_123"},
                {"type": "tool_use", "id": "orphan", "name": "t", "input": {}},
            ]},
        ]
        _strip_orphaned_tool_blocks(result)
        assert result[0].get("_thinking_signature_invalidated") is True


class TestMergeConsecutiveRoles:
    def test_merges_consecutive_user_messages_with_strings(self):
        result = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "world"},
        ]
        merged = _merge_consecutive_roles(result)
        assert len(merged) == 1
        assert merged[0]["content"] == "hello\nworld"

    def test_merges_consecutive_user_messages_with_lists(self):
        result = [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            {"role": "user", "content": [{"type": "text", "text": "b"}]},
        ]
        merged = _merge_consecutive_roles(result)
        assert len(merged) == 1
        assert len(merged[0]["content"]) == 2

    def test_merges_consecutive_assistant_messages_dropping_thinking(self):
        result = [
            {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "deep"},
                {"type": "text", "text": "b"},
            ]},
        ]
        merged = _merge_consecutive_roles(result)
        assert len(merged) == 1
        # thinking should be dropped from the second message
        assert all(b.get("type") != "thinking" for b in merged[0]["content"])

    def test_preserves_alternating_roles(self):
        result = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
            {"role": "user", "content": "c"},
        ]
        merged = _merge_consecutive_roles(result)
        assert len(merged) == 3
        assert [m["role"] for m in merged] == ["user", "assistant", "user"]


class TestManageThinkingSignatures:
    def test_third_party_strips_all_thinking_blocks(self):
        result = [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "deep", "signature": "sig_123"},
                {"type": "text", "text": "hello"},
            ]},
        ]
        _manage_thinking_signatures(result, base_url="https://minimax.io/anthropic", model=None)
        assert all(b.get("type") != "thinking" for b in result[0]["content"])

    def test_direct_anthropic_keeps_signed_thinking_on_latest_assistant(self):
        result = [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "deep", "signature": "sig_123"},
                {"type": "text", "text": "hello"},
            ]},
        ]
        _manage_thinking_signatures(result, base_url=None, model="claude-opus-4-7")
        thinking_blocks = [b for b in result[0]["content"] if b.get("type") == "thinking"]
        assert len(thinking_blocks) == 1

    def test_direct_anthropic_strips_thinking_from_non_latest_assistant(self):
        result = [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "deep", "signature": "sig_123"},
                {"type": "text", "text": "first"},
            ]},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": [{"type": "text", "text": "second"}]},
        ]
        _manage_thinking_signatures(result, base_url=None, model="claude-opus-4-7")
        # First assistant should have thinking stripped
        assert all(b.get("type") != "thinking" for b in result[0]["content"])

    def test_downgrades_unsigned_thinking_to_text_on_latest(self):
        result = [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "deep"},  # no signature
                {"type": "text", "text": "hello"},
            ]},
        ]
        _manage_thinking_signatures(result, base_url=None, model="claude-opus-4-7")
        # Unsigned thinking should become a text block
        assert all(b.get("type") != "thinking" for b in result[0]["content"])
        assert any(
            b.get("type") == "text" and b.get("text") == "deep"
            for b in result[0]["content"]
        )

    def test_signature_invalidated_demotes_to_text(self):
        result = [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "deep", "signature": "sig_123"},
                {"type": "text", "text": "hello"},
            ], "_thinking_signature_invalidated": True},
        ]
        _manage_thinking_signatures(result, base_url=None, model="claude-opus-4-7")
        assert all(b.get("type") != "thinking" for b in result[0]["content"])
        assert any(
            b.get("type") == "text" and b.get("text") == "deep"
            for b in result[0]["content"]
        )
        # Bookkeeping flag should be dropped
        assert "_thinking_signature_invalidated" not in result[0]

    def test_kimi_preserves_unsigned_thinking(self):
        result = [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "deep"},  # unsigned
                {"type": "text", "text": "hello"},
            ]},
        ]
        _manage_thinking_signatures(
            result, base_url="https://api.kimi.com/coding", model="kimi-k2"
        )
        thinking_blocks = [b for b in result[0]["content"] if b.get("type") == "thinking"]
        assert len(thinking_blocks) == 1


class TestEvictOldScreenshots:
    def test_keeps_only_three_most_recent_screenshots(self):
        # Build 5 tool_result messages each with an image.
        result = []
        for i in range(5):
            result.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": f"call_{i}", "name": "screenshot", "input": {}}
            ]})
            result.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": f"call_{i}", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": f"img{i}"}},
                ]},
            ]})
        _evict_old_screenshots(result)
        # The first 2 (oldest) should have their images replaced; the last 3 should keep them.
        image_count = 0
        replaced_count = 0
        for msg in result:
            if not isinstance(msg.get("content"), list):
                continue
            for block in msg["content"]:
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                inner = block.get("content")
                if not isinstance(inner, list):
                    continue
                for b in inner:
                    if isinstance(b, dict):
                        if b.get("type") == "image":
                            image_count += 1
                        elif b.get("type") == "text" and "screenshot removed" in b.get("text", ""):
                            replaced_count += 1
        assert image_count == 3
        assert replaced_count == 2


# ---------------------------------------------------------------------------
# Prompt caching
# ---------------------------------------------------------------------------

class TestPromptCaching:
    def test_apply_cache_control_to_last_text_block(self):
        blocks = [
            {"type": "thinking", "thinking": "x"},
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        _apply_assistant_cache_control_to_last_cacheable_block(blocks, {"type": "ephemeral"})
        # Last text block should have cache_control
        assert blocks[2]["cache_control"] == {"type": "ephemeral"}
        # First text block should NOT
        assert "cache_control" not in blocks[1]

    def test_apply_cache_control_to_last_tool_use_block(self):
        blocks = [
            {"type": "text", "text": "x"},
            {"type": "tool_use", "id": "1", "name": "t", "input": {}},
        ]
        _apply_assistant_cache_control_to_last_cacheable_block(blocks, {"type": "ephemeral"})
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}

    def test_apply_cache_control_no_op_when_not_dict(self):
        blocks = [{"type": "text", "text": "x"}]
        _apply_assistant_cache_control_to_last_cacheable_block(blocks, None)
        assert "cache_control" not in blocks[0]

    def test_apply_cache_control_to_last_tool(self):
        tools = [
            {"name": "a", "description": "a", "input_schema": {"type": "object"}},
            {"name": "b", "description": "b", "input_schema": {"type": "object"}},
        ]
        apply_cache_control_to_last_tool(tools)
        assert "cache_control" not in tools[0]
        assert tools[1]["cache_control"] == {"type": "ephemeral"}

    def test_apply_cache_control_to_system_string(self):
        result = apply_cache_control_to_system("You are helpful.")
        assert isinstance(result, list)
        assert result[0]["text"] == "You are helpful."
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_apply_cache_control_to_system_list(self):
        blocks = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        result = apply_cache_control_to_system(blocks)
        assert "cache_control" not in result[0]
        assert result[1]["cache_control"] == {"type": "ephemeral"}

    def test_apply_cache_control_to_system_no_op_for_empty(self):
        assert apply_cache_control_to_system(None) is None
        assert apply_cache_control_to_system("") == ""


# ---------------------------------------------------------------------------
# Model-name normalization + capability detection
# ---------------------------------------------------------------------------

class TestNormalizeModelName:
    def test_strips_anthropic_prefix(self):
        assert normalize_model_name("anthropic/claude-opus-4-6") == "claude-opus-4-6"

    def test_converts_dots_to_hyphens_for_claude(self):
        assert normalize_model_name("claude-opus-4.6") == "claude-opus-4-6"

    def test_preserves_dots_when_requested(self):
        assert normalize_model_name("qwen3.5-plus", preserve_dots=True) == "qwen3.5-plus"

    def test_preserves_bedrock_model_ids(self):
        assert normalize_model_name("anthropic.claude-opus-4-7") == "anthropic.claude-opus-4-7"
        assert normalize_model_name("us.anthropic.claude-sonnet-4-5-v1:0") == "us.anthropic.claude-sonnet-4-5-v1:0"

    def test_does_not_convert_dots_for_non_claude_models(self):
        assert normalize_model_name("gpt-5.4") == "gpt-5.4"


class TestModelCapabilityDetection:
    def test_supports_adaptive_thinking_for_modern_claude(self):
        assert _supports_adaptive_thinking("claude-opus-4-7") is True
        assert _supports_adaptive_thinking("claude-opus-4-6") is True

    def test_does_not_support_adaptive_thinking_for_legacy_claude(self):
        assert _supports_adaptive_thinking("claude-3-5-sonnet") is False
        assert _supports_adaptive_thinking("claude-opus-4-1") is False

    def test_does_not_support_adaptive_thinking_for_non_claude(self):
        assert _supports_adaptive_thinking("gpt-5") is False

    def test_supports_xhigh_for_modern_claude(self):
        assert _supports_xhigh_effort("claude-opus-4-7") is True

    def test_does_not_support_xhigh_for_4_6(self):
        assert _supports_xhigh_effort("claude-opus-4-6") is False

    def test_forbids_sampling_params_for_modern_claude(self):
        assert _forbids_sampling_params("claude-opus-4-7") is True

    def test_allows_sampling_params_for_4_6(self):
        assert _forbids_sampling_params("claude-opus-4-6") is False

    def test_allows_sampling_params_for_legacy(self):
        assert _forbids_sampling_params("claude-3-5-sonnet") is False

    def test_supports_fast_mode_for_opus_4_6(self):
        assert _supports_fast_mode("claude-opus-4-6") is True
        assert _supports_fast_mode("claude-opus-4.6") is True

    def test_does_not_support_fast_mode_for_others(self):
        assert _supports_fast_mode("claude-opus-4-7") is False


class TestMaxTokensResolution:
    def test_returns_requested_when_positive(self):
        assert _resolve_positive_max_tokens(4096) == 4096
        assert _resolve_positive_max_tokens(8192.0) == 8192

    def test_returns_none_for_zero_or_negative(self):
        assert _resolve_positive_max_tokens(0) is None
        assert _resolve_positive_max_tokens(-1) is None

    def test_returns_none_for_bool(self):
        assert _resolve_positive_max_tokens(True) is None

    def test_returns_none_for_non_numeric(self):
        assert _resolve_positive_max_tokens("4096") is None
        assert _resolve_positive_max_tokens(None) is None

    def test_falls_back_to_model_ceiling(self):
        result = _resolve_anthropic_messages_max_tokens(None, "claude-opus-4-7")
        assert result == 128_000

    def test_falls_back_to_default_for_unknown_model(self):
        result = _resolve_anthropic_messages_max_tokens(None, "claude-future-9-9")
        assert result == 128_000

    def test_raises_when_no_resolution_possible(self):
        # Use a model with explicit 0 ceiling — can't easily construct, so
        # just verify the ValueError path with a non-positive requested
        # and a model that maps to 0. We monkey-patch the limits dict instead.
        import niaharness.providers.anthropic_transport as mod
        original = mod._ANTHROPIC_OUTPUT_LIMITS
        try:
            mod._ANTHROPIC_OUTPUT_LIMITS = {}
            mod._ANTHROPIC_DEFAULT_OUTPUT_LIMIT = 0
            with pytest.raises(ValueError, match="positive max_tokens"):
                _resolve_anthropic_messages_max_tokens(None, "unknown-model")
        finally:
            mod._ANTHROPIC_OUTPUT_LIMITS = original
            mod._ANTHROPIC_DEFAULT_OUTPUT_LIMIT = 128_000


# ---------------------------------------------------------------------------
# Endpoint classifiers
# ---------------------------------------------------------------------------

class TestEndpointClassifiers:
    def test_is_third_party_anthropic_endpoint(self):
        assert _is_third_party_anthropic_endpoint("https://api.minimax.io/anthropic") is True
        assert _is_third_party_anthropic_endpoint("https://api.anthropic.com") is False
        assert _is_third_party_anthropic_endpoint(None) is False
        assert _is_third_party_anthropic_endpoint("") is False

    def test_is_kimi_coding_endpoint(self):
        assert _is_kimi_coding_endpoint("https://api.kimi.com/coding") is True
        assert _is_kimi_coding_endpoint("https://api.kimi.com/coding/") is True
        assert _is_kimi_coding_endpoint("https://api.kimi.com/v1") is False

    def test_is_kimi_family_endpoint_by_url(self):
        assert _is_kimi_family_endpoint("https://api.kimi.com/coding", None) is True
        assert _is_kimi_family_endpoint("https://moonshot.ai/v1", None) is True

    def test_is_kimi_family_endpoint_by_model(self):
        assert _is_kimi_family_endpoint(None, "kimi-k2") is True
        assert _is_kimi_family_endpoint(None, "moonshot-v1") is True
        assert _is_kimi_family_endpoint(None, "claude-opus-4-7") is False

    def test_is_deepseek_anthropic_endpoint(self):
        assert _is_deepseek_anthropic_endpoint("https://api.deepseek.com/anthropic") is True
        assert _is_deepseek_anthropic_endpoint("https://api.deepseek.com/v1") is False

    def test_is_minimax_anthropic_endpoint(self):
        assert _is_minimax_anthropic_endpoint("https://api.minimax.io/anthropic") is True
        assert _is_minimax_anthropic_endpoint("https://api.minimaxi.com/anthropic") is True
        assert _is_minimax_anthropic_endpoint("https://api.anthropic.com") is False

    def test_is_azure_anthropic_endpoint(self):
        assert _is_azure_anthropic_endpoint(
            "https://my-resource.services.ai.azure.com/models/anthropic"
        ) is True
        assert _is_azure_anthropic_endpoint(
            "https://my-resource.openai.azure.com/anthropic"
        ) is True
        assert _is_azure_anthropic_endpoint("https://api.anthropic.com") is False

    def test_base_url_host_matches(self):
        assert base_url_host_matches("https://api.kimi.com/coding", "api.kimi.com") is True
        assert base_url_host_matches("https://sub.api.kimi.com/v1", "api.kimi.com") is True
        assert base_url_host_matches("https://api.anthropic.com", "api.kimi.com") is False
        assert base_url_host_matches(None, "api.kimi.com") is False


# ---------------------------------------------------------------------------
# OAuth token detection
# ---------------------------------------------------------------------------

class TestIsOAuthToken:
    def test_regular_api_key_is_not_oauth(self):
        assert _is_oauth_token("sk-ant-api03-abc123") is False

    def test_oauth_token_starts_with_sk_ant_non_api(self):
        assert _is_oauth_token("sk-ant-oat01-abc123") is True

    def test_jwt_starts_with_eyJ(self):
        assert _is_oauth_token("eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9") is True

    def test_cc_prefix(self):
        assert _is_oauth_token("cc-abc123") is True

    def test_empty_or_non_string(self):
        assert _is_oauth_token("") is False
        assert _is_oauth_token(None) is False
        assert _is_oauth_token(123) is False


# ---------------------------------------------------------------------------
# sanitize_anthropic_kwargs (Responses-API leak guard)
# ---------------------------------------------------------------------------

class TestSanitizeAnthropicKwargs:
    def test_strips_responses_only_keys(self):
        kwargs = {
            "model": "claude-opus-4-7",
            "messages": [],
            "instructions": "be helpful",  # Responses-only
            "input": "foo",  # Responses-only
            "store": True,  # Responses-only
            "parallel_tool_calls": True,  # Responses-only
        }
        result = sanitize_anthropic_kwargs(kwargs)
        assert "instructions" not in result
        assert "input" not in result
        assert "store" not in result
        assert "parallel_tool_calls" not in result
        assert result["model"] == "claude-opus-4-7"

    def test_passes_through_clean_kwargs(self):
        kwargs = {"model": "claude-opus-4-7", "messages": [], "max_tokens": 4096}
        result = sanitize_anthropic_kwargs(kwargs)
        assert result == kwargs

    def test_returns_non_dict_unchanged(self):
        assert sanitize_anthropic_kwargs("not a dict") == "not a dict"
        assert sanitize_anthropic_kwargs(None) is None


# ---------------------------------------------------------------------------
# Block-replay sanitization
# ---------------------------------------------------------------------------

class TestSanitizeReplayBlock:
    def test_text_block_whitelist(self):
        b = {
            "type": "text",
            "text": "hello",
            "parsed_output": "should be stripped",  # output-only
            "citations": None,  # should be stripped (None)
        }
        result = _sanitize_replay_block(b)
        assert result == {"type": "text", "text": "hello"}

    def test_text_block_preserves_non_empty_citations(self):
        b = {"type": "text", "text": "hello", "citations": [{"x": 1}]}
        result = _sanitize_replay_block(b)
        assert result["citations"] == [{"x": 1}]

    def test_thinking_block_preserves_signature(self):
        b = {"type": "thinking", "thinking": "deep", "signature": "sig_123"}
        result = _sanitize_replay_block(b)
        assert result == {"type": "thinking", "thinking": "deep", "signature": "sig_123"}

    def test_redacted_thinking_dropped_without_data(self):
        b = {"type": "redacted_thinking"}
        assert _sanitize_replay_block(b) is None

    def test_tool_use_block_sanitizes_id(self):
        b = {
            "type": "tool_use",
            "id": "tool.id.with.dots",
            "name": "t",
            "input": {"x": 1},
            "caller": "should be stripped",  # output-only
        }
        result = _sanitize_replay_block(b)
        assert result["id"] == "tool_id_with_dots"
        assert "caller" not in result

    def test_image_block_passthrough(self):
        b = {"type": "image", "source": {"type": "url", "url": "https://x.com/y.png"}}
        result = _sanitize_replay_block(b)
        assert result == {"type": "image", "source": {"type": "url", "url": "https://x.com/y.png"}}

    def test_unknown_block_returns_none(self):
        b = {"type": "unknown_type", "data": "x"}
        assert _sanitize_replay_block(b) is None


# ---------------------------------------------------------------------------
# build_anthropic_kwargs (central entry point)
# ---------------------------------------------------------------------------

class TestBuildAnthropicKwargs:
    def test_basic_kwargs_for_simple_user_message(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
        )
        assert kwargs["model"] == "claude-opus-4-7"
        assert kwargs["max_tokens"] == 1024
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"

    def test_normalizes_model_name(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4.6",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
        )
        assert kwargs["model"] == "claude-opus-4-6"

    def test_resolves_max_tokens_from_model_ceiling_when_none(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=None,
        )
        assert kwargs["max_tokens"] == 128_000

    def test_applies_prompt_caching_to_system_prompt(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="You are helpful.",
            max_tokens=1024,
        )
        system = kwargs["system"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_applies_prompt_caching_to_last_tool(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {"function": {"name": "a", "description": "a", "parameters": {"type": "object"}}},
                {"function": {"name": "b", "description": "b", "parameters": {"type": "object"}}},
            ],
            max_tokens=1024,
        )
        assert "cache_control" not in kwargs["tools"][0]
        assert kwargs["tools"][1]["cache_control"] == {"type": "ephemeral"}

    def test_disables_caching_when_flag_false(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="You are helpful.",
            max_tokens=1024,
            enable_caching=False,
        )
        # System should be a plain string (no cache_control)
        assert isinstance(kwargs["system"], str)

    def test_adaptive_thinking_for_modern_claude(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096,
            reasoning_effort="high",
        )
        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["output_config"]["effort"] == "high"

    def test_adaptive_thinking_downgrades_xhigh_for_4_6(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096,
            reasoning_effort="xhigh",
        )
        # 4.6 doesn't support xhigh — should downgrade to high
        assert kwargs["thinking"]["output_config"]["effort"] == "high"

    def test_manual_thinking_for_legacy_claude(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-1",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=32_768,  # must exceed THINKING_BUDGET["high"] = 16_000
            reasoning_effort="high",
        )
        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["budget_tokens"] == 16_000  # THINKING_BUDGET["high"]

    def test_manual_thinking_clamps_budget_to_max_tokens_minus_one(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-1",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096,
            reasoning_effort="high",
        )
        # Anthropic requires budget_tokens < max_tokens.
        assert kwargs["thinking"]["budget_tokens"] == 4095

    def test_strips_sampling_params_for_modern_claude(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
        )
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs

    def test_keeps_sampling_params_for_4_6(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
            temperature=0.7,
        )
        assert kwargs["temperature"] == 0.7

    def test_oauth_transforms_apply_system_prefix(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="You are NIA.",
            max_tokens=1024,
            is_oauth=True,
        )
        system = kwargs["system"]
        # System should have the Claude Code prefix prepended
        assert isinstance(system, list)
        assert "Claude Code" in system[0]["text"]

    def test_oauth_transforms_prefix_tool_names(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {"function": {"name": "get_weather", "description": "x", "parameters": {"type": "object"}}},
            ],
            max_tokens=1024,
            is_oauth=True,
        )
        assert kwargs["tools"][0]["name"] == "mcp__get_weather"

    def test_tool_choice_becomes_typed_dict(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
            tool_choice="auto",
        )
        assert kwargs["tool_choice"] == {"type": "auto"}

    def test_stop_sequences_passed_through(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
            stop_sequences=["END", "STOP"],
        )
        assert kwargs["stop_sequences"] == ["END", "STOP"]

    def test_strips_responses_only_kwargs(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
            instructions="should be stripped",
        )
        assert "instructions" not in kwargs


# ---------------------------------------------------------------------------
# NIA ConversationMessage conversion path
# ---------------------------------------------------------------------------

class TestConvertConversationMessagesToAnthropic:
    def test_simple_user_message(self):
        # Use plain dicts in NIA shape (avoid pydantic dep in test).
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        system, result = convert_conversation_messages_to_anthropic(messages)
        assert system is None
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_assistant_with_tool_use(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "SF"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"},
            ]},
        ]
        system, result = convert_conversation_messages_to_anthropic(messages)
        # Should produce 3 messages: user, assistant (with tool_use), user (with tool_result)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        # Tool use should be preserved as a block
        tool_use_blocks = [
            b for b in result[1]["content"] if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        assert len(tool_use_blocks) == 1
        # Tool result should be a user message with tool_result block
        assert result[2]["role"] == "user"
        tool_result_blocks = [
            b for b in result[2]["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        assert len(tool_result_blocks) == 1

    def test_injects_system_prompt(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        system, result = convert_conversation_messages_to_anthropic(
            messages, system_prompt="You are NIA."
        )
        assert system == "You are NIA."
        assert len(result) == 1

    def test_strips_orphaned_tool_use(self):
        # Assistant has a tool_use but no following tool_result.
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "do thing"}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "calling tool"},
                {"type": "tool_use", "id": "orphan", "name": "t", "input": {}},
            ]},
            {"role": "user", "content": [{"type": "text", "text": "next"}]},
        ]
        system, result = convert_conversation_messages_to_anthropic(messages)
        # The orphaned tool_use should be stripped
        for msg in result:
            if msg["role"] == "assistant":
                for b in msg["content"]:
                    if isinstance(b, dict):
                        assert not (b.get("type") == "tool_use" and b.get("id") == "orphan")


# ---------------------------------------------------------------------------
# Top-level convert_messages_to_anthropic (OpenAI format)
# ---------------------------------------------------------------------------

class TestConvertMessagesToAnthropic:
    def test_extracts_system_message(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        system, result = convert_messages_to_anthropic(messages)
        assert system == "You are helpful."
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_system_with_cache_control_becomes_list(self):
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}},
            ]},
            {"role": "user", "content": "hi"},
        ]
        system, result = convert_messages_to_anthropic(messages)
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_runs_repair_pipeline(self):
        # Two consecutive user messages should be merged.
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "world"},
        ]
        system, result = convert_messages_to_anthropic(messages)
        assert len(result) == 1
        assert "hello" in result[0]["content"]
        assert "world" in result[0]["content"]


# ---------------------------------------------------------------------------
# Content parts → Anthropic blocks
# ---------------------------------------------------------------------------

class TestContentPartsToAnthropicBlocks:
    def test_extracts_text_and_image(self):
        parts = [
            {"type": "text", "text": "screenshot:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        result = _content_parts_to_anthropic_blocks(parts)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "screenshot:"}
        assert result[1]["type"] == "image"
        assert result[1]["source"]["media_type"] == "image/png"

    def test_drops_empty_text(self):
        parts = [{"type": "text", "text": ""}]
        result = _content_parts_to_anthropic_blocks(parts)
        assert result == []

    def test_drops_non_image_non_text(self):
        parts = [{"type": "audio_url", "audio_url": {"url": "x"}}]
        result = _content_parts_to_anthropic_blocks(parts)
        assert result == []

    def test_returns_empty_for_non_list(self):
        assert _content_parts_to_anthropic_blocks("not a list") == []
        assert _content_parts_to_anthropic_blocks(None) == []


# ---------------------------------------------------------------------------
# _convert_content_to_anthropic
# ---------------------------------------------------------------------------

class TestConvertContentToAnthropic:
    def test_passthrough_for_non_list(self):
        assert _convert_content_to_anthropic("hello") == "hello"
        assert _convert_content_to_anthropic(None) is None

    def test_converts_list_of_parts(self):
        result = _convert_content_to_anthropic([
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ])
        assert len(result) == 2

    def test_skips_none_parts(self):
        result = _convert_content_to_anthropic([
            {"type": "text", "text": "a"},
            None,
        ])
        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
