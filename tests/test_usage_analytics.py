import asyncio
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

from src.api.endpoints import extract_usage_metrics, validate_api_key
from src.core.config import config
from src.core.constants import Constants
from src.conversion.response_converter import (
    convert_openai_streaming_to_claude,
    convert_openai_to_claude_response,
    extract_textual_tool_call,
    parse_textual_tool_call,
    split_streamable_text,
    strip_internal_tool_results,
)
from src.models.claude import ClaudeMessagesRequest


def test_extract_usage_metrics_includes_cache_read_tokens():
    usage = {
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "prompt_tokens_details": {"cached_tokens": 80},
    }

    assert extract_usage_metrics(usage) == {
        "input_tokens": 120,
        "output_tokens": 45,
        "cache_read_input_tokens": 80,
    }


def test_extract_usage_metrics_defaults_cache_read_tokens_to_zero():
    usage = {"prompt_tokens": 120, "completion_tokens": 45}

    assert extract_usage_metrics(usage) == {
        "input_tokens": 120,
        "output_tokens": 45,
        "cache_read_input_tokens": 0,
    }


def test_validate_api_key_accepts_allowed_authorization_when_x_api_key_is_stale(monkeypatch):
    monkeypatch.setattr(config, "anthropic_api_key", "allowed-gateway-key")
    monkeypatch.setattr(config, "anthropic_api_keys", [])

    asyncio.run(
        validate_api_key(
            x_api_key="stale-sdk-key",
            authorization="Bearer allowed-gateway-key",
        )
    )


def test_parse_textual_tool_call():
    parsed = parse_textual_tool_call(
        '[Tool call id=fc_toolu_123 name=mcp__pencil__get_editor_state input={"include_schema": true}]'
    )

    assert parsed == {
        "id": "fc_toolu_123",
        "name": "mcp__pencil__get_editor_state",
        "input": {"include_schema": True},
    }


def test_extract_textual_tool_call_strips_echoed_tool_result_prefix():
    extracted = extract_textual_tool_call(
        "[Tool result for id=fc_toolu_abc]\n"
        "Active file: /tmp/design.pen\n"
        "<schema>internal schema docs</schema>\n\n"
        "[Tool call id=fc_toolu_123 name=mcp__pencil__get_guidelines input={}]"
    )

    assert extracted == {
        "prefix": "",
        "tool_call": {
            "id": "fc_toolu_123",
            "name": "mcp__pencil__get_guidelines",
            "input": {},
        },
    }


def test_extract_antml_tool_call():
    extracted = extract_textual_tool_call(
        "让我先检查一下正确的 text node 属性格式。\n"
        "[antml:function_calls>\n"
        '<invoke name="mcp__pencil__batch_get">\n'
        '<parameter name="filePath">/Users/zhanghongliang/Documents/agent工程自迭代实验/design.pen</parameter>\n'
        '<parameter name="nodeIds">["bi8Au"]</parameter>\n'
        '<parameter name="readDepth">3</parameter>\n'
        "</invoke>\n"
        "</antml:function_calls>"
    )

    assert extracted["prefix"] == "让我先检查一下正确的 text node 属性格式。"
    assert extracted["tool_call"]["id"].startswith("toolu_")
    assert extracted["tool_call"]["name"] == "mcp__pencil__batch_get"
    assert extracted["tool_call"]["input"] == {
        "filePath": "/Users/zhanghongliang/Documents/agent工程自迭代实验/design.pen",
        "nodeIds": ["bi8Au"],
        "readDepth": 3,
    }


def test_strip_internal_tool_results_hides_schema_echo():
    assert (
        strip_internal_tool_results(
            "[Tool result for id=fc_toolu_abc]\n<schema>internal schema docs</schema>"
        )
        == ""
    )


def test_split_streamable_text_streams_normal_text_immediately():
    streamable, pending = split_streamable_text("普通回答内容")

    assert streamable == "普通回答内容"
    assert pending == ""


def test_split_streamable_text_holds_tool_markup_suffix():
    streamable, pending = split_streamable_text("先说一句话。[antml:function_calls>")

    assert streamable == "先说一句话。"
    assert pending == "[antml:function_calls>"


def test_split_streamable_text_holds_partial_marker_suffix():
    streamable, pending = split_streamable_text("先说一句话。[antml:fun")

    assert streamable == "先说一句话。"
    assert pending == "[antml:fun"


def test_non_streaming_response_converts_textual_tool_call_to_tool_use():
    request = ClaudeMessagesRequest(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "Check editor state"}],
    )
    response = convert_openai_to_claude_response(
        {
            "id": "chatcmpl_1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '[Tool call id=fc_toolu_123 name=mcp__pencil__get_editor_state input={"include_schema": true}]',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 12},
        },
        request,
    )

    assert response["stop_reason"] == "tool_use"
    assert response["content"] == [
        {
            "type": "tool_use",
            "id": "fc_toolu_123",
            "name": "mcp__pencil__get_editor_state",
            "input": {"include_schema": True},
        }
    ]


def test_non_streaming_response_maps_cached_tokens_to_cache_read_input_tokens():
    request = ClaudeMessagesRequest(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello"}],
    )
    response = convert_openai_to_claude_response(
        {
            "id": "chatcmpl_1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 800},
            },
        },
        request,
    )

    assert response["usage"] == {
        "input_tokens": 1200,
        "output_tokens": 20,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 800,
    }


def test_non_streaming_response_hides_tool_result_prefix_before_textual_tool_call():
    request = ClaudeMessagesRequest(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "Create a diagram"}],
    )
    response = convert_openai_to_claude_response(
        {
            "id": "chatcmpl_1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "[Tool result for id=fc_toolu_abc]\n"
                            "Active file: /tmp/design.pen\n"
                            "<schema>internal schema docs</schema>\n\n"
                            "[Tool call id=fc_toolu_123 "
                            "name=mcp__pencil__get_guidelines input={}]"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 12},
        },
        request,
    )

    assert response["stop_reason"] == "tool_use"
    assert response["content"] == [
        {
            "type": "tool_use",
            "id": "fc_toolu_123",
            "name": "mcp__pencil__get_guidelines",
            "input": {},
        }
    ]


async def test_streaming_response_converts_textual_tool_call_to_tool_use():
    request = ClaudeMessagesRequest(
        model="claude-sonnet-4-6",
        max_tokens=100,
        stream=True,
        messages=[{"role": "user", "content": "Check editor state"}],
    )

    async def fake_stream():
        yield (
            'data: {"choices":[{"delta":{"content":"[Tool call id=fc_toolu_123 "},'
            '"finish_reason":null}]}'
        )
        yield (
            'data: {"choices":[{"delta":{"content":"name=mcp__pencil__get_editor_state '
            'input={\\"include_schema\\": true}]"},"finish_reason":"stop"}]}'
        )
        yield "data: [DONE]"

    class FakeLogger:
        def warning(self, *_args, **_kwargs):
            pass

        def error(self, *_args, **_kwargs):
            pass

    events = [
        event
        async for event in convert_openai_streaming_to_claude(fake_stream(), request, FakeLogger())
    ]
    payload = "".join(events)

    assert '"type": "tool_use"' in payload
    assert '"name": "mcp__pencil__get_editor_state"' in payload
    assert '\\"include_schema\\": true' in payload
    assert '"stop_reason": "tool_use"' in payload
    assert "Tool call id=fc_toolu_123" not in payload


async def test_streaming_response_streams_normal_text_before_done():
    request = ClaudeMessagesRequest(
        model="claude-sonnet-4-6",
        max_tokens=100,
        stream=True,
        messages=[{"role": "user", "content": "Say hello"}],
    )

    async def fake_stream():
        yield 'data: {"choices":[{"delta":{"content":"你好，"}, "finish_reason":null}]}'
        yield 'data: {"choices":[{"delta":{"content":"马上开始。"}, "finish_reason":null}]}'
        yield 'data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}'
        yield "data: [DONE]"

    class FakeLogger:
        def warning(self, *_args, **_kwargs):
            pass

        def error(self, *_args, **_kwargs):
            pass

    events = [
        event
        async for event in convert_openai_streaming_to_claude(fake_stream(), request, FakeLogger())
    ]

    first_text_index = next(index for index, event in enumerate(events) if "你好，" in event)
    stop_index = next(
        index for index, event in enumerate(events) if Constants.EVENT_MESSAGE_STOP in event
    )

    assert first_text_index < stop_index


async def test_streaming_response_converts_antml_tool_call_to_tool_use():
    request = ClaudeMessagesRequest(
        model="claude-sonnet-4-6",
        max_tokens=100,
        stream=True,
        messages=[{"role": "user", "content": "Check text node"}],
    )

    async def fake_stream():
        yield (
            'data: {"choices":[{"delta":{"content":"让我先检查一下正确的 text node 属性格式。\\n'
            '[antml:function_calls>\\n<invoke name=\\"mcp__pencil__batch_get\\">\\n"},'
            '"finish_reason":null}]}'
        )
        yield (
            'data: {"choices":[{"delta":{"content":"<parameter name=\\"filePath\\">/tmp/design.pen</parameter>\\n'
            '<parameter name=\\"nodeIds\\">[\\"bi8Au\\"]</parameter>\\n'
            '<parameter name=\\"readDepth\\">3</parameter>\\n</invoke>\\n</antml:function_calls>"},'
            '"finish_reason":"stop"}]}'
        )
        yield "data: [DONE]"

    class FakeLogger:
        def warning(self, *_args, **_kwargs):
            pass

        def error(self, *_args, **_kwargs):
            pass

    events = [
        event
        async for event in convert_openai_streaming_to_claude(fake_stream(), request, FakeLogger())
    ]
    payload = "".join(events)

    assert '"type": "tool_use"' in payload
    assert '"name": "mcp__pencil__batch_get"' in payload
    assert '\\"nodeIds\\": [\\"bi8Au\\"]' in payload
    assert '\\"readDepth\\": 3' in payload
    assert '"stop_reason": "tool_use"' in payload
    assert "antml:function_calls" not in payload
