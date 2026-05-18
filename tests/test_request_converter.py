import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("OPENAI_BASE_URL", "https://router.z.ai/api/v1")

from src.conversion.request_converter import (
    convert_claude_to_openai,
    normalize_messages_for_text_only_provider,
)
from src.core.config import config
from src.core.model_manager import model_manager
from src.models.claude import ClaudeMessagesRequest


def test_posthog_log_payload_uses_otlp_shape():
    from src.core.analytics import build_posthog_log_payload

    payload = build_posthog_log_payload(
        severity_text="ERROR",
        message="upstream failed",
        attributes={"http.status_code": 400, "request_id": "req-1"},
    )

    record = payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    assert record["severityText"] == "ERROR"
    assert record["body"] == {"stringValue": "upstream failed"}
    assert {"key": "http.status_code", "value": {"intValue": 400}} in record["attributes"]
    assert {"key": "request_id", "value": {"stringValue": "req-1"}} in record["attributes"]


def test_zai_compat_flattens_multimodal_user_content_to_text():
    original = config.flatten_multimodal_content
    config.flatten_multimodal_content = True
    try:
        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read this screenshot."},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc123",
                            },
                        },
                    ],
                }
            ],
        )

        converted = convert_claude_to_openai(request, model_manager)

        assert (
            converted["messages"][0]["content"]
            == "Read this screenshot.\n\n[Image omitted: image/png]"
        )
    finally:
        config.flatten_multimodal_content = original


def test_standard_compat_keeps_multimodal_user_content_parts():
    original = config.flatten_multimodal_content
    config.flatten_multimodal_content = False
    try:
        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read this screenshot."},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc123",
                            },
                        },
                    ],
                }
            ],
        )

        converted = convert_claude_to_openai(request, model_manager)

        assert converted["messages"][0]["content"] == [
            {"type": "text", "text": "Read this screenshot."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc123"},
            },
        ]
    finally:
        config.flatten_multimodal_content = original


def test_text_only_provider_normalizes_all_message_content_arrays():
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": {"text": "hello"}}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "I will call a tool"}],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_123",
            "content": [{"type": "text", "text": "tool result"}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
            ],
        },
    ]

    normalize_messages_for_text_only_provider(messages)

    assert messages[0]["content"] == "hello"
    assert messages[1]["content"] == "I will call a tool"
    assert messages[2]["content"] == "tool result"
    assert messages[3]["content"] is None
    assert all(not isinstance(message.get("content"), list) for message in messages)


def test_zai_compat_textualizes_tool_history():
    original = config.flatten_multimodal_content
    config.flatten_multimodal_content = True
    try:
        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-6",
            max_tokens=100,
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            messages=[
                {"role": "user", "content": "What is the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": [{"type": "text", "text": "Sunny"}],
                        }
                    ],
                },
            ],
        )

        converted = convert_claude_to_openai(request, model_manager)

        assert [message["role"] for message in converted["messages"]] == [
            "user",
            "assistant",
            "user",
        ]
        assert "tool_calls" not in converted["messages"][1]
        assert "get_weather" in converted["messages"][1]["content"]
        assert "toolu_123" in converted["messages"][2]["content"]
        assert "Sunny" in converted["messages"][2]["content"]
        assert converted["tools"][0]["type"] == "function"
    finally:
        config.flatten_multimodal_content = original
