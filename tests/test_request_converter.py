import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("OPENAI_BASE_URL", "https://router.z.ai/api/v1")

from src.conversion.request_converter import convert_claude_to_openai
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

        assert converted["messages"][0]["content"] == "Read this screenshot.\n\n[Image omitted: image/png]"
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
