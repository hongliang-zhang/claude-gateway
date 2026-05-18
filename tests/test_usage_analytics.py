import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

from src.api.endpoints import extract_usage_metrics


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
