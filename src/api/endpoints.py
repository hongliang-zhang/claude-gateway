from fastapi import APIRouter, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime
import uuid
from typing import Any, Dict, Optional

from src.core.config import config
from src.core.analytics import (
    capture_posthog_log,
    capture_posthog_event,
    extract_client_api_key,
    hash_identifier,
)
from src.core.logging import logger
from src.core.client import OpenAIClient
from src.models.claude import ClaudeMessagesRequest, ClaudeTokenCountRequest
from src.conversion.request_converter import convert_claude_to_openai
from src.conversion.response_converter import (
    convert_openai_to_claude_response,
    convert_openai_streaming_to_claude_with_cancellation,
)
from src.core.model_manager import model_manager

router = APIRouter()

# Get custom headers from config
custom_headers = config.get_custom_headers()


def build_openai_client(api_key: str) -> OpenAIClient:
    return OpenAIClient(
        api_key,
        config.openai_base_url,
        config.request_timeout,
        api_version=config.azure_api_version,
        custom_headers=custom_headers,
    )


def classify_openai_error(error_detail: Any) -> str:
    api_key = config.openai_api_key or "unused"
    return build_openai_client(api_key).classify_openai_error(error_detail)


def extract_authorization_api_key(
    x_api_key: Optional[str], authorization: Optional[str]
) -> Optional[str]:
    if authorization and authorization.startswith("Bearer "):
        return authorization.replace("Bearer ", "", 1)
    return x_api_key


def capture_gateway_error_log(
    http_request: Request,
    request: ClaudeMessagesRequest,
    request_id: str,
    status_code: int,
    message: str,
) -> None:
    client_key_hash = hash_identifier(extract_client_api_key(dict(http_request.headers)))
    capture_posthog_log(
        "ERROR",
        message,
        {
            "request_id": request_id,
            "http.method": http_request.method,
            "http.route": http_request.url.path,
            "http.status_code": status_code,
            "client_key_hash": client_key_hash,
            "model": request.model,
            "stream": bool(request.stream),
            "message_count": len(request.messages),
            "tool_count": len(request.tools or []),
        },
    )


def extract_usage_metrics(usage: Dict[str, Any]) -> Dict[str, int]:
    prompt_tokens_details = usage.get("prompt_tokens_details") or {}
    return {
        "input_tokens": usage.get("prompt_tokens", 0) or 0,
        "output_tokens": usage.get("completion_tokens", 0) or 0,
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": (
            usage.get("cache_read_input_tokens")
            or prompt_tokens_details.get("cached_tokens", 0)
            or 0
        ),
    }


def describe_openai_message_shapes(openai_request: Dict[str, Any]) -> list[Dict[str, Any]]:
    shapes = []
    for index, message in enumerate(openai_request.get("messages", [])):
        content = message.get("content")
        shape: Dict[str, Any] = {
            "index": index,
            "role": message.get("role"),
            "content_type": type(content).__name__,
            "has_tool_calls": bool(message.get("tool_calls")),
        }
        if isinstance(content, list):
            shape["parts"] = [
                {
                    "index": part_index,
                    "type": part.get("type") if isinstance(part, dict) else type(part).__name__,
                    "text_type": (
                        type(part.get("text")).__name__
                        if isinstance(part, dict) and "text" in part
                        else None
                    ),
                }
                for part_index, part in enumerate(content)
            ]
        elif isinstance(content, str):
            shape["content_length"] = len(content)
        shapes.append(shape)
    return shapes


def capture_gateway_completion_event(
    http_request: Request,
    request: ClaudeMessagesRequest,
    openai_model: Optional[str],
    stop_reason: Optional[str],
    usage_metrics: Dict[str, int],
) -> None:
    client_key_hash = hash_identifier(extract_client_api_key(dict(http_request.headers)))
    capture_posthog_event(
        "gateway completion",
        client_key_hash or "anonymous",
        {
            "$process_person_profile": False,
            "model": request.model,
            "upstream_model": openai_model,
            "stream": bool(request.stream),
            "message_count": len(request.messages),
            "tool_count": len(request.tools or []),
            "stop_reason": stop_reason,
            "input_tokens": usage_metrics.get("input_tokens", 0),
            "output_tokens": usage_metrics.get("output_tokens", 0),
            "cache_creation_input_tokens": usage_metrics.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage_metrics.get("cache_read_input_tokens", 0),
            "client_key_hash": client_key_hash,
        },
    )


async def validate_api_key(
    x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)
) -> str:
    """Validate the client's API key from either x-api-key header or Authorization header."""
    if config.gateway_auth_mode == "pass_through":
        user_api_key = extract_authorization_api_key(x_api_key, authorization)
        if not user_api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Provide your Z.AI API key as the Anthropic API key.",
            )
        return user_api_key

    client_api_keys = []
    if x_api_key:
        client_api_keys.append(x_api_key)
    if authorization and authorization.startswith("Bearer "):
        client_api_keys.append(authorization.replace("Bearer ", "", 1))

    # Skip validation if no client API key allowlist is set in the environment
    if not config.anthropic_api_key and not config.anthropic_api_keys:
        return config.openai_api_key

    # Some clients send both auth headers. Accept either one if it matches the allowlist.
    if not any(config.validate_client_api_key(key) for key in client_api_keys):
        logger.warning(f"Invalid API key provided by client")
        raise HTTPException(
            status_code=401, detail="Invalid API key. Please provide a valid Anthropic API key."
        )
    return config.openai_api_key


@router.post("/v1/messages")
async def create_message(
    request: ClaudeMessagesRequest,
    http_request: Request,
    upstream_api_key: str = Depends(validate_api_key),
):
    try:
        logger.debug(f"Processing Claude request: model={request.model}, stream={request.stream}")

        # Generate unique request ID for cancellation tracking
        request_id = str(uuid.uuid4())
        request_openai_client = build_openai_client(upstream_api_key)

        # Convert Claude request to OpenAI format
        openai_request = convert_claude_to_openai(request, model_manager)
        logger.info(
            "Converted OpenAI request shape: "
            f"flatten={config.flatten_multimodal_content}, "
            f"stream={request.stream}, "
            f"messages={describe_openai_message_shapes(openai_request)}"
        )

        # Check if client disconnected before processing
        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if request.stream:
            # Streaming response - wrap in error handling
            try:
                openai_stream = request_openai_client.create_chat_completion_stream(
                    openai_request, request_id
                )
                return StreamingResponse(
                    convert_openai_streaming_to_claude_with_cancellation(
                        openai_stream,
                        request,
                        logger,
                        http_request,
                        request_openai_client,
                        request_id,
                        on_complete=lambda usage, stop_reason, openai_model: capture_gateway_completion_event(
                            http_request,
                            request,
                            openai_model,
                            stop_reason,
                            usage,
                        ),
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            except HTTPException as e:
                # Convert to proper error response for streaming
                logger.error(f"Streaming error: {e.detail}")
                capture_gateway_error_log(
                    http_request,
                    request,
                    request_id,
                    e.status_code,
                    str(e.detail),
                )
                import traceback

                logger.error(traceback.format_exc())
                error_message = request_openai_client.classify_openai_error(e.detail)
                error_response = {
                    "type": "error",
                    "error": {"type": "api_error", "message": error_message},
                }
                return JSONResponse(status_code=e.status_code, content=error_response)
        else:
            # Non-streaming response
            openai_response = await request_openai_client.create_chat_completion(
                openai_request, request_id
            )
            claude_response = convert_openai_to_claude_response(openai_response, request)
            capture_gateway_completion_event(
                http_request,
                request,
                openai_response.get("model"),
                claude_response.get("stop_reason"),
                extract_usage_metrics(openai_response.get("usage", {})),
            )
            return claude_response
    except HTTPException as e:
        request_id_for_log = locals().get("request_id", "unassigned")
        capture_gateway_error_log(
            http_request,
            request,
            request_id_for_log,
            e.status_code,
            str(e.detail),
        )
        raise
    except Exception as e:
        import traceback

        logger.error(f"Unexpected error processing request: {e}")
        logger.error(traceback.format_exc())
        error_message = classify_openai_error(str(e))
        request_id_for_log = locals().get("request_id", "unassigned")
        capture_gateway_error_log(
            http_request,
            request,
            request_id_for_log,
            500,
            error_message,
        )
        raise HTTPException(status_code=500, detail=error_message)


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request: ClaudeTokenCountRequest, http_request: Request, _: str = Depends(validate_api_key)
):
    try:
        # For token counting, we'll use a simple estimation
        # In a real implementation, you might want to use tiktoken or similar

        total_chars = 0

        # Count system message characters
        if request.system:
            if isinstance(request.system, str):
                total_chars += len(request.system)
            elif isinstance(request.system, list):
                for block in request.system:
                    if hasattr(block, "text"):
                        total_chars += len(block.text)

        # Count message characters
        for msg in request.messages:
            if msg.content is None:
                continue
            elif isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text") and block.text is not None:
                        total_chars += len(block.text)

        # Rough estimation: 4 characters per token
        estimated_tokens = max(1, total_chars // 4)
        client_key_hash = hash_identifier(extract_client_api_key(dict(http_request.headers)))
        capture_posthog_event(
            "gateway token count",
            client_key_hash or "anonymous",
            {
                "$process_person_profile": False,
                "model": request.model,
                "estimated_input_tokens": estimated_tokens,
                "message_count": len(request.messages),
                "client_key_hash": client_key_hash,
            },
        )

        return {"input_tokens": estimated_tokens}

    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "auth_mode": config.gateway_auth_mode,
        "server_api_key_configured": bool(config.openai_api_key),
        "server_api_key_valid": config.validate_api_key() if config.openai_api_key else None,
        "client_api_key_validation": (
            config.gateway_auth_mode == "pass_through" or bool(config.anthropic_api_key)
        ),
    }


@router.get("/test-connection")
async def test_connection(upstream_api_key: str = Depends(validate_api_key)):
    """Test API connectivity to OpenAI"""
    try:
        request_openai_client = build_openai_client(upstream_api_key)
        # Simple test request to verify API connectivity
        test_response = await request_openai_client.create_chat_completion(
            {
                "model": config.small_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            }
        )

        return {
            "status": "success",
            "message": "Successfully connected to OpenAI API",
            "model_used": config.small_model,
            "timestamp": datetime.now().isoformat(),
            "response_id": test_response.get("id", "unknown"),
        }

    except Exception as e:
        logger.error(f"API connectivity test failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error_type": "API Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
                "suggestions": [
                    "Check the provided API key is valid",
                    "Verify the API key has the necessary permissions",
                    "Check if you have reached rate limits",
                ],
            },
        )


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Claude-to-OpenAI API Proxy v1.0.0",
        "status": "running",
        "config": {
            "openai_base_url": config.openai_base_url,
            "auth_mode": config.gateway_auth_mode,
            "max_tokens_limit": config.max_tokens_limit,
            "server_api_key_configured": bool(config.openai_api_key),
            "client_api_key_validation": (
                config.gateway_auth_mode == "pass_through" or bool(config.anthropic_api_key)
            ),
            "big_model": config.big_model,
            "small_model": config.small_model,
        },
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/health",
            "test_connection": "/test-connection",
        },
    }
