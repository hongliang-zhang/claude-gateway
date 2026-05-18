import asyncio
import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
from fastapi import Request, Response

from src.core.config import config
from src.core.logging import logger


def hash_identifier(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def extract_client_api_key(headers: Dict[str, str]) -> Optional[str]:
    api_key = headers.get("x-api-key")
    authorization = headers.get("authorization")
    if api_key:
        return api_key
    if authorization and authorization.startswith("Bearer "):
        return authorization.replace("Bearer ", "", 1)
    return None


def _client_ip(request: Request) -> Optional[str]:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    if request.client:
        return request.client.host
    return None


async def _send_posthog_event(event: str, distinct_id: str, properties: Dict[str, Any]) -> None:
    if not config.posthog_api_key:
        return

    payload = {
        "api_key": config.posthog_api_key,
        "event": event,
        "distinct_id": distinct_id,
        "properties": properties,
    }

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(f"{config.posthog_host}/i/v0/e/", json=payload)
            response.raise_for_status()
    except Exception as exc:
        logger.warning(f"PostHog capture failed: {exc}")


def capture_posthog_event(event: str, distinct_id: str, properties: Dict[str, Any]) -> None:
    if not config.posthog_api_key:
        return
    asyncio.create_task(_send_posthog_event(event, distinct_id, properties))


def _otlp_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": value}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": "" if value is None else str(value)}


def build_posthog_log_payload(
    severity_text: str,
    message: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now_ns = str(int(datetime.now(timezone.utc).timestamp() * 1_000_000_000))
    resource_attributes = [
        {
            "key": "service.name",
            "value": {"stringValue": config.posthog_service_name},
        }
    ]
    log_attributes = [
        {"key": key, "value": _otlp_value(value)}
        for key, value in (attributes or {}).items()
        if value is not None
    ]

    return {
        "resourceLogs": [
            {
                "resource": {"attributes": resource_attributes},
                "scopeLogs": [
                    {
                        "scope": {"name": "claude-code-proxy"},
                        "logRecords": [
                            {
                                "timeUnixNano": now_ns,
                                "severityText": severity_text,
                                "body": {"stringValue": message},
                                "attributes": log_attributes,
                            }
                        ],
                    }
                ],
            }
        ]
    }


async def _send_posthog_log(
    severity_text: str,
    message: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    if not config.posthog_api_key or not config.posthog_logs_enabled:
        return

    payload = build_posthog_log_payload(severity_text, message, attributes)
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(
                f"{config.posthog_host}/i/v1/logs",
                json=payload,
                headers={"Authorization": f"Bearer {config.posthog_api_key}"},
            )
            response.raise_for_status()
    except Exception as exc:
        logger.warning(f"PostHog log capture failed: {exc}")


def capture_posthog_log(
    severity_text: str,
    message: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    if not config.posthog_api_key or not config.posthog_logs_enabled:
        return
    asyncio.create_task(_send_posthog_log(severity_text, message, attributes))


async def posthog_traffic_middleware(request: Request, call_next):
    started_at = time.perf_counter()
    status_code = 500
    response: Optional[Response] = None
    error_type: Optional[str] = None

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        client_key_hash = hash_identifier(extract_client_api_key(dict(request.headers)))
        distinct_id = client_key_hash or hash_identifier(_client_ip(request)) or "anonymous"

        capture_posthog_event(
            "gateway request",
            distinct_id,
            {
                "$process_person_profile": False,
                "method": request.method,
                "path": request.url.path,
                "query": request.url.query,
                "status_code": status_code,
                "duration_ms": elapsed_ms,
                "client_key_hash": client_key_hash,
                "client_ip_hash": hash_identifier(_client_ip(request)),
                "user_agent": request.headers.get("user-agent"),
                "error_type": error_type,
            },
        )
