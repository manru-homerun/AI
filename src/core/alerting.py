from __future__ import annotations

import contextvars
import datetime as dt
import json
import os
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any
from urllib.request import Request, urlopen

from src.core.logging import DEFAULT_SERVICE_NAME, get_logger, get_request_id


DISCORD_WEBHOOK_URL_ENV = "DISCORD_WEBHOOK_URL"
DISCORD_TIMEOUT_SECONDS = 3.0

_logger = get_logger(__name__)
_request_alerted: contextvars.ContextVar[bool] = contextvars.ContextVar("request_alerted", default=False)
_executor: ThreadPoolExecutor | None = None


def begin_alert_context() -> contextvars.Token[bool]:
    return _request_alerted.set(False)


def reset_alert_context(token: contextvars.Token[bool]) -> None:
    _request_alerted.reset(token)


def has_request_alerted() -> bool:
    return _request_alerted.get()


def _executor_instance() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="discord-alert")
    return _executor


def notify_discord(
    event: str,
    severity: str,
    message: str,
    context: Mapping[str, Any] | None = None,
) -> Future[None] | None:
    _request_alerted.set(True)
    webhook_url = os.environ.get(DISCORD_WEBHOOK_URL_ENV)
    if not webhook_url:
        return None

    payload_context = dict(context or {})
    payload_context.setdefault("request_id", get_request_id())
    payload_context.setdefault("service", os.environ.get("SERVICE_NAME", DEFAULT_SERVICE_NAME))
    payload_context.setdefault("timestamp", dt.datetime.now(tz=dt.timezone.utc).isoformat())
    payload = _build_discord_payload(event, severity, message, payload_context)
    return _executor_instance().submit(_deliver_discord_alert, webhook_url, payload, event)


def _build_discord_payload(event: str, severity: str, message: str, context: Mapping[str, Any]) -> dict[str, Any]:
    title = f"[{severity}] Travel AI Server Error"
    fields = [
        ("Event", event),
        ("Endpoint", _endpoint_label(context)),
        ("Request ID", context.get("request_id", "-")),
        ("Runtime", context.get("runtime", "-")),
        ("Fallback Reason", context.get("fallback_reason", "-")),
        ("Error Type", context.get("error_type", "-")),
        ("Timestamp", context.get("timestamp", "-")),
    ]
    field_lines = "\n".join(f"**{name}:** {_format_alert_value(value)}" for name, value in fields)
    content = f"{title}\n\n{field_lines}\n**Message:** {_format_alert_value(message)}"
    return {"content": content[:2000]}


def _endpoint_label(context: Mapping[str, Any]) -> str:
    endpoint = context.get("endpoint", "-")
    method = context.get("method")
    if method and endpoint and endpoint != "-":
        return f"{method} {endpoint}"
    return str(endpoint)


def _format_alert_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def _deliver_discord_alert(webhook_url: str, payload: Mapping[str, Any], event: str) -> None:
    try:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=DISCORD_TIMEOUT_SECONDS):
            return
    except Exception as exc:
        _logger.warning(
            "discord alert delivery failed",
            extra={
                "event": "discord_alert_delivery_failure",
                "alert": False,
                "alert_severity": "-",
                "error_type": type(exc).__name__,
                "source_event": event,
            },
        )
