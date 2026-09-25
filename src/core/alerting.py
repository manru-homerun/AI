from __future__ import annotations

import contextvars
import datetime as dt
import json
import os
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.core.logging import DEFAULT_SERVICE_NAME, get_logger, get_request_id


DISCORD_WEBHOOK_URL_ENV = "DISCORD_WEBHOOK_URL"
DISCORD_TIMEOUT_SECONDS = 3.0
DISCORD_ALERT_DEDUPE_TTL_SECONDS = 300.0
DISCORD_USER_AGENT = "travel-ai-alerting/1.0"
SEVERITY_RANKS = {
    "HIGH": 10,
    "CRITICAL": 20,
}
EVENT_LABELS = {
    "runtime_initialization_failure": "런타임 초기화 실패",
    "model_unavailable": "모델 사용 불가",
    "recommend_inference_failure": "추천 모델 추론 실패",
    "course_inference_failure": "코스 생성 모델 추론 실패",
    "recommend_fallback_failure": "추천 fallback 생성 실패",
    "course_fallback_failure": "코스 fallback 생성 실패",
    "request_failed": "처리되지 않은 서버 오류",
}
EVENT_DESCRIPTIONS = {
    "runtime_initialization_failure": "모델 런타임 초기화에 실패했습니다. artifact 경로 또는 필수 파일을 확인하세요.",
    "model_unavailable": "모델 런타임이 로드되지 않아 fallback 응답을 사용 중입니다.",
    "recommend_inference_failure": "추천 모델 추론 중 오류가 발생했습니다. fallback 추천으로 복구를 시도했습니다.",
    "course_inference_failure": "코스 생성 모델 추론 중 오류가 발생했습니다. fallback 코스로 복구를 시도했습니다.",
    "recommend_fallback_failure": "추천 fallback 생성까지 실패했습니다. 사용자 요청이 500으로 실패할 수 있습니다.",
    "course_fallback_failure": "코스 fallback 생성까지 실패했습니다. 사용자 요청이 500으로 실패할 수 있습니다.",
    "request_failed": "처리되지 않은 서버 오류입니다. request_id로 서버 로그를 확인하세요.",
}
ERROR_TYPE_LABELS = {
    "FileNotFoundError": "필요한 파일을 찾을 수 없음",
    "RuntimeError": "런타임 처리 오류",
    "HTTPError": "외부 HTTP 요청 오류",
}
FALLBACK_REASON_LABELS = {
    "inference_error": "모델 추론 중 오류",
    "model_unavailable": "모델 사용 불가",
    "unexpected_error": "예상하지 못한 fallback 실패",
}

_logger = get_logger(__name__)
_request_alert_severity_rank: contextvars.ContextVar[int] = contextvars.ContextVar(
    "request_alert_severity_rank",
    default=0,
)
_dedupe_lock = threading.Lock()
_dedupe_expires_at: dict[tuple[str, str, str], float] = {}
_executor: ThreadPoolExecutor | None = None


def begin_alert_context() -> contextvars.Token[int]:
    return _request_alert_severity_rank.set(0)


def reset_alert_context(token: contextvars.Token[int]) -> None:
    _request_alert_severity_rank.reset(token)


def has_request_alerted() -> bool:
    return _request_alert_severity_rank.get() > 0


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
    normalized_severity = severity.upper()
    severity_rank = SEVERITY_RANKS.get(normalized_severity, 0)
    payload_context = dict(context or {})
    payload_context.setdefault("request_id", get_request_id())

    if _is_request_suppressed(event, normalized_severity, severity_rank, payload_context):
        return None

    webhook_url = os.environ.get(DISCORD_WEBHOOK_URL_ENV)
    if not webhook_url:
        return None

    dedupe_key = _dedupe_key(event, payload_context)
    now = _monotonic()
    if _is_deduplicated(dedupe_key, now):
        _log_suppressed(
            event=event,
            severity=normalized_severity,
            context=payload_context,
            suppression_reason="deduplicated",
        )
        return None

    payload_context.setdefault("service", os.environ.get("SERVICE_NAME", DEFAULT_SERVICE_NAME))
    payload_context.setdefault("timestamp", dt.datetime.now(tz=dt.timezone.utc).isoformat())
    payload = _build_discord_payload(event, normalized_severity, message, payload_context)
    return _executor_instance().submit(_deliver_discord_alert, webhook_url, payload, event, dedupe_key)


def _is_request_suppressed(
    event: str,
    severity: str,
    severity_rank: int,
    context: Mapping[str, Any],
) -> bool:
    current_rank = _request_alert_severity_rank.get()
    if severity_rank <= current_rank:
        _log_suppressed(
            event=event,
            severity=severity,
            context=context,
            suppression_reason="request_severity",
        )
        return True
    _request_alert_severity_rank.set(severity_rank)
    return False


def _dedupe_key(event: str, context: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _normalize_dedupe_value(event),
        _normalize_dedupe_value(context.get("error_type")),
        _normalize_dedupe_value(context.get("fallback_reason")),
    )


def _normalize_dedupe_value(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return text or "-"


def _is_deduplicated(dedupe_key: tuple[str, str, str], now: float) -> bool:
    with _dedupe_lock:
        _cleanup_expired_locked(now)
        if dedupe_key in _dedupe_expires_at:
            return True
        _dedupe_expires_at[dedupe_key] = now + DISCORD_ALERT_DEDUPE_TTL_SECONDS
        return False


def _release_dedupe_key(dedupe_key: tuple[str, str, str]) -> None:
    with _dedupe_lock:
        _dedupe_expires_at.pop(dedupe_key, None)


def _cleanup_expired_locked(now: float) -> None:
    expired_keys = [key for key, expires_at in _dedupe_expires_at.items() if expires_at <= now]
    for key in expired_keys:
        del _dedupe_expires_at[key]


def _monotonic() -> float:
    return time.monotonic()


def _log_suppressed(
    *,
    event: str,
    severity: str,
    context: Mapping[str, Any],
    suppression_reason: str,
) -> None:
    _logger.debug(
        "discord alert suppressed",
        extra={
            "event": "discord_alert_suppressed",
            "alert": False,
            "alert_severity": severity,
            "alert_event": event,
            "suppression_reason": suppression_reason,
            "error_type": context.get("error_type", "-"),
            "fallback_reason": context.get("fallback_reason", "-"),
        },
    )


def reset_alerting_state_for_tests() -> None:
    _request_alert_severity_rank.set(0)
    with _dedupe_lock:
        _dedupe_expires_at.clear()


def _build_discord_payload(event: str, severity: str, message: str, context: Mapping[str, Any]) -> dict[str, Any]:
    title = f"[{severity}] Travel AI 서버 장애 알림"
    fields = [
        ("장애 유형", _label_with_raw(event, EVENT_LABELS)),
        ("장애 설명", EVENT_DESCRIPTIONS.get(event, "등록되지 않은 장애 유형입니다. 원문 event를 기준으로 서버 로그를 확인하세요.")),
        ("엔드포인트", _endpoint_label(context)),
        ("요청 ID", context.get("request_id", "-")),
        ("런타임", context.get("runtime", "-")),
        ("Fallback 사유", _label_with_raw(context.get("fallback_reason"), FALLBACK_REASON_LABELS)),
        ("예외 타입", _label_with_raw(context.get("error_type"), ERROR_TYPE_LABELS)),
        ("발생 시각", context.get("timestamp", "-")),
        ("요약 메시지", message),
    ]
    field_lines = "\n".join(f"**{name}:** {_format_alert_value(value)}" for name, value in fields)
    content = f"{title}\n\n{field_lines}\n\n**상세 확인:** 서버 로그에서 request_id로 검색하세요."
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


def _label_with_raw(value: Any, labels: Mapping[str, str]) -> str:
    normalized_value = _format_alert_value(value)
    label = labels.get(normalized_value)
    if label is None:
        return normalized_value
    return f"{label} ({normalized_value})"


def _deliver_discord_alert(
    webhook_url: str,
    payload: Mapping[str, Any],
    event: str,
    dedupe_key: tuple[str, str, str],
) -> None:
    try:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            webhook_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": DISCORD_USER_AGENT,
            },
            method="POST",
        )
        with urlopen(request, timeout=DISCORD_TIMEOUT_SECONDS):
            return
    except HTTPError as exc:
        _release_dedupe_key(dedupe_key)
        _logger.warning(
            "discord alert delivery failed",
            extra={
                "event": "discord_alert_delivery_failure",
                "alert": False,
                "alert_severity": "-",
                "error_type": type(exc).__name__,
                "http_status": exc.code,
                "source_event": event,
            },
        )
    except Exception as exc:
        _release_dedupe_key(dedupe_key)
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
