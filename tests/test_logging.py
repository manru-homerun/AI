from __future__ import annotations

import logging
import sys

import pytest
from fastapi import HTTPException

from src.core.logging import COMMON_LOG_FIELDS, KeyValueFormatter, RequestContextFilter, reset_request_id, set_request_id
from src.services import travel_service


def test_key_value_formatter_includes_common_fields_without_request_context() -> None:
    record = logging.LogRecord(
        name="src.tests",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello logging",
        args=(),
        exc_info=None,
    )
    RequestContextFilter("travel-ai").filter(record)

    formatted = KeyValueFormatter().format(record)

    assert "timestamp=" in formatted
    assert "level=INFO" in formatted
    assert "service=travel-ai" in formatted
    assert "logger=src.tests" in formatted
    assert "request_id=-" in formatted
    assert "event=-" in formatted
    assert 'message="hello logging"' in formatted
    assert "endpoint=-" in formatted
    assert "runtime=-" in formatted
    assert "fallback_reason=-" in formatted
    assert "error_type=-" in formatted
    assert "elapsed_ms=-" in formatted
    assert "alert=-" in formatted
    assert "alert_severity=-" in formatted

    field_positions = [formatted.index(f"{field_name}=") for field_name in COMMON_LOG_FIELDS]
    assert field_positions == sorted(field_positions)


def test_key_value_formatter_uses_request_context() -> None:
    token = set_request_id("request-123")
    try:
        record = logging.LogRecord(
            name="src.tests",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        record.event = "test_event"
        RequestContextFilter("travel-ai").filter(record)

        formatted = KeyValueFormatter().format(record)
    finally:
        reset_request_id(token)

    assert "request_id=request-123" in formatted
    assert "event=test_event" in formatted


def test_key_value_formatter_includes_error_type_for_exception_log() -> None:
    try:
        raise RuntimeError("formatting failed")
    except RuntimeError:
        record = logging.LogRecord(
            name="src.tests",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="exception happened",
            args=(),
            exc_info=sys.exc_info(),
        )
    record.error_type = "RuntimeError"
    RequestContextFilter("travel-ai").filter(record)

    formatted = KeyValueFormatter().format(record)

    assert "error_type=RuntimeError" in formatted
    assert "exception=" in formatted


def test_course_fallback_failure_event_is_not_overwritten(monkeypatch, caplog) -> None:
    def fail_fallback(*_args, **_kwargs):
        raise RuntimeError("course fallback exploded")

    monkeypatch.setattr(travel_service, "fallback_course_response", fail_fallback)
    caplog.set_level(logging.ERROR)

    with pytest.raises(HTTPException):
        travel_service.fallback_course_response_or_500(
            "11000",
            2,
            log_extra={
                "event": "course_fallback",
                "endpoint": "/generate-course",
                "runtime": "shared_next_poi_gru",
            },
        )

    record = next(item for item in caplog.records if item.event == "course_fallback_failure")
    assert record.endpoint == "/generate-course"
    assert record.runtime == "shared_next_poi_gru"
    assert record.fallback_reason == "unexpected_error"
    assert record.error_type == "RuntimeError"
    assert record.alert is True
    assert record.alert_severity == "CRITICAL"


def test_recommend_fallback_failure_event_is_not_overwritten(monkeypatch, caplog) -> None:
    def fail_fallback(*_args, **_kwargs):
        raise RuntimeError("recommend fallback exploded")

    monkeypatch.setattr(travel_service, "fallback_recommend_response", fail_fallback)
    caplog.set_level(logging.ERROR)

    with pytest.raises(HTTPException):
        travel_service.fallback_recommend_response_or_500(
            "11000",
            ["2815426"],
            log_extra={
                "event": "recommend_fallback",
                "endpoint": "/recommend",
                "runtime": "shared_next_poi_gru",
            },
        )

    record = next(item for item in caplog.records if item.event == "recommend_fallback_failure")
    assert record.endpoint == "/recommend"
    assert record.runtime == "shared_next_poi_gru"
    assert record.fallback_reason == "unexpected_error"
    assert record.error_type == "RuntimeError"
    assert record.alert is True
    assert record.alert_severity == "CRITICAL"
