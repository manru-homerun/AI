from __future__ import annotations

import logging

from src.core.logging import KeyValueFormatter, RequestContextFilter, reset_request_id, set_request_id


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
