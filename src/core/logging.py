from __future__ import annotations

import contextvars
import datetime as dt
import logging
import os
import sys
from collections.abc import Mapping
from typing import Any


DEFAULT_SERVICE_NAME = "travel-ai"
REQUEST_ID_HEADER = "X-Request-ID"
COMMON_LOG_FIELDS = (
    "timestamp",
    "level",
    "service",
    "logger",
    "request_id",
    "event",
    "message",
    "endpoint",
    "runtime",
    "fallback_reason",
    "error_type",
    "elapsed_ms",
    "alert",
    "alert_severity",
)
OPTIONAL_COMMON_LOG_FIELDS = COMMON_LOG_FIELDS[7:]

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_STANDARD_LOG_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)
_STANDARD_LOG_RECORD_ATTRS.update({"message", "asctime"})


def get_request_id() -> str:
    return _request_id.get()


def set_request_id(request_id: str) -> contextvars.Token[str]:
    return _request_id.set(request_id)


def reset_request_id(token: contextvars.Token[str]) -> None:
    _request_id.reset(token)


class RequestContextFilter(logging.Filter):
    def __init__(self, service_name: str = DEFAULT_SERVICE_NAME) -> None:
        super().__init__()
        self.service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "service"):
            record.service = self.service_name
        if not hasattr(record, "request_id"):
            record.request_id = get_request_id()
        if not hasattr(record, "event"):
            record.event = "-"
        for field_name in OPTIONAL_COMMON_LOG_FIELDS:
            if not hasattr(record, field_name):
                setattr(record, field_name, "-")
        return True


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        if not hasattr(record, "service"):
            record.service = DEFAULT_SERVICE_NAME
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        if not hasattr(record, "event"):
            record.event = "-"
        for field_name in OPTIONAL_COMMON_LOG_FIELDS:
            if not hasattr(record, field_name):
                setattr(record, field_name, "-")

        fields: dict[str, Any] = {
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc).isoformat(),
            "level": record.levelname,
            "service": record.service,
            "logger": record.name,
            "request_id": record.request_id,
            "event": record.event,
            "message": record.message,
            "endpoint": record.endpoint,
            "runtime": record.runtime,
            "fallback_reason": record.fallback_reason,
            "error_type": record.error_type,
            "elapsed_ms": record.elapsed_ms,
            "alert": record.alert,
            "alert_severity": record.alert_severity,
        }
        for key, value in self._extra_fields(record).items():
            if key not in fields:
                fields[key] = value

        line = " ".join(f"{key}={self._format_value(value)}" for key, value in fields.items())
        if record.exc_info:
            line = f"{line} exception={self._format_value(self.formatException(record.exc_info))}"
        return line

    @staticmethod
    def _extra_fields(record: logging.LogRecord) -> Mapping[str, Any]:
        return {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_ATTRS and not key.startswith("_")
        }

    @staticmethod
    def _format_value(value: Any) -> str:
        if value is None:
            return "-"
        text = str(value)
        if not text:
            return '""'
        if any(char.isspace() or char in {'"', "="} for char in text):
            return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
        return text


def _has_filter(handler: logging.Handler, filter_type: type[logging.Filter]) -> bool:
    return any(isinstance(existing, filter_type) for existing in handler.filters)


def configure_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    service_name = os.environ.get("SERVICE_NAME", DEFAULT_SERVICE_NAME)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        root_logger.addHandler(logging.StreamHandler(sys.stdout))

    for handler in root_logger.handlers:
        handler.setLevel(level)
        handler.setFormatter(KeyValueFormatter())
        if not _has_filter(handler, RequestContextFilter):
            handler.addFilter(RequestContextFilter(service_name))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
