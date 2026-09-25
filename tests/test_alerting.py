from __future__ import annotations

import logging

from src.core import alerting
from src.core.logging import reset_request_id, set_request_id


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_notify_discord_noops_without_webhook_url(monkeypatch) -> None:
    monkeypatch.delenv(alerting.DISCORD_WEBHOOK_URL_ENV, raising=False)

    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("Discord should not be called")

    monkeypatch.setattr(alerting, "urlopen", fail_urlopen)

    future = alerting.notify_discord(
        event="recommend_inference_failure",
        severity="HIGH",
        message="recommendation failed",
        context={"endpoint": "/recommend"},
    )

    assert future is None


def test_notify_discord_sends_expected_payload_with_request_id(monkeypatch) -> None:
    sent = {}
    monkeypatch.setenv(alerting.DISCORD_WEBHOOK_URL_ENV, "https://discord.example/webhook")

    def fake_urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["body"] = request.data.decode("utf-8")
        sent["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(alerting, "urlopen", fake_urlopen)
    token = set_request_id("request-123")
    try:
        future = alerting.notify_discord(
            event="recommend_inference_failure",
            severity="HIGH",
            message="recommendation inference failed",
            context={
                "endpoint": "/recommend",
                "method": "POST",
                "runtime": "shared_next_poi_gru",
                "fallback_reason": "inference_error",
                "error_type": "RuntimeError",
            },
        )
        assert future is not None
        future.result(timeout=1)
    finally:
        reset_request_id(token)

    assert sent["url"] == "https://discord.example/webhook"
    assert sent["timeout"] == alerting.DISCORD_TIMEOUT_SECONDS
    assert "[HIGH] Travel AI Server Error" in sent["body"]
    assert "recommend_inference_failure" in sent["body"]
    assert "POST /recommend" in sent["body"]
    assert "request-123" in sent["body"]
    assert "RuntimeError" in sent["body"]


def test_discord_delivery_failure_is_logged_without_raising(monkeypatch, caplog) -> None:
    webhook_url = "https://discord.example/webhook"
    monkeypatch.setenv(alerting.DISCORD_WEBHOOK_URL_ENV, webhook_url)

    def fail_urlopen(*_args, **_kwargs):
        raise RuntimeError("discord unavailable")

    monkeypatch.setattr(alerting, "urlopen", fail_urlopen)
    caplog.set_level(logging.WARNING)

    future = alerting.notify_discord(
        event="course_fallback_failure",
        severity="CRITICAL",
        message="fallback failed",
        context={"endpoint": "/generate-course", "error_type": "RuntimeError"},
    )

    assert future is not None
    future.result(timeout=1)
    record = next(item for item in caplog.records if item.event == "discord_alert_delivery_failure")
    assert record.error_type == "RuntimeError"
    assert record.source_event == "course_fallback_failure"
    assert webhook_url not in caplog.text
