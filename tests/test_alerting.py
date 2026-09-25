from __future__ import annotations

import logging
from urllib.error import HTTPError

import pytest

from src.core import alerting
from src.core.logging import reset_request_id, set_request_id


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


@pytest.fixture(autouse=True)
def reset_alerting(monkeypatch):
    alerting.reset_alerting_state_for_tests()
    monkeypatch.delenv(alerting.DISCORD_WEBHOOK_URL_ENV, raising=False)
    yield
    alerting.reset_alerting_state_for_tests()


def _fake_success_urlopen(sent):
    def fake_urlopen(request, timeout):
        sent.append(
            {
                "url": request.full_url,
                "body": request.data.decode("utf-8"),
                "user_agent": request.headers.get("User-agent"),
                "timeout": timeout,
            }
        )
        return FakeResponse()

    return fake_urlopen


def test_notify_discord_noops_without_webhook_url(monkeypatch) -> None:
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
    sent = []
    monkeypatch.setenv(alerting.DISCORD_WEBHOOK_URL_ENV, "https://discord.example/webhook")
    monkeypatch.setattr(alerting, "urlopen", _fake_success_urlopen(sent))
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

    assert sent[0]["url"] == "https://discord.example/webhook"
    assert sent[0]["timeout"] == alerting.DISCORD_TIMEOUT_SECONDS
    assert sent[0]["user_agent"] == alerting.DISCORD_USER_AGENT
    assert "[HIGH] Travel AI Server Error" in sent[0]["body"]
    assert "recommend_inference_failure" in sent[0]["body"]
    assert "POST /recommend" in sent[0]["body"]
    assert "request-123" in sent[0]["body"]
    assert "RuntimeError" in sent[0]["body"]


def test_discord_delivery_failure_is_logged_without_raising(monkeypatch, caplog) -> None:
    webhook_url = "https://discord.example/webhook"
    monkeypatch.setenv(alerting.DISCORD_WEBHOOK_URL_ENV, webhook_url)

    def fail_urlopen(*_args, **_kwargs):
        raise HTTPError(
            url=webhook_url,
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

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
    assert record.error_type == "HTTPError"
    assert record.http_status == 404
    assert record.source_event == "course_fallback_failure"
    assert webhook_url not in caplog.text


def test_identical_alert_is_deduplicated_inside_ttl(monkeypatch, caplog) -> None:
    sent = []
    now = 1000.0
    monkeypatch.setenv(alerting.DISCORD_WEBHOOK_URL_ENV, "https://discord.example/webhook")
    monkeypatch.setattr(alerting, "urlopen", _fake_success_urlopen(sent))
    monkeypatch.setattr(alerting, "_monotonic", lambda: now)
    caplog.set_level(logging.DEBUG)

    first = alerting.notify_discord(
        "recommend_inference_failure",
        "HIGH",
        "recommendation failed",
        {"error_type": "RuntimeError", "fallback_reason": "inference_error"},
    )
    alert_token = alerting.begin_alert_context()
    try:
        second = alerting.notify_discord(
            "recommend_inference_failure",
            "HIGH",
            "recommendation failed again",
            {"error_type": "RuntimeError", "fallback_reason": "inference_error"},
        )
    finally:
        alerting.reset_alert_context(alert_token)

    assert first is not None
    first.result(timeout=1)
    assert second is None
    assert len(sent) == 1
    record = next(item for item in caplog.records if item.event == "discord_alert_suppressed")
    assert record.suppression_reason == "deduplicated"
    assert record.alert_event == "recommend_inference_failure"


def test_identical_alert_after_ttl_is_sent_again(monkeypatch) -> None:
    sent = []
    now = 1000.0
    monkeypatch.setenv(alerting.DISCORD_WEBHOOK_URL_ENV, "https://discord.example/webhook")
    monkeypatch.setattr(alerting, "urlopen", _fake_success_urlopen(sent))
    monkeypatch.setattr(alerting, "_monotonic", lambda: now)

    first = alerting.notify_discord(
        "recommend_inference_failure",
        "HIGH",
        "recommendation failed",
        {"error_type": "RuntimeError", "fallback_reason": "inference_error"},
    )
    assert first is not None
    first.result(timeout=1)

    alert_token = alerting.begin_alert_context()
    now += alerting.DISCORD_ALERT_DEDUPE_TTL_SECONDS + 1
    try:
        second = alerting.notify_discord(
            "recommend_inference_failure",
            "HIGH",
            "recommendation failed after ttl",
            {"error_type": "RuntimeError", "fallback_reason": "inference_error"},
        )
        assert second is not None
        second.result(timeout=1)
    finally:
        alerting.reset_alert_context(alert_token)

    assert len(sent) == 2


def test_different_alert_keys_are_sent_independently(monkeypatch) -> None:
    sent = []
    monkeypatch.setenv(alerting.DISCORD_WEBHOOK_URL_ENV, "https://discord.example/webhook")
    monkeypatch.setattr(alerting, "urlopen", _fake_success_urlopen(sent))

    first = alerting.notify_discord(
        "recommend_inference_failure",
        "HIGH",
        "recommendation failed",
        {"error_type": "RuntimeError", "fallback_reason": "inference_error"},
    )
    second_token = alerting.begin_alert_context()
    try:
        second = alerting.notify_discord(
            "course_inference_failure",
            "HIGH",
            "course failed",
            {"error_type": "RuntimeError", "fallback_reason": "inference_error"},
        )
    finally:
        alerting.reset_alert_context(second_token)
    third_token = alerting.begin_alert_context()
    try:
        third = alerting.notify_discord(
            "recommend_inference_failure",
            "HIGH",
            "recommendation failed differently",
            {"error_type": "ValueError", "fallback_reason": "inference_error"},
        )
    finally:
        alerting.reset_alert_context(third_token)

    assert first is not None
    assert second is not None
    assert third is not None
    for future in (first, second, third):
        future.result(timeout=1)
    assert len(sent) == 3


def test_request_severity_gate_suppresses_duplicate_high(monkeypatch) -> None:
    sent = []
    monkeypatch.setenv(alerting.DISCORD_WEBHOOK_URL_ENV, "https://discord.example/webhook")
    monkeypatch.setattr(alerting, "urlopen", _fake_success_urlopen(sent))
    token = alerting.begin_alert_context()
    try:
        first = alerting.notify_discord(
            "recommend_inference_failure",
            "HIGH",
            "recommendation failed",
            {"error_type": "RuntimeError", "fallback_reason": "inference_error"},
        )
        second = alerting.notify_discord(
            "course_inference_failure",
            "HIGH",
            "course failed",
            {"error_type": "ValueError", "fallback_reason": "inference_error"},
        )
    finally:
        alerting.reset_alert_context(token)

    assert first is not None
    first.result(timeout=1)
    assert second is None
    assert len(sent) == 1


def test_request_severity_gate_allows_critical_escalation(monkeypatch) -> None:
    sent = []
    monkeypatch.setenv(alerting.DISCORD_WEBHOOK_URL_ENV, "https://discord.example/webhook")
    monkeypatch.setattr(alerting, "urlopen", _fake_success_urlopen(sent))
    token = alerting.begin_alert_context()
    try:
        high = alerting.notify_discord(
            "recommend_inference_failure",
            "HIGH",
            "recommendation failed",
            {"error_type": "RuntimeError", "fallback_reason": "inference_error"},
        )
        critical = alerting.notify_discord(
            "recommend_fallback_failure",
            "CRITICAL",
            "fallback failed",
            {"error_type": "RuntimeError", "fallback_reason": "unexpected_error"},
        )
    finally:
        alerting.reset_alert_context(token)

    assert high is not None
    assert critical is not None
    high.result(timeout=1)
    critical.result(timeout=1)
    assert len(sent) == 2


def test_request_severity_gate_suppresses_duplicate_critical(monkeypatch) -> None:
    sent = []
    monkeypatch.setenv(alerting.DISCORD_WEBHOOK_URL_ENV, "https://discord.example/webhook")
    monkeypatch.setattr(alerting, "urlopen", _fake_success_urlopen(sent))
    token = alerting.begin_alert_context()
    try:
        first = alerting.notify_discord(
            "recommend_fallback_failure",
            "CRITICAL",
            "fallback failed",
            {"error_type": "RuntimeError", "fallback_reason": "unexpected_error"},
        )
        second = alerting.notify_discord(
            "request_failed",
            "CRITICAL",
            "request failed",
            {"error_type": "RuntimeError", "fallback_reason": "-"},
        )
    finally:
        alerting.reset_alert_context(token)

    assert first is not None
    first.result(timeout=1)
    assert second is None
    assert len(sent) == 1


def test_delivery_failure_releases_dedupe_reservation(monkeypatch) -> None:
    sent = []
    monkeypatch.setenv(alerting.DISCORD_WEBHOOK_URL_ENV, "https://discord.example/webhook")

    def fail_once_then_succeed(request, timeout):
        if not sent:
            sent.append("failed")
            raise RuntimeError("discord unavailable")
        sent.append(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(alerting, "urlopen", fail_once_then_succeed)

    first = alerting.notify_discord(
        "recommend_inference_failure",
        "HIGH",
        "recommendation failed",
        {"error_type": "RuntimeError", "fallback_reason": "inference_error"},
    )
    assert first is not None
    first.result(timeout=1)

    token = alerting.begin_alert_context()
    try:
        second = alerting.notify_discord(
            "recommend_inference_failure",
            "HIGH",
            "recommendation failed retry",
            {"error_type": "RuntimeError", "fallback_reason": "inference_error"},
        )
        assert second is not None
        second.result(timeout=1)
    finally:
        alerting.reset_alert_context(token)

    assert len(sent) == 2


def test_lazy_cleanup_removes_expired_dedupe_entries(monkeypatch) -> None:
    sent = []
    now = 1000.0
    monkeypatch.setenv(alerting.DISCORD_WEBHOOK_URL_ENV, "https://discord.example/webhook")
    monkeypatch.setattr(alerting, "urlopen", _fake_success_urlopen(sent))
    monkeypatch.setattr(alerting, "_monotonic", lambda: now)

    first = alerting.notify_discord(
        "recommend_inference_failure",
        "HIGH",
        "recommendation failed",
        {"error_type": "RuntimeError", "fallback_reason": "inference_error"},
    )
    assert first is not None
    first.result(timeout=1)
    assert len(alerting._dedupe_expires_at) == 1

    now += alerting.DISCORD_ALERT_DEDUPE_TTL_SECONDS + 1
    token = alerting.begin_alert_context()
    try:
        second = alerting.notify_discord(
            "course_inference_failure",
            "HIGH",
            "course failed",
            {"error_type": "RuntimeError", "fallback_reason": "inference_error"},
        )
        assert second is not None
        second.result(timeout=1)
    finally:
        alerting.reset_alert_context(token)

    assert len(alerting._dedupe_expires_at) == 1
    assert ("course_inference_failure", "RuntimeError", "inference_error") in alerting._dedupe_expires_at
