from __future__ import annotations

import logging
from uuid import UUID

from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api import tiny_gru_app


def test_request_id_is_generated_and_returned(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    client = TestClient(tiny_gru_app.app)

    response = client.get("/live")

    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    UUID(request_id)
    finished = next(record for record in caplog.records if getattr(record, "event", None) == "request_finished")
    assert finished.request_id == request_id
    assert finished.endpoint == "/live"
    assert finished.path == "/live"
    assert finished.status_code == 200
    assert isinstance(finished.elapsed_ms, float)


def test_request_id_header_is_reused(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    client = TestClient(tiny_gru_app.app)

    response = client.get("/live", headers={"X-Request-ID": "test-request-id"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-id"
    finished = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "request_finished" and record.request_id == "test-request-id"
    )
    assert finished.elapsed_ms >= 0


def test_health_access_logs_are_below_info_level(caplog) -> None:
    caplog.set_level(logging.INFO)
    client = TestClient(tiny_gru_app.app)

    for path in ("/live", "/ready", "/health"):
        client.get(path)

    assert not any(
        getattr(record, "event", None) in {"request_started", "request_finished"}
        and getattr(record, "endpoint", None) in {"/live", "/ready", "/health"}
        for record in caplog.records
    )


def test_health_access_logs_are_available_at_debug_level(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    client = TestClient(tiny_gru_app.app)

    for path in ("/live", "/ready", "/health"):
        client.get(path)

    finished_endpoints = {
        record.endpoint
        for record in caplog.records
        if getattr(record, "event", None) == "request_finished"
        and getattr(record, "endpoint", None) in {"/live", "/ready", "/health"}
    }
    assert finished_endpoints == {"/live", "/ready", "/health"}


def test_health_endpoints_do_not_alert(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "notify_discord",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("health endpoints should not alert")),
    )
    client = TestClient(tiny_gru_app.app)

    for path in ("/live", "/ready", "/health"):
        client.get(path)


def test_unhandled_server_error_uses_generic_request_failed_alert(monkeypatch) -> None:
    app = app_module.create_app()

    @app.get("/boom")
    def boom():
        raise RuntimeError("boom")

    alert_calls = []

    def spy_notify(*args, **kwargs):
        alert_calls.append((args, kwargs))
        return None

    monkeypatch.setattr(app_module, "notify_discord", spy_notify)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/boom")

    assert response.status_code == 500
    assert [call[1]["event"] for call in alert_calls] == ["request_failed"]
    assert alert_calls[0][1]["severity"] == "CRITICAL"
