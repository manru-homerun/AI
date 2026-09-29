from __future__ import annotations

import logging

from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from src.api import app as app_module


def _metric_value(metrics_text: str, metric_name: str, labels: dict[str, str]) -> float:
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name == metric_name and all(sample.labels.get(key) == value for key, value in labels.items()):
                return float(sample.value)
    return 0.0


def test_metrics_endpoint_returns_prometheus_text() -> None:
    client = TestClient(app_module.create_app())

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# HELP api_http_requests_total" in response.text
    assert "# HELP api_http_request_duration_seconds" in response.text


def test_api_call_increments_request_counter_and_latency_histogram() -> None:
    client = TestClient(app_module.create_app())
    labels = {"method": "GET", "endpoint": "/live", "status_code": "200"}
    before = _metric_value(client.get("/metrics").text, "api_http_requests_total", labels)

    response = client.get("/live")
    metrics_text = client.get("/metrics").text

    assert response.status_code == 200
    assert _metric_value(metrics_text, "api_http_requests_total", labels) == before + 1
    assert _metric_value(metrics_text, "api_http_request_duration_seconds_count", labels) >= 1
    assert _metric_value(metrics_text, "api_http_request_duration_seconds_sum", labels) >= 0
    assert "api_http_request_duration_seconds_bucket" in metrics_text


def test_status_code_and_unmatched_endpoint_are_recorded_without_raw_path() -> None:
    client = TestClient(app_module.create_app())
    raw_path = "/unknown/12345/free-form-value"

    response = client.get(raw_path)
    metrics_text = client.get("/metrics").text

    assert response.status_code == 404
    assert _metric_value(
        metrics_text,
        "api_http_requests_total",
        {"method": "GET", "endpoint": "unmatched", "status_code": "404"},
    ) >= 1
    assert raw_path not in metrics_text


def test_metrics_endpoint_is_not_logged_alerted_or_recorded(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        app_module,
        "notify_discord",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("/metrics should not alert")),
    )
    caplog.set_level(logging.DEBUG)
    client = TestClient(app_module.create_app())

    response = client.get("/metrics")

    assert response.status_code == 200
    assert 'endpoint="/metrics"' not in response.text
    assert not any(
        getattr(record, "event", None) in {"request_started", "request_finished", "request_failed"}
        and getattr(record, "endpoint", None) == "/metrics"
        for record in caplog.records
    )


def test_metrics_endpoint_is_hidden_from_openapi() -> None:
    client = TestClient(app_module.create_app())

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/metrics" not in response.json()["paths"]
