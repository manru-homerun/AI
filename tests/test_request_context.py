from __future__ import annotations

import logging
from uuid import UUID

from fastapi.testclient import TestClient

from src.api import tiny_gru_app


def test_request_id_is_generated_and_returned(caplog) -> None:
    caplog.set_level(logging.INFO)
    client = TestClient(tiny_gru_app.app)

    response = client.get("/live")

    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    UUID(request_id)
    finished = next(record for record in caplog.records if record.event == "request_finished")
    assert finished.request_id == request_id
    assert finished.path == "/live"
    assert finished.status_code == 200
    assert isinstance(finished.elapsed_ms, float)


def test_request_id_header_is_reused(caplog) -> None:
    caplog.set_level(logging.INFO)
    client = TestClient(tiny_gru_app.app)

    response = client.get("/live", headers={"X-Request-ID": "test-request-id"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-id"
    finished = next(
        record
        for record in caplog.records
        if record.event == "request_finished" and record.request_id == "test-request-id"
    )
    assert finished.elapsed_ms >= 0
