from __future__ import annotations

import logging

from src.core.config import Settings
from src.core import alerting
from src.inference import runtime as runtime_state


class DummySharedRuntime:
    def __init__(self, artifact_dir):
        self.artifact_dir = artifact_dir


def test_initialize_runtimes_logs_success(monkeypatch, tmp_path, caplog) -> None:
    artifact_dir = tmp_path / "shared"
    artifact_dir.mkdir()
    settings = Settings(shared_gru_artifact_dir=artifact_dir)
    monkeypatch.setattr(runtime_state, "OnnxSharedNextPoiRuntime", DummySharedRuntime)
    caplog.set_level(logging.INFO)

    runtime_state.initialize_runtimes(settings)

    assert runtime_state.SHARED_RUNTIME is not None
    assert runtime_state.RUNTIME is runtime_state.SHARED_RUNTIME
    assert runtime_state.COURSE_RUNTIME is runtime_state.SHARED_RUNTIME
    assert runtime_state.SHARED_RUNTIME_ERROR is None
    assert runtime_state.RUNTIME_ERROR is None
    assert runtime_state.COURSE_RUNTIME_ERROR is None
    assert "shared next-POI GRU runtime initialized" in caplog.text
    record = next(item for item in caplog.records if getattr(item, "event", None) == "runtime_initialized")
    assert record.runtime == "shared_next_poi_gru"


def test_initialize_runtimes_logs_shared_gru_failure_with_exception(monkeypatch, tmp_path, caplog) -> None:
    artifact_dir = tmp_path / "shared"
    artifact_dir.mkdir()
    settings = Settings(shared_gru_artifact_dir=artifact_dir)

    def fail_load(_artifact_dir):
        raise RuntimeError("shared load failed")

    monkeypatch.setattr(runtime_state, "OnnxSharedNextPoiRuntime", fail_load)
    alert_calls = []

    def spy_notify(*args, **kwargs):
        alert_calls.append((args, kwargs))
        return alerting.notify_discord(*args, **kwargs)

    monkeypatch.setattr(runtime_state, "notify_discord", spy_notify)
    caplog.set_level(logging.ERROR)

    runtime_state.initialize_runtimes(settings)

    assert runtime_state.SHARED_RUNTIME is None
    assert runtime_state.RUNTIME is None
    assert runtime_state.COURSE_RUNTIME is None
    assert runtime_state.SHARED_RUNTIME_ERROR == "shared load failed"
    assert runtime_state.RUNTIME_ERROR == "shared load failed"
    assert runtime_state.COURSE_RUNTIME_ERROR == "shared load failed"
    assert "shared next-POI GRU runtime initialization failed" in caplog.text
    assert "shared load failed" in caplog.text
    record = next(item for item in caplog.records if getattr(item, "event", None) == "runtime_initialization_failure")
    assert record.runtime == "shared_next_poi_gru"
    assert record.error_type == "RuntimeError"
    assert record.alert is True
    assert record.alert_severity == "CRITICAL"
    assert [call[1]["event"] for call in alert_calls] == ["runtime_initialization_failure"]
