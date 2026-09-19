from __future__ import annotations

import logging

from src.core.config import Settings
from src.inference import runtime as runtime_state


class DummyCourseGenerator:
    def __init__(self, artifact_dir):
        self.artifact_dir = artifact_dir


def test_initialize_runtimes_logs_success(monkeypatch, tmp_path, caplog) -> None:
    tiny_dir = tmp_path / "tiny"
    course_dir = tmp_path / "course"
    tiny_dir.mkdir()
    course_dir.mkdir()
    (course_dir / "course_user_encoder.onnx").write_text("", encoding="utf-8")
    (course_dir / "course_decoder_step.onnx").write_text("", encoding="utf-8")
    settings = Settings(tiny_gru_artifact_dir=tiny_dir, course_decoder_artifact_dir=course_dir)
    monkeypatch.setattr(runtime_state, "load_runtime", lambda _artifact_dir: {"session": object()})
    monkeypatch.setattr(runtime_state, "OnnxCourseGenerator", DummyCourseGenerator)
    caplog.set_level(logging.INFO)

    runtime_state.initialize_runtimes(settings)

    assert runtime_state.RUNTIME is not None
    assert runtime_state.COURSE_RUNTIME is not None
    assert "tiny GRU runtime initialized" in caplog.text
    assert "course decoder runtime initialized" in caplog.text


def test_initialize_runtimes_logs_tiny_gru_failure_with_exception(monkeypatch, tmp_path, caplog) -> None:
    tiny_dir = tmp_path / "tiny"
    course_dir = tmp_path / "course"
    tiny_dir.mkdir()
    course_dir.mkdir()
    settings = Settings(tiny_gru_artifact_dir=tiny_dir, course_decoder_artifact_dir=course_dir)

    def fail_load(_artifact_dir):
        raise RuntimeError("tiny load failed")

    monkeypatch.setattr(runtime_state, "load_runtime", fail_load)
    caplog.set_level(logging.ERROR)

    runtime_state.initialize_runtimes(settings)

    assert runtime_state.RUNTIME is None
    assert runtime_state.RUNTIME_ERROR == "tiny load failed"
    assert "tiny GRU runtime initialization failed" in caplog.text
    assert "tiny load failed" in caplog.text


def test_initialize_runtimes_logs_course_generator_failure_with_exception(monkeypatch, tmp_path, caplog) -> None:
    tiny_dir = tmp_path / "tiny"
    course_dir = tmp_path / "course"
    tiny_dir.mkdir()
    course_dir.mkdir()
    (course_dir / "course_user_encoder.onnx").write_text("", encoding="utf-8")
    (course_dir / "course_decoder_step.onnx").write_text("", encoding="utf-8")
    settings = Settings(tiny_gru_artifact_dir=tiny_dir, course_decoder_artifact_dir=course_dir)
    monkeypatch.setattr(runtime_state, "load_runtime", lambda _artifact_dir: {"session": object()})

    def fail_course_generator(_artifact_dir):
        raise RuntimeError("course load failed")

    monkeypatch.setattr(runtime_state, "OnnxCourseGenerator", fail_course_generator)
    caplog.set_level(logging.ERROR)

    runtime_state.initialize_runtimes(settings)

    assert runtime_state.COURSE_RUNTIME is None
    assert runtime_state.COURSE_RUNTIME_ERROR == "course load failed"
    assert "course decoder runtime initialization failed" in caplog.text
    assert "course load failed" in caplog.text
