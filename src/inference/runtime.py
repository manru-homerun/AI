from __future__ import annotations

from typing import Any

from src.core.config import SETTINGS, Settings
from src.inference.course_generator import OnnxCourseGenerator
from src.inference.recommender import load_runtime


RUNTIME: dict[str, Any] | None = None
COURSE_RUNTIME: OnnxCourseGenerator | None = None
RUNTIME_ERROR: str | None = None
COURSE_RUNTIME_ERROR: str | None = None


def initialize_runtimes(settings: Settings = SETTINGS) -> None:
    global RUNTIME, COURSE_RUNTIME, RUNTIME_ERROR, COURSE_RUNTIME_ERROR
    try:
        RUNTIME = load_runtime(settings.tiny_gru_artifact_dir)
        RUNTIME_ERROR = None
    except Exception as exc:
        RUNTIME = None
        RUNTIME_ERROR = str(exc)

    try:
        if (settings.course_decoder_artifact_dir / "course_user_encoder.onnx").exists() and (
            settings.course_decoder_artifact_dir / "course_decoder_step.onnx"
        ).exists():
            COURSE_RUNTIME = OnnxCourseGenerator(settings.course_decoder_artifact_dir)
            COURSE_RUNTIME_ERROR = None
        else:
            COURSE_RUNTIME = None
            COURSE_RUNTIME_ERROR = "course decoder ONNX artifacts are missing"
    except Exception as exc:
        COURSE_RUNTIME = None
        COURSE_RUNTIME_ERROR = str(exc)


def health_payload(settings: Settings = SETTINGS) -> dict[str, Any]:
    return {
        "status": "ok" if RUNTIME is not None and COURSE_RUNTIME is not None else "degraded",
        "tiny_gru_loaded": RUNTIME is not None,
        "course_decoder_loaded": COURSE_RUNTIME is not None,
        "tiny_gru_artifact_dir": str(settings.tiny_gru_artifact_dir),
        "course_decoder_artifact_dir": str(settings.course_decoder_artifact_dir),
        "tiny_gru_error": RUNTIME_ERROR,
        "course_decoder_error": COURSE_RUNTIME_ERROR,
    }
