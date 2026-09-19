from __future__ import annotations

from typing import Any

from src.core.config import SETTINGS, Settings
from src.core.logging import get_logger
from src.inference.course_generator import OnnxCourseGenerator
from src.inference.recommender import load_runtime


logger = get_logger(__name__)

RUNTIME: dict[str, Any] | None = None
COURSE_RUNTIME: OnnxCourseGenerator | None = None
RUNTIME_ERROR: str | None = None
COURSE_RUNTIME_ERROR: str | None = None


def initialize_runtimes(settings: Settings = SETTINGS) -> None:
    global RUNTIME, COURSE_RUNTIME, RUNTIME_ERROR, COURSE_RUNTIME_ERROR
    try:
        RUNTIME = load_runtime(settings.tiny_gru_artifact_dir)
        RUNTIME_ERROR = None
        logger.info("tiny GRU runtime initialized")
    except Exception as exc:
        RUNTIME = None
        RUNTIME_ERROR = str(exc)
        logger.exception("tiny GRU runtime initialization failed")

    try:
        if (settings.course_decoder_artifact_dir / "course_user_encoder.onnx").exists() and (
            settings.course_decoder_artifact_dir / "course_decoder_step.onnx"
        ).exists():
            COURSE_RUNTIME = OnnxCourseGenerator(settings.course_decoder_artifact_dir)
            COURSE_RUNTIME_ERROR = None
            logger.info("course decoder runtime initialized")
        else:
            COURSE_RUNTIME = None
            COURSE_RUNTIME_ERROR = "course decoder ONNX artifacts are missing"
            logger.warning("course decoder runtime initialization skipped: ONNX artifacts are missing")
    except Exception as exc:
        COURSE_RUNTIME = None
        COURSE_RUNTIME_ERROR = str(exc)
        logger.exception("course decoder runtime initialization failed")


def is_ready() -> bool:
    return RUNTIME is not None and COURSE_RUNTIME is not None


def health_payload() -> dict[str, Any]:
    return {
        "status": "ok" if is_ready() else "degraded",
        "tiny_gru_loaded": RUNTIME is not None,
        "course_decoder_loaded": COURSE_RUNTIME is not None,
    }


def readiness_payload() -> dict[str, Any]:
    return health_payload()
