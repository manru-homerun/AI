from __future__ import annotations

from typing import Any

from src.core.config import SETTINGS, Settings
from src.core.logging import get_logger
from src.inference.shared_next_poi import OnnxSharedNextPoiRuntime


logger = get_logger(__name__)

SHARED_RUNTIME: OnnxSharedNextPoiRuntime | None = None
RUNTIME: Any | None = None
COURSE_RUNTIME: Any | None = None
SHARED_RUNTIME_ERROR: str | None = None
RUNTIME_ERROR: str | None = None
COURSE_RUNTIME_ERROR: str | None = None


def initialize_runtimes(settings: Settings = SETTINGS) -> None:
    global SHARED_RUNTIME, RUNTIME, COURSE_RUNTIME, SHARED_RUNTIME_ERROR, RUNTIME_ERROR, COURSE_RUNTIME_ERROR
    try:
        SHARED_RUNTIME = OnnxSharedNextPoiRuntime(settings.shared_gru_artifact_dir)
        RUNTIME = SHARED_RUNTIME
        COURSE_RUNTIME = SHARED_RUNTIME
        SHARED_RUNTIME_ERROR = None
        RUNTIME_ERROR = None
        COURSE_RUNTIME_ERROR = None
        logger.info("shared next-POI GRU runtime initialized")
    except Exception as exc:
        SHARED_RUNTIME = None
        RUNTIME = None
        COURSE_RUNTIME = None
        SHARED_RUNTIME_ERROR = str(exc)
        RUNTIME_ERROR = str(exc)
        COURSE_RUNTIME_ERROR = str(exc)
        logger.exception("shared next-POI GRU runtime initialization failed")


def is_ready() -> bool:
    return SHARED_RUNTIME is not None


def health_payload() -> dict[str, Any]:
    shared_loaded = SHARED_RUNTIME is not None
    return {
        "status": "ok" if is_ready() else "degraded",
        "shared_gru_loaded": shared_loaded,
        "tiny_gru_loaded": shared_loaded,
        "course_decoder_loaded": shared_loaded,
    }


def readiness_payload() -> dict[str, Any]:
    return health_payload()
