from __future__ import annotations

from typing import Any

from src.core.alerting import notify_discord
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
        logger.info(
            "shared next-POI GRU runtime initialized",
            extra={"event": "runtime_initialized", "runtime": "shared_next_poi_gru"},
        )
    except Exception as exc:
        SHARED_RUNTIME = None
        RUNTIME = None
        COURSE_RUNTIME = None
        SHARED_RUNTIME_ERROR = str(exc)
        RUNTIME_ERROR = str(exc)
        COURSE_RUNTIME_ERROR = str(exc)
        alert_extra = {
            "event": "runtime_initialization_failure",
            "runtime": "shared_next_poi_gru",
            "error_type": type(exc).__name__,
            "alert": True,
            "alert_severity": "CRITICAL",
        }
        logger.exception(
            "shared next-POI GRU runtime initialization failed",
            extra=alert_extra,
        )
        notify_discord(
            event="runtime_initialization_failure",
            severity="CRITICAL",
            message="shared next-POI GRU runtime initialization failed",
            context=alert_extra,
        )


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
