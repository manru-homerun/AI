from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.routes.travel import router as travel_router
from src.core.alerting import begin_alert_context, has_request_alerted, notify_discord, reset_alert_context
from src.core.logging import REQUEST_ID_HEADER, configure_logging, get_logger, reset_request_id, set_request_id
from src.inference import runtime as runtime_state
from src.services.travel_service import (
    invalid_course_input_fallback_response,
    invalid_recommend_input_fallback_response,
)


logger = get_logger(__name__)
HEALTH_ENDPOINTS = {"/live", "/ready", "/health"}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    runtime_state.initialize_runtimes()
    yield


async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
    token = set_request_id(request_id)
    alert_token = begin_alert_context()
    started_at = time.perf_counter()
    endpoint = request.url.path
    access_log_level = "debug" if endpoint in HEALTH_ENDPOINTS else "info"
    getattr(logger, access_log_level)(
        "request started",
        extra={
            "event": "request_started",
            "request_id": request_id,
            "endpoint": endpoint,
            "method": request.method,
            "path": endpoint,
        },
    )
    try:
        response = await call_next(request)
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        failure_extra = {
            "event": "request_failed",
            "request_id": request_id,
            "endpoint": endpoint,
            "method": request.method,
            "path": endpoint,
            "elapsed_ms": elapsed_ms,
            "error_type": type(exc).__name__,
        }
        if endpoint in HEALTH_ENDPOINTS:
            logger.debug("request failed", exc_info=True, extra=failure_extra)
        else:
            alert_extra = dict(failure_extra)
            if not has_request_alerted():
                alert_extra.update({"alert": True, "alert_severity": "CRITICAL"})
                notify_discord(
                    event="request_failed",
                    severity="CRITICAL",
                    message="request failed with unhandled server error",
                    context=alert_extra,
                )
            logger.exception("request failed", extra=alert_extra)
        raise
    else:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        response.headers[REQUEST_ID_HEADER] = request_id
        if response.status_code >= 500 and endpoint not in HEALTH_ENDPOINTS and not has_request_alerted():
            failure_extra = {
                "event": "request_failed",
                "request_id": request_id,
                "endpoint": endpoint,
                "method": request.method,
                "path": endpoint,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
                "alert": True,
                "alert_severity": "CRITICAL",
            }
            logger.error("request finished with server error", extra=failure_extra)
            notify_discord(
                event="request_failed",
                severity="CRITICAL",
                message="request finished with server error",
                context=failure_extra,
            )
        getattr(logger, access_log_level)(
            "request finished",
            extra={
                "event": "request_finished",
                "request_id": request_id,
                "endpoint": endpoint,
                "method": request.method,
                "path": endpoint,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
            },
        )
        return response
    finally:
        reset_alert_context(alert_token)
        reset_request_id(token)


async def backend_request_validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.method != "POST" or request.url.path not in {"/generate-course", "/recommend"}:
        return await request_validation_exception_handler(request, exc)
    try:
        body = await request.json()
    except ValueError:
        body = {}
    payload = body if isinstance(body, dict) else {}
    logger.warning(
        "backend request validation failed; using fallback response",
        extra={
            "event": "backend_request_validation_fallback",
            "path": request.url.path,
            "validation_error_count": len(exc.errors()),
        },
    )
    if request.url.path == "/generate-course":
        response = invalid_course_input_fallback_response(payload, fallback_reason="request_validation_error")
    else:
        response = invalid_recommend_input_fallback_response(payload, fallback_reason="request_validation_error")
    return JSONResponse(status_code=200, content=jsonable_encoder(response))


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Tiny GRU POI Recommender", lifespan=lifespan)
    app.add_exception_handler(RequestValidationError, backend_request_validation_exception_handler)
    app.middleware("http")(request_context_middleware)
    app.include_router(travel_router)
    return app


app = create_app()
