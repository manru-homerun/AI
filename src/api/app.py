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
from src.core.logging import REQUEST_ID_HEADER, configure_logging, get_logger, reset_request_id, set_request_id
from src.inference import runtime as runtime_state
from src.services.travel_service import (
    invalid_course_input_fallback_response,
    invalid_recommend_input_fallback_response,
    log_qa_validation_request_summary,
)


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    runtime_state.initialize_runtimes()
    yield


async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
    token = set_request_id(request_id)
    started_at = time.perf_counter()
    logger.info(
        "request started",
        extra={
            "event": "request_started",
            "method": request.method,
            "path": request.url.path,
        },
    )
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.exception(
            "request failed",
            extra={
                "event": "request_failed",
                "method": request.method,
                "path": request.url.path,
                "elapsed_ms": elapsed_ms,
            },
        )
        raise
    else:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request finished",
            extra={
                "event": "request_finished",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
            },
        )
        return response
    finally:
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
    log_qa_validation_request_summary(request.url.path, payload, len(exc.errors()))
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
