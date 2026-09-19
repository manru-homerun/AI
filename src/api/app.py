from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, Request

from src.api.routes.travel import router as travel_router
from src.core.logging import REQUEST_ID_HEADER, configure_logging, get_logger, reset_request_id, set_request_id
from src.inference import runtime as runtime_state


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


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Tiny GRU POI Recommender", lifespan=lifespan)
    app.middleware("http")(request_context_middleware)
    app.include_router(travel_router)
    return app


app = create_app()
