from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from src.api.routes.travel import router as travel_router
from src.inference import runtime as runtime_state


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    runtime_state.initialize_runtimes()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Tiny GRU POI Recommender", lifespan=lifespan)
    app.include_router(travel_router)
    return app


app = create_app()
