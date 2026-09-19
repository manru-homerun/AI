from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.inference import runtime as runtime_state
from src.schemas.travel import GenerateCourseResponse, RecommendResponse, TravelGenerateRequest, TravelSpotSuggestionsRequest
from src.services.travel_service import TravelService


router = APIRouter()
travel_service = TravelService()


def _runtime_status_response() -> JSONResponse:
    status_code = 200 if runtime_state.is_ready() else 503
    return JSONResponse(status_code=status_code, content=runtime_state.health_payload())


@router.get("/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> JSONResponse:
    return _runtime_status_response()


@router.get("/health")
def health() -> JSONResponse:
    return _runtime_status_response()


@router.post("/generate-course", response_model=GenerateCourseResponse)
def generate_travel(request: TravelGenerateRequest) -> GenerateCourseResponse:
    return travel_service.generate_travel_course(request)


@router.post("/recommend", response_model=RecommendResponse)
def suggest_travel_spots(request: TravelSpotSuggestionsRequest) -> RecommendResponse:
    return travel_service.suggest_travel_spots(request)
