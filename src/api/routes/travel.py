from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from src.inference import runtime as runtime_state
from src.schemas.travel import GenerateCourseResponse, RecommendResponse, TravelGenerateRequest, TravelSpotSuggestionsRequest
from src.services.travel_service import TravelService


router = APIRouter()
travel_service = TravelService()


@router.get("/health")
def health() -> dict[str, Any]:
    return runtime_state.health_payload()


@router.post("/generate-course", response_model=GenerateCourseResponse)
def generate_travel(request: TravelGenerateRequest) -> GenerateCourseResponse:
    return travel_service.generate_travel_course(request)


@router.post("/recommend", response_model=RecommendResponse)
def suggest_travel_spots(request: TravelSpotSuggestionsRequest) -> RecommendResponse:
    return travel_service.suggest_travel_spots(request)
