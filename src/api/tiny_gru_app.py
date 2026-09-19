from __future__ import annotations

import sys
import types
from typing import Any

from fastapi import HTTPException

from src.api.app import app, create_app
from src.core.config import SETTINGS
from src.fallback.travel import (
    FALLBACK_CONTENT_IDS_BY_AREA,
    build_area_limited_course_ids,
    fallback_content_ids_for_area,
    unique_preserve_order,
    validate_forced_content_ids_fit,
)
from src.inference import runtime as runtime_state
from src.inference.recommender import load_runtime, recommend_internal as _recommend_internal
from src.schemas.travel import (
    GenerateCourseRequest,
    GenerateCourseResponse,
    GenerateCourseStep,
    RecommendItem,
    RecommendRequest,
    RecommendResponse,
    TravelBackendRequest,
    TravelGenerateRequest,
    TravelSpotSuggestionsRequest,
)
from src.services.travel_service import (
    TravelService,
    fallback_course_response,
    fallback_recommend_response,
    normalize_code_list,
    parse_int_choice,
    split_codes,
    travel_generate_request_to_user_features,
)


ARTIFACT_DIR = SETTINGS.tiny_gru_artifact_dir
COURSE_ARTIFACT_DIR = SETTINGS.course_decoder_artifact_dir
BACKEND_RECOMMENDATION_TOP_K = SETTINGS.backend_recommendation_top_k


def startup() -> None:
    runtime_state.initialize_runtimes()


def health() -> dict[str, Any]:
    return runtime_state.health_payload()


def recommend_internal(request: RecommendRequest) -> RecommendResponse:
    if runtime_state.RUNTIME is None:
        raise HTTPException(status_code=503, detail="runtime is not loaded")
    try:
        return _recommend_internal(runtime_state.RUNTIME, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid user_features: {exc}") from exc


def generate_course_internal(request: GenerateCourseRequest) -> GenerateCourseResponse:
    if runtime_state.COURSE_RUNTIME is None:
        raise HTTPException(status_code=503, detail="course decoder runtime is not loaded")
    try:
        content_ids, steps = runtime_state.COURSE_RUNTIME.generate(
            user_features=request.user_features,
            trip_days=request.trip_days,
            desired_poi_count=request.desired_poi_count,
            duplicate_masking=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid course generation request: {exc}") from exc
    return GenerateCourseResponse(
        content_id_sequence=content_ids,
        steps=[
            GenerateCourseStep(
                rank=step.rank,
                day_index=step.day_index,
                slot_index=step.slot_index,
                content_id=step.content_id,
                token_id=step.token_id,
                score=step.score,
            )
            for step in steps
        ],
    )


def recommend_for_backend(request: TravelSpotSuggestionsRequest) -> RecommendResponse:
    return TravelService().recommend_for_backend(request)


def generate_travel(request: TravelGenerateRequest) -> GenerateCourseResponse:
    return TravelService().generate_travel_course(request)


def suggest_travel_spots(request: TravelSpotSuggestionsRequest) -> RecommendResponse:
    return TravelService().suggest_travel_spots(request)


def complete_area_limited_recommendations(
    area_code: str,
    seen: set[str],
    recommendations: list[RecommendItem],
) -> RecommendResponse:
    from src.services.travel_service import complete_area_limited_recommendations as _complete

    return _complete(area_code, seen, recommendations)


def __getattr__(name: str) -> Any:
    if name in {"RUNTIME", "COURSE_RUNTIME", "RUNTIME_ERROR", "COURSE_RUNTIME_ERROR"}:
        return getattr(runtime_state, name)
    raise AttributeError(name)


class _CompatModule(types.ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"RUNTIME", "COURSE_RUNTIME", "RUNTIME_ERROR", "COURSE_RUNTIME_ERROR"}:
            setattr(runtime_state, name, value)
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _CompatModule
