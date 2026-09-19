from __future__ import annotations

import re
from typing import Any, Optional

from fastapi import HTTPException

from src.core.config import SETTINGS, Settings
from src.core.logging import get_logger
from src.fallback.travel import (
    build_fallback_course_payload,
    build_fallback_recommendation_payload,
    fallback_content_ids_for_area,
    validate_forced_content_ids_fit,
)
from src.inference import runtime as runtime_state
from src.inference.recommender import recommend_backend_area_limited
from src.schemas.travel import (
    GenerateCourseResponse,
    GenerateCourseStep,
    RecommendItem,
    RecommendResponse,
    TravelBackendRequest,
    TravelGenerateRequest,
    TravelSpotSuggestionsRequest,
)


logger = get_logger(__name__)


def _raise_fallback_error(endpoint: str, exc: Exception) -> None:
    logger.exception(
        "%s fallback response generation failed fallback_reason=unexpected_error",
        endpoint,
    )
    raise HTTPException(status_code=500, detail="fallback response generation failed") from exc


def split_codes(value: str) -> list[str]:
    return [part for part in re.split(r"[;,\s]+", str(value).strip()) if part]


def normalize_code_list(value: str, max_items: Optional[int] = None) -> str:
    codes = split_codes(value)
    if max_items is not None:
        codes = codes[:max_items]
    return ";".join(codes)


def parse_int_choice(field_name: str, value: str, allowed_values: set[int]) -> int:
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} must be one of {sorted(allowed_values)}") from exc
    if parsed not in allowed_values:
        raise ValueError(f"{field_name} must be one of {sorted(allowed_values)}")
    return parsed


def travel_generate_request_to_user_features(request: TravelBackendRequest) -> tuple[dict[str, Any], int]:
    trip_days = parse_int_choice("travelDuration", request.travelDuration, {1, 2, 3})
    p0_age = parse_int_choice("ageGroup", request.ageGroup, {20, 30, 40, 50, 60})
    style_codes = split_codes(request.travelerStyle)
    if len(style_codes) != 1:
        raise ValueError("travelerStyle must contain exactly one code")
    preferred_area = ";".join(request.preferredArea)
    residence_area = str(request.residenceArea).strip()
    if not residence_area:
        raise ValueError("residenceArea must not be empty")

    style = style_codes[0]
    user_features = {
        "area_code": str(request.areaCode).strip(),
        "trip_days": trip_days,
        "theme": str(request.travelPersona),
        "has_child": int(request.hasChild),
        "has_elderly": int(request.hasElderly),
        "has_disabled": int(request.hasDisabled),
        "companion_count": int(request.companionCount),
        "p0_age": p0_age,
        "p0_gender": request.gender,
        "p0_style": ";".join([style] * 8),
        "p0_home": residence_area,
        "p0_preferred": preferred_area,
        "p1_age": None,
        "p1_gender": None,
        "p1_style": None,
        "p1_home": None,
        "p1_preferred": None,
    }
    if not user_features["area_code"]:
        raise ValueError("areaCode must not be empty")
    return user_features, trip_days


def _course_response_from_payload(content_ids: list[str], steps: list[Any]) -> GenerateCourseResponse:
    return GenerateCourseResponse(
        content_id_sequence=content_ids,
        steps=[
            GenerateCourseStep(
                rank=step["rank"] if isinstance(step, dict) else step.rank,
                day_index=step["day_index"] if isinstance(step, dict) else step.day_index,
                slot_index=step["slot_index"] if isinstance(step, dict) else step.slot_index,
                content_id=step["content_id"] if isinstance(step, dict) else step.content_id,
                token_id=step["token_id"] if isinstance(step, dict) else step.token_id,
                score=step["score"] if isinstance(step, dict) else step.score,
            )
            for step in steps
        ],
    )


def fallback_course_response(
    area_code: str,
    trip_days: int,
    forced_content_ids: list[str] | None = None,
) -> GenerateCourseResponse:
    content_ids, steps = build_fallback_course_payload(area_code, trip_days, forced_content_ids)
    return _course_response_from_payload(content_ids, steps)


def fallback_recommend_response(area_code: str, content_id_sequence: list[str]) -> RecommendResponse:
    items = build_fallback_recommendation_payload(area_code, content_id_sequence)
    return RecommendResponse(recommendations=[RecommendItem(**item) for item in items])


def fallback_course_response_or_500(
    area_code: str,
    trip_days: int,
    forced_content_ids: list[str] | None = None,
) -> GenerateCourseResponse:
    try:
        return fallback_course_response(area_code, trip_days, forced_content_ids)
    except Exception as exc:
        _raise_fallback_error("generate-course", exc)


def fallback_recommend_response_or_500(area_code: str, content_id_sequence: list[str]) -> RecommendResponse:
    try:
        return fallback_recommend_response(area_code, content_id_sequence)
    except Exception as exc:
        _raise_fallback_error("recommend", exc)


def complete_area_limited_recommendations(
    area_code: str,
    seen: set[str],
    recommendations: list[RecommendItem],
    top_k: int = SETTINGS.backend_recommendation_top_k,
) -> RecommendResponse:
    area_content_ids = fallback_content_ids_for_area(area_code)
    recommended_ids = {item.content_id for item in recommendations}
    for content_id in area_content_ids:
        if content_id in seen or content_id in recommended_ids:
            continue
        recommendations.append(
            RecommendItem(
                content_id=content_id,
                token_id=len(recommendations) + 3,
                score=1.0 - (len(recommendations) * 0.05),
            )
        )
        recommended_ids.add(content_id)
        if len(recommendations) == top_k:
            break
    if len(recommendations) < top_k:
        for content_id in area_content_ids:
            if content_id in recommended_ids:
                continue
            recommendations.append(
                RecommendItem(
                    content_id=content_id,
                    token_id=len(recommendations) + 3,
                    score=1.0 - (len(recommendations) * 0.05),
                )
            )
            recommended_ids.add(content_id)
            if len(recommendations) == top_k:
                break
    return RecommendResponse(recommendations=recommendations[:top_k])


class TravelService:
    def __init__(self, settings: Settings = SETTINGS) -> None:
        self.settings = settings

    def generate_travel_course(self, request: TravelGenerateRequest) -> GenerateCourseResponse:
        try:
            user_features, trip_days = travel_generate_request_to_user_features(request)
            desired_poi_count = trip_days * 3
            forced_content_ids = validate_forced_content_ids_fit(request.contentIdList, desired_poi_count)
            if runtime_state.COURSE_RUNTIME is None:
                logger.warning(
                    "course runtime unavailable; using fallback course response fallback_reason=model_unavailable"
                )
                return fallback_course_response_or_500(request.areaCode, trip_days, forced_content_ids)
            content_ids, steps = runtime_state.COURSE_RUNTIME.generate(
                user_features=user_features,
                trip_days=trip_days,
                desired_poi_count=desired_poi_count,
                duplicate_masking=True,
                forced_content_ids=forced_content_ids,
                allowed_content_ids=fallback_content_ids_for_area(request.areaCode),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            logger.exception("course inference failed; using fallback course response fallback_reason=inference_error")
            trip_days = parse_int_choice("travelDuration", request.travelDuration, {1, 2, 3})
            forced_content_ids = validate_forced_content_ids_fit(request.contentIdList, trip_days * 3)
            return fallback_course_response_or_500(request.areaCode, trip_days, forced_content_ids)
        return _course_response_from_payload(content_ids, steps)

    def suggest_travel_spots(self, request: TravelSpotSuggestionsRequest) -> RecommendResponse:
        try:
            return self.recommend_for_backend(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def recommend_for_backend(self, request: TravelSpotSuggestionsRequest) -> RecommendResponse:
        user_features, _ = travel_generate_request_to_user_features(request)
        if not request.contentIdSequence:
            return fallback_recommend_response_or_500(request.areaCode, [])
        if runtime_state.RUNTIME is None:
            logger.warning(
                "recommendation runtime unavailable; using fallback recommendation response "
                "fallback_reason=model_unavailable"
            )
            return fallback_recommend_response_or_500(request.areaCode, request.contentIdSequence)

        try:
            allowed_content_ids = set(fallback_content_ids_for_area(request.areaCode))
            recommendations = recommend_backend_area_limited(
                runtime=runtime_state.RUNTIME,
                content_id_sequence=request.contentIdSequence,
                user_features=user_features,
                allowed_content_ids=allowed_content_ids,
                top_k=self.settings.backend_recommendation_top_k,
            )
            if len(recommendations) == self.settings.backend_recommendation_top_k:
                return RecommendResponse(recommendations=recommendations)
            seen = {str(content_id) for content_id in request.contentIdSequence}
            return complete_area_limited_recommendations(
                request.areaCode,
                seen,
                recommendations,
                self.settings.backend_recommendation_top_k,
            )
        except Exception:
            logger.exception(
                "recommendation inference failed; using fallback recommendation response "
                "fallback_reason=inference_error"
            )
            return fallback_recommend_response_or_500(request.areaCode, request.contentIdSequence)
