from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from fastapi import HTTPException

from src.core.alerting import notify_discord
from src.core.config import COURSE_POIS_PER_DAY, SETTINGS, Settings
from src.core.logging import get_logger
from src.fallback.travel import (
    FALLBACK_CONTENT_IDS_BY_AREA,
    build_central_tourism_recommendation_payload,
    build_fallback_course_payload,
    build_fallback_recommendation_payload,
    fallback_content_ids_for_area,
    validate_forced_content_ids_fit,
)
from src.inference import runtime as runtime_state
from src.schemas.travel import (
    GenerateCourseResponse,
    GenerateCourseStep,
    RecommendItem,
    RecommendResponse,
    TravelBackendRequest,
    TravelGenerateRequest,
    TravelSpotSuggestionsRequest,
)
from src.services.accessibility import filter_recommendations_by_accessibility, requested_accessibility_features


logger = get_logger(__name__)
DEFAULT_FALLBACK_AREA_CODE = "11000"
DEFAULT_FALLBACK_TRIP_DAYS = 2


def _log_context(
    *,
    endpoint: str,
    area_code: str,
    trip_days: int | None,
    runtime: str,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "endpoint": endpoint,
        "method": "POST",
        "area_code": area_code,
        "runtime": runtime,
    }
    if trip_days is not None:
        context["trip_days"] = trip_days
    if fallback_reason is not None:
        context["fallback_reason"] = fallback_reason
    return context


def _raise_fallback_error(endpoint: str, exc: Exception, log_extra: Mapping[str, Any] | None = None) -> None:
    extra: dict[str, Any] = {}
    if log_extra is not None:
        extra.update(dict(log_extra))
    extra["event"] = f"{endpoint}_fallback_failure"
    extra["fallback_reason"] = "unexpected_error"
    extra["error_type"] = type(exc).__name__
    extra["alert"] = True
    extra["alert_severity"] = "CRITICAL"
    notify_discord(
        event=extra["event"],
        severity="CRITICAL",
        message="fallback response generation failed",
        context=extra,
    )
    logger.exception(
        "%s fallback response generation failed fallback_reason=unexpected_error",
        endpoint,
        extra=extra,
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


def normalize_age_group(value: Any) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (OverflowError, ValueError) as exc:
        raise ValueError("ageGroup must be numeric") from exc
    if parsed < 20:
        return 20
    if parsed >= 60:
        return 60
    return (parsed // 10) * 10


def fallback_area_code_from_value(value: Any) -> str:
    normalized = str(value).strip()
    if normalized in FALLBACK_CONTENT_IDS_BY_AREA:
        return normalized
    return DEFAULT_FALLBACK_AREA_CODE


def fallback_trip_days_from_value(value: Any) -> int:
    try:
        return parse_int_choice("travelDuration", str(value), {1, 2, 3})
    except ValueError:
        return DEFAULT_FALLBACK_TRIP_DAYS


def valid_content_id_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for content_id in value:
        text = str(content_id).strip()
        if text.isdigit():
            normalized.append(text)
    return normalized


def raw_backend_payload(request_or_payload: TravelBackendRequest | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(request_or_payload, Mapping):
        return request_or_payload
    return request_or_payload.model_dump()


def invalid_course_input_fallback_response(
    request_or_payload: TravelBackendRequest | Mapping[str, Any],
    fallback_reason: str = "invalid_input",
) -> GenerateCourseResponse:
    payload = raw_backend_payload(request_or_payload)
    area_code = fallback_area_code_from_value(payload.get("areaCode"))
    trip_days = fallback_trip_days_from_value(payload.get("travelDuration"))
    desired_poi_count = trip_days * COURSE_POIS_PER_DAY
    forced_content_ids = valid_content_id_values(payload.get("contentIdList"))[:desired_poi_count]
    log_extra = {
        "event": "course_fallback",
        **_log_context(
            endpoint="/generate-course",
            area_code=area_code,
            trip_days=trip_days,
            runtime="shared_next_poi_gru",
            fallback_reason=fallback_reason,
        ),
    }
    logger.warning(
        "course request input invalid; using fallback course response fallback_reason=%s",
        fallback_reason,
        extra=log_extra,
    )
    return fallback_course_response_or_500(area_code, trip_days, forced_content_ids, log_extra)


def invalid_recommend_input_fallback_response(
    request_or_payload: TravelBackendRequest | Mapping[str, Any],
    fallback_reason: str = "invalid_input",
) -> RecommendResponse:
    payload = raw_backend_payload(request_or_payload)
    area_code = fallback_area_code_from_value(payload.get("areaCode"))
    trip_days = fallback_trip_days_from_value(payload.get("travelDuration"))
    content_id_sequence = valid_content_id_values(payload.get("contentIdSequence"))
    log_extra = {
        "event": "recommend_fallback",
        **_log_context(
            endpoint="/recommend",
            area_code=area_code,
            trip_days=trip_days,
            runtime="shared_next_poi_gru",
            fallback_reason=fallback_reason,
        ),
    }
    logger.warning(
        "recommendation request input invalid; using central tourism fallback response fallback_reason=%s",
        fallback_reason,
        extra=log_extra,
    )
    return central_tourism_recommend_response(
        area_code=area_code,
        content_id_sequence=content_id_sequence,
        top_k=SETTINGS.backend_recommendation_top_k,
        log_extra=log_extra,
    )


def travel_generate_request_to_user_features(request: TravelBackendRequest) -> tuple[dict[str, Any], int]:
    trip_days = parse_int_choice("travelDuration", request.travelDuration, {1, 2, 3})
    p0_age = normalize_age_group(request.ageGroup)
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


def passthrough_over_requested_course_response(content_ids: list[str], trip_days: int) -> GenerateCourseResponse:
    steps: list[dict[str, Any]] = []
    last_day_start_index = max((trip_days - 1) * COURSE_POIS_PER_DAY, 0)
    for index, content_id in enumerate(content_ids):
        if index < last_day_start_index:
            day_index = index // COURSE_POIS_PER_DAY + 1
            slot_index = index % COURSE_POIS_PER_DAY + 1
        else:
            day_index = trip_days
            slot_index = index - last_day_start_index + 1
        steps.append(
            {
                "rank": index + 1,
                "day_index": day_index,
                "slot_index": slot_index,
                "content_id": content_id,
                "token_id": -1,
                "score": 0.0,
            }
        )
    return _course_response_from_payload(content_ids, steps)


def fallback_course_response(
    area_code: str,
    trip_days: int,
    forced_content_ids: list[str] | None = None,
) -> GenerateCourseResponse:
    content_ids, steps = build_fallback_course_payload(area_code, trip_days, forced_content_ids)
    return _course_response_from_payload(content_ids, steps)


def fallback_recommend_response(
    area_code: str,
    content_id_sequence: list[str],
    top_k: int = SETTINGS.backend_recommendation_top_k,
    required_accessibility_features: set[str] | None = None,
    log_extra: Mapping[str, Any] | None = None,
) -> RecommendResponse:
    candidate_k = max(top_k, SETTINGS.backend_recommendation_candidate_k)
    items = build_fallback_recommendation_payload(area_code, content_id_sequence, candidate_k)
    recommendations = [RecommendItem(**item) for item in items]
    if required_accessibility_features:
        recommendations = filter_recommendations_by_accessibility(
            recommendations,
            required_accessibility_features,
            log_extra=log_extra,
            logger=logger,
        )
    return RecommendResponse(recommendations=recommendations[:top_k])


def central_tourism_recommend_response(
    area_code: str,
    content_id_sequence: list[str],
    top_k: int,
    log_extra: Mapping[str, Any],
    required_accessibility_features: set[str] | None = None,
) -> RecommendResponse:
    candidate_k = max(top_k, SETTINGS.backend_recommendation_candidate_k)
    try:
        items = build_central_tourism_recommendation_payload(area_code, candidate_k)
    except Exception as exc:
        logger.exception(
            "central tourism fallback failed; using static recommendation response",
            extra={
                **log_extra,
                "event": "recommend_central_tourism_fallback_failure",
                "error_type": type(exc).__name__,
            },
        )
        return fallback_recommend_response_or_500(
            area_code,
            content_id_sequence,
            log_extra,
            required_accessibility_features,
        )
    recommendations = [RecommendItem(**item) for item in items]
    if required_accessibility_features:
        recommendations = filter_recommendations_by_accessibility(
            recommendations,
            required_accessibility_features,
            log_extra=log_extra,
            logger=logger,
        )
    return RecommendResponse(recommendations=recommendations[:top_k])


def fallback_course_response_or_500(
    area_code: str,
    trip_days: int,
    forced_content_ids: list[str] | None = None,
    log_extra: Mapping[str, Any] | None = None,
) -> GenerateCourseResponse:
    try:
        return fallback_course_response(area_code, trip_days, forced_content_ids)
    except Exception as exc:
        _raise_fallback_error("course", exc, log_extra)


def fallback_recommend_response_or_500(
    area_code: str,
    content_id_sequence: list[str],
    log_extra: Mapping[str, Any] | None = None,
    required_accessibility_features: set[str] | None = None,
) -> RecommendResponse:
    try:
        return fallback_recommend_response(
            area_code,
            content_id_sequence,
            SETTINGS.backend_recommendation_top_k,
            required_accessibility_features,
            log_extra,
        )
    except Exception as exc:
        _raise_fallback_error("recommend", exc, log_extra)


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
            trip_days = parse_int_choice("travelDuration", request.travelDuration, {1, 2, 3})
            desired_poi_count = trip_days * COURSE_POIS_PER_DAY
            if len(request.contentIdList) > desired_poi_count:
                # Defensive guard while upstream may temporarily send more POIs than the generation target.
                logger.info(
                    "course generation short-circuited for over-requested contentIdList",
                    extra={
                        "event": "course_short_circuit_over_requested_content_ids",
                        **_log_context(
                            endpoint="/generate-course",
                            area_code=request.areaCode,
                            trip_days=trip_days,
                            runtime="shared_next_poi_gru",
                        ),
                        "input_count": len(request.contentIdList),
                        "target_count": desired_poi_count,
                    },
                )
                return passthrough_over_requested_course_response(list(request.contentIdList), trip_days)
            user_features, trip_days = travel_generate_request_to_user_features(request)
            forced_content_ids = validate_forced_content_ids_fit(request.contentIdList, desired_poi_count)
        except ValueError:
            return invalid_course_input_fallback_response(request)

        if runtime_state.SHARED_RUNTIME is None:
            log_extra = {
                "event": "course_fallback",
                **_log_context(
                    endpoint="/generate-course",
                    area_code=request.areaCode,
                    trip_days=trip_days,
                    runtime="shared_next_poi_gru",
                    fallback_reason="model_unavailable",
                ),
            }
            logger.warning(
                "course runtime unavailable; using fallback course response fallback_reason=model_unavailable",
                extra=log_extra,
            )
            alert_extra = {
                **log_extra,
                "event": "model_unavailable",
                "alert": True,
                "alert_severity": "HIGH",
            }
            logger.error("course runtime unavailable", extra=alert_extra)
            notify_discord(
                event="model_unavailable",
                severity="HIGH",
                message="course runtime unavailable; using fallback course response",
                context=alert_extra,
            )
            return fallback_course_response_or_500(request.areaCode, trip_days, forced_content_ids, log_extra)

        try:
            content_ids, steps = runtime_state.SHARED_RUNTIME.generate(
                user_features=user_features,
                trip_days=trip_days,
                desired_poi_count=desired_poi_count,
                duplicate_masking=True,
                forced_content_ids=forced_content_ids,
            )
        except Exception as exc:
            log_extra = {
                "event": "course_inference_failure",
                **_log_context(
                    endpoint="/generate-course",
                    area_code=request.areaCode,
                    trip_days=trip_days,
                    runtime="shared_next_poi_gru",
                    fallback_reason="inference_error",
                ),
            }
            logger.exception(
                "course inference failed; using fallback course response fallback_reason=inference_error",
                extra={
                    **log_extra,
                    "error_type": type(exc).__name__,
                    "alert": True,
                    "alert_severity": "HIGH",
                },
            )
            notify_discord(
                event="course_inference_failure",
                severity="HIGH",
                message="course inference failed; using fallback course response",
                context={**log_extra, "error_type": type(exc).__name__, "alert": True, "alert_severity": "HIGH"},
            )
            return fallback_course_response_or_500(request.areaCode, trip_days, forced_content_ids, log_extra)
        logger.info(
            "course inference succeeded",
            extra={
                "event": "course_inference_success",
                **_log_context(
                    endpoint="/generate-course",
                    area_code=request.areaCode,
                    trip_days=trip_days,
                    runtime="shared_next_poi_gru",
                ),
            },
        )
        return _course_response_from_payload(content_ids, steps)

    def suggest_travel_spots(self, request: TravelSpotSuggestionsRequest) -> RecommendResponse:
        try:
            return self.recommend_for_backend(request)
        except ValueError:
            return invalid_recommend_input_fallback_response(request)

    def recommend_for_backend(self, request: TravelSpotSuggestionsRequest) -> RecommendResponse:
        user_features, trip_days = travel_generate_request_to_user_features(request)
        desired_poi_count = trip_days * COURSE_POIS_PER_DAY
        required_features = requested_accessibility_features(
            has_disabled=bool(request.hasDisabled),
            has_elderly=bool(request.hasElderly),
            has_child=bool(request.hasChild),
        )
        if not request.contentIdSequence:
            log_extra = {
                "event": "recommend_fallback",
                **_log_context(
                    endpoint="/recommend",
                    area_code=request.areaCode,
                    trip_days=trip_days,
                    runtime="shared_next_poi_gru",
                    fallback_reason="empty_sequence_central_tourism",
                ),
            }
            logger.info(
                "recommendation skipped for empty content sequence; using central tourism fallback response",
                extra=log_extra,
            )
            return central_tourism_recommend_response(
                area_code=request.areaCode,
                content_id_sequence=[],
                top_k=self.settings.backend_recommendation_top_k,
                log_extra=log_extra,
                required_accessibility_features=required_features,
            )
        if len(request.contentIdSequence) == desired_poi_count:
            log_extra = {
                "event": "recommend_fallback",
                **_log_context(
                    endpoint="/recommend",
                    area_code=request.areaCode,
                    trip_days=trip_days,
                    runtime="shared_next_poi_gru",
                    fallback_reason="full_course_sequence",
                ),
            }
            logger.info(
                "recommendation skipped for full course content sequence; using central tourism fallback response",
                extra=log_extra,
            )
            return central_tourism_recommend_response(
                area_code=request.areaCode,
                content_id_sequence=request.contentIdSequence,
                top_k=self.settings.backend_recommendation_top_k,
                log_extra=log_extra,
                required_accessibility_features=required_features,
            )
        if runtime_state.SHARED_RUNTIME is None:
            log_extra = {
                "event": "recommend_fallback",
                **_log_context(
                    endpoint="/recommend",
                    area_code=request.areaCode,
                    trip_days=trip_days,
                    runtime="shared_next_poi_gru",
                    fallback_reason="model_unavailable",
                ),
            }
            logger.warning(
                "recommendation runtime unavailable; using fallback recommendation response "
                "fallback_reason=model_unavailable",
                extra=log_extra,
            )
            alert_extra = {
                **log_extra,
                "event": "model_unavailable",
                "alert": True,
                "alert_severity": "HIGH",
            }
            logger.error("recommendation runtime unavailable", extra=alert_extra)
            notify_discord(
                event="model_unavailable",
                severity="HIGH",
                message="recommendation runtime unavailable; using fallback recommendation response",
                context=alert_extra,
            )
            return fallback_recommend_response_or_500(
                request.areaCode,
                request.contentIdSequence,
                log_extra,
                required_features,
            )

        try:
            recommendations = runtime_state.SHARED_RUNTIME.recommend(
                content_id_sequence=request.contentIdSequence,
                user_features=user_features,
                area_code=request.areaCode,
                top_k=self.settings.backend_recommendation_candidate_k,
            )
            if not recommendations:
                log_extra = {
                    "event": "recommend_fallback",
                    **_log_context(
                        endpoint="/recommend",
                        area_code=request.areaCode,
                        trip_days=trip_days,
                        runtime="shared_next_poi_gru",
                        fallback_reason="empty_known_sequence",
                    ),
                }
                logger.info(
                    "recommendation skipped for empty known sequence; using fallback recommendation response",
                    extra=log_extra,
                )
                return fallback_recommend_response_or_500(
                    request.areaCode,
                    request.contentIdSequence,
                    log_extra,
                    required_features,
                )
            seen = {str(content_id) for content_id in request.contentIdSequence}
            if len(recommendations) < self.settings.backend_recommendation_candidate_k:
                recommendations = complete_area_limited_recommendations(
                    request.areaCode,
                    seen,
                    recommendations,
                    self.settings.backend_recommendation_candidate_k,
                ).recommendations
            if required_features:
                recommendations = filter_recommendations_by_accessibility(
                    recommendations,
                    required_features,
                    log_extra={
                        "event": "recommend_accessibility_filter",
                        **_log_context(
                            endpoint="/recommend",
                            area_code=request.areaCode,
                            trip_days=trip_days,
                            runtime="shared_next_poi_gru",
                        ),
                    },
                    logger=logger,
                )
            if len(recommendations) >= self.settings.backend_recommendation_top_k:
                logger.info(
                    "recommendation inference succeeded",
                    extra={
                        "event": "recommend_inference_success",
                        **_log_context(
                            endpoint="/recommend",
                            area_code=request.areaCode,
                            trip_days=trip_days,
                            runtime="shared_next_poi_gru",
                        ),
                    },
                )
                return RecommendResponse(
                    recommendations=recommendations[: self.settings.backend_recommendation_top_k]
                )
            response = RecommendResponse(recommendations=recommendations[: self.settings.backend_recommendation_top_k])
            logger.info(
                "recommendation inference succeeded",
                extra={
                    "event": "recommend_inference_success",
                    **_log_context(
                        endpoint="/recommend",
                        area_code=request.areaCode,
                        trip_days=trip_days,
                        runtime="shared_next_poi_gru",
                    ),
                },
            )
            return response
        except Exception as exc:
            log_extra = {
                "event": "recommend_inference_failure",
                **_log_context(
                    endpoint="/recommend",
                    area_code=request.areaCode,
                    trip_days=trip_days,
                    runtime="shared_next_poi_gru",
                    fallback_reason="inference_error",
                ),
            }
            logger.exception(
                "recommendation inference failed; using fallback recommendation response "
                "fallback_reason=inference_error",
                extra={
                    **log_extra,
                    "error_type": type(exc).__name__,
                    "alert": True,
                    "alert_severity": "HIGH",
                },
            )
            notify_discord(
                event="recommend_inference_failure",
                severity="HIGH",
                message="recommendation inference failed; using fallback recommendation response",
                context={**log_extra, "error_type": type(exc).__name__, "alert": True, "alert_severity": "HIGH"},
            )
            return fallback_recommend_response_or_500(
                request.areaCode,
                request.contentIdSequence,
                log_extra,
                required_features,
            )
