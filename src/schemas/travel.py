from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.fallback.travel import FALLBACK_CONTENT_IDS_BY_AREA


class RecommendRequest(BaseModel):
    content_id_sequence: list[str] = Field(min_length=1)
    user_features: dict[str, Any]
    top_k: int = Field(default=10, ge=1, le=50)


class RecommendItem(BaseModel):
    content_id: str
    token_id: int
    score: float


class RecommendResponse(BaseModel):
    recommendations: list[RecommendItem]


class GenerateCourseRequest(BaseModel):
    user_features: dict[str, Any]
    trip_days: int = Field(default=1, ge=1, le=30)
    desired_poi_count: Optional[int] = Field(default=None, ge=1, le=50)
    top_k: int = Field(default=10, ge=1, le=50)


class GenerateCourseStep(BaseModel):
    rank: int
    day_index: int
    slot_index: int
    content_id: str
    token_id: int
    score: float


class GenerateCourseResponse(BaseModel):
    content_id_sequence: list[str]
    steps: list[GenerateCourseStep]


class TravelBackendRequest(BaseModel):
    areaCode: str
    travelDuration: str
    travelPersona: int = Field(ge=1, le=7)
    ageGroup: str
    gender: str
    travelerStyle: str
    preferredArea: list[str] = Field(min_length=1, max_length=3)
    residenceArea: str
    hasChild: bool
    hasElderly: bool
    hasDisabled: bool
    companionCount: int = Field(ge=0)

    @field_validator("areaCode")
    @classmethod
    def validate_area_code(cls, value: str) -> str:
        normalized = str(value).strip()
        if normalized not in FALLBACK_CONTENT_IDS_BY_AREA:
            raise ValueError(f"areaCode must be one of {sorted(FALLBACK_CONTENT_IDS_BY_AREA)}")
        return normalized

    @field_validator("preferredArea")
    @classmethod
    def validate_preferred_area(cls, value: list[str]) -> list[str]:
        for code in value:
            if not re.fullmatch(r"\d{5}", str(code)):
                raise ValueError("preferredArea items must be 5-digit strings")
        return value


class TravelGenerateRequest(TravelBackendRequest):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "areaCode": "11000",
                "contentIdList": ["2815426", "2773265"],
                "travelDuration": "2",
                "travelPersona": 3,
                "ageGroup": "30",
                "gender": "F",
                "travelerStyle": "4",
                "preferredArea": ["50110", "26350"],
                "residenceArea": "11000",
                "hasChild": 0,
                "hasElderly": 0,
                "hasDisabled": 0,
                "companionCount": 1,
            }
        }
    )

    contentIdList: list[str] = Field(min_length=1)

    @field_validator("contentIdList")
    @classmethod
    def validate_content_id_list(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for content_id in value:
            normalized_content_id = str(content_id).strip()
            if not re.fullmatch(r"\d+", normalized_content_id):
                raise ValueError("contentIdList items must be digit strings")
            normalized.append(normalized_content_id)
        return normalized


class TravelSpotSuggestionsRequest(TravelBackendRequest):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "contentIdSequence": ["2815426", "2773265"],
                "areaCode": "11000",
                "travelDuration": "2",
                "travelPersona": 3,
                "ageGroup": "30",
                "gender": "F",
                "travelerStyle": "4",
                "preferredArea": ["50110", "26350"],
                "residenceArea": "11",
                "hasChild": False,
                "hasElderly": False,
                "hasDisabled": False,
                "companionCount": 1,
            }
        }
    )

    contentIdSequence: list[str]

    @field_validator("contentIdSequence")
    @classmethod
    def validate_content_id_sequence(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for content_id in value:
            normalized_content_id = str(content_id).strip()
            if not re.fullmatch(r"\d+", normalized_content_id):
                raise ValueError("contentIdSequence items must be digit strings")
            normalized.append(normalized_content_id)
        return normalized
