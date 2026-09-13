from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.inference.course_generator import DEFAULT_ARTIFACT_DIR as DEFAULT_COURSE_ARTIFACT_DIR
from src.inference.course_generator import OnnxCourseGenerator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "tiny_gru_onnx_experiment"


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


class TravelGenerateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "areaCode": "11000",
                "travelDuration": "2",
                "travelPersona": 3,
                "ageGroup": "30",
                "gender": "남",
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
    companionCount: int = Field(ge=1)

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


class TravelSpotSuggestionsRequest(TravelGenerateRequest):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "contentIdSequence": ["2815426", "2773265"],
                "areaCode": "11000",
                "travelDuration": "2",
                "travelPersona": 3,
                "ageGroup": "30",
                "gender": "남",
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

    contentIdSequence: list[str] = Field(min_length=1)


def load_runtime(artifact_dir: Path) -> dict[str, Any]:
    import onnxruntime as ort

    with open(artifact_dir / "content_id_vocab.json", encoding="utf-8") as fp:
        vocab = json.load(fp)
    with open(artifact_dir / "feature_encoder.pkl", "rb") as fp:
        feature_encoder = pickle.load(fp)
    with open(artifact_dir / "train_config.json", encoding="utf-8") as fp:
        train_config = json.load(fp)

    session = ort.InferenceSession(str(artifact_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    content_id_to_token = vocab["content_id_to_token"]
    token_to_content_id = {int(key): value for key, value in vocab["token_to_content_id"].items()}
    return {
        "session": session,
        "feature_encoder": feature_encoder,
        "content_id_to_token": content_id_to_token,
        "token_to_content_id": token_to_content_id,
        "unk_token": vocab["unk_token"],
        "max_sequence_len": int(train_config["max_sequence_len"]),
    }


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


def travel_generate_request_to_user_features(request: TravelGenerateRequest) -> tuple[dict[str, Any], int]:
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


ARTIFACT_DIR = Path(os.environ.get("TINY_GRU_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)).resolve()
COURSE_ARTIFACT_DIR = Path(os.environ.get("COURSE_DECODER_ARTIFACT_DIR", DEFAULT_COURSE_ARTIFACT_DIR)).resolve()
BACKEND_RECOMMENDATION_TOP_K = 4
FALLBACK_CONTENT_IDS_BY_AREA = {
    "11000": [
        "2815426",
        "2773265",
        "3076141",
        "2758179",
        "3060919",
        "130289",
        "250469",
        "2930884",
        "129854",
        "1750737",
        "809490",
        "2930839",
    ],
    "41110": [
        "2868656",
        "2747132",
        "2892936",
        "3355253",
        "4076762",
        "2662855",
        "2944481",
        "2829013",
        "2753679",
        "2893042",
        "2613658",
        "1064469",
    ],
    "28000": [
        "2458348",
        "2994418",
        "1030642",
        "2612802",
        "2734015",
        "947611",
        "2767580",
        "1113230",
        "2834112",
        "3097744",
        "852304",
        "2837034",
    ],
    "30000": [
        "2580239",
        "1720749",
        "2900942",
        "1807489",
        "129785",
        "2662681",
        "2899345",
        "3454325",
        "2721465",
        "2752861",
        "2580604",
        "2738011",
    ],
    "27000": [
        "2864490",
        "130132",
        "1956986",
        "651687",
        "1871383",
        "2930650",
        "637751",
        "1611891",
        "2756646",
        "2785275",
        "126130",
        "2470055",
    ],
    "12000": [
        "2488192",
        "4065124",
        "1621360",
        "2755010",
        "2783851",
        "127539",
        "2779116",
        "3056157",
        "129761",
        "130065",
        "129782",
        "2033294",
    ],
    "26000": [
        "127488",
        "3345026",
        "2456767",
        "4011128",
        "4011143",
        "2931381",
        "2609623",
        "2785272",
        "2869241",
        "127925",
        "126078",
        "2991028",
    ],
    "48120": [
        "2864150",
        "2575790",
        "2627175",
        "1622590",
        "2844199",
        "2838789",
        "1622326",
        "2864877",
        "3456409",
        "3397899",
        "2863958",
        "3386455",
    ],
}
RUNTIME: dict[str, Any] | None = None
COURSE_RUNTIME: OnnxCourseGenerator | None = None
RUNTIME_ERROR: str | None = None
COURSE_RUNTIME_ERROR: str | None = None
app = FastAPI(title="Tiny GRU POI Recommender")


@app.on_event("startup")
def startup() -> None:
    global RUNTIME, COURSE_RUNTIME, RUNTIME_ERROR, COURSE_RUNTIME_ERROR
    try:
        RUNTIME = load_runtime(ARTIFACT_DIR)
        RUNTIME_ERROR = None
    except Exception as exc:
        RUNTIME = None
        RUNTIME_ERROR = str(exc)

    try:
        if (COURSE_ARTIFACT_DIR / "course_user_encoder.onnx").exists() and (
            COURSE_ARTIFACT_DIR / "course_decoder_step.onnx"
        ).exists():
            COURSE_RUNTIME = OnnxCourseGenerator(COURSE_ARTIFACT_DIR)
            COURSE_RUNTIME_ERROR = None
        else:
            COURSE_RUNTIME = None
            COURSE_RUNTIME_ERROR = "course decoder ONNX artifacts are missing"
    except Exception as exc:
        COURSE_RUNTIME = None
        COURSE_RUNTIME_ERROR = str(exc)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if RUNTIME is not None and COURSE_RUNTIME is not None else "degraded",
        "tiny_gru_loaded": RUNTIME is not None,
        "course_decoder_loaded": COURSE_RUNTIME is not None,
        "tiny_gru_artifact_dir": str(ARTIFACT_DIR),
        "course_decoder_artifact_dir": str(COURSE_ARTIFACT_DIR),
        "tiny_gru_error": RUNTIME_ERROR,
        "course_decoder_error": COURSE_RUNTIME_ERROR,
    }


def recommend_internal(request: RecommendRequest) -> RecommendResponse:
    if RUNTIME is None:
        raise HTTPException(status_code=503, detail="runtime is not loaded")

    content_id_to_token = RUNTIME["content_id_to_token"]
    unk_token = RUNTIME["unk_token"]
    tokens = [content_id_to_token.get(str(content_id), content_id_to_token[unk_token]) for content_id in request.content_id_sequence]
    tokens = tokens[-RUNTIME["max_sequence_len"] :]
    if not tokens:
        raise HTTPException(status_code=400, detail="content_id_sequence is empty")

    feature_df = pd.DataFrame([request.user_features])
    try:
        user_features = RUNTIME["feature_encoder"].transform(feature_df).astype(np.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid user_features: {exc}") from exc

    sequences = np.asarray([tokens], dtype=np.int64)
    lengths = np.asarray([len(tokens)], dtype=np.int64)
    logits = RUNTIME["session"].run(
        None,
        {
            "user_features": user_features,
            "sequences": sequences,
            "lengths": lengths,
        },
    )[0][0]

    top_k = min(request.top_k, logits.shape[0])
    order = np.argsort(-logits)[:top_k]
    recommendations = [
        RecommendItem(
            content_id=RUNTIME["token_to_content_id"].get(int(token_id), str(token_id)),
            token_id=int(token_id),
            score=float(logits[token_id]),
        )
        for token_id in order
    ]
    return RecommendResponse(recommendations=recommendations)


def generate_course_internal(request: GenerateCourseRequest) -> GenerateCourseResponse:
    if COURSE_RUNTIME is None:
        raise HTTPException(status_code=503, detail="course decoder runtime is not loaded")
    try:
        content_ids, steps = COURSE_RUNTIME.generate(
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


def fallback_content_ids_for_area(area_code: str) -> list[str]:
    return FALLBACK_CONTENT_IDS_BY_AREA[str(area_code).strip()]


def fallback_course_response(area_code: str, trip_days: int) -> GenerateCourseResponse:
    desired_poi_count = max(int(trip_days), 1) * 3
    area_content_ids = fallback_content_ids_for_area(area_code)
    content_ids = [area_content_ids[index % len(area_content_ids)] for index in range(desired_poi_count)]
    return GenerateCourseResponse(
        content_id_sequence=content_ids,
        steps=[
            GenerateCourseStep(
                rank=index + 1,
                day_index=index // 3 + 1,
                slot_index=index % 3 + 1,
                content_id=content_id,
                token_id=index + 3,
                score=1.0 - (index * 0.01),
            )
            for index, content_id in enumerate(content_ids)
        ],
    )


def fallback_recommend_response(area_code: str, content_id_sequence: list[str]) -> RecommendResponse:
    area_content_ids = fallback_content_ids_for_area(area_code)
    seen = {str(content_id) for content_id in content_id_sequence}
    candidates = [content_id for content_id in area_content_ids if content_id not in seen]
    if len(candidates) < BACKEND_RECOMMENDATION_TOP_K:
        candidates.extend(content_id for content_id in area_content_ids if content_id in seen)
    return RecommendResponse(
        recommendations=[
            RecommendItem(
                content_id=content_id,
                token_id=index + 3,
                score=1.0 - (index * 0.05),
            )
            for index, content_id in enumerate(candidates[:BACKEND_RECOMMENDATION_TOP_K])
        ]
    )


def recommend_for_backend(request: TravelSpotSuggestionsRequest) -> RecommendResponse:
    if RUNTIME is None:
        return fallback_recommend_response(request.areaCode, request.contentIdSequence)

    try:
        user_features, _ = travel_generate_request_to_user_features(request)
        content_id_to_token = RUNTIME["content_id_to_token"]
        unk_token = RUNTIME["unk_token"]
        tokens = [
            content_id_to_token.get(str(content_id), content_id_to_token[unk_token])
            for content_id in request.contentIdSequence
        ]
        tokens = tokens[-RUNTIME["max_sequence_len"] :]
        feature_df = pd.DataFrame([user_features])
        encoded_user_features = RUNTIME["feature_encoder"].transform(feature_df).astype(np.float32)
        sequences = np.asarray([tokens], dtype=np.int64)
        lengths = np.asarray([len(tokens)], dtype=np.int64)
        logits = RUNTIME["session"].run(
            None,
            {
                "user_features": encoded_user_features,
                "sequences": sequences,
                "lengths": lengths,
            },
        )[0][0]
        order = np.argsort(-logits)
        seen = {str(content_id) for content_id in request.contentIdSequence}
        recommendations: list[RecommendItem] = []
        for token_id in order:
            content_id = RUNTIME["token_to_content_id"].get(int(token_id), str(token_id))
            if content_id in seen:
                continue
            recommendations.append(
                RecommendItem(
                    content_id=content_id,
                    token_id=int(token_id),
                    score=float(logits[token_id]),
                )
            )
            if len(recommendations) == BACKEND_RECOMMENDATION_TOP_K:
                return RecommendResponse(recommendations=recommendations)
    except Exception:
        pass
    return fallback_recommend_response(request.areaCode, request.contentIdSequence)


@app.post("/generate-course", response_model=GenerateCourseResponse)
def generate_travel(request: TravelGenerateRequest) -> GenerateCourseResponse:
    try:
        user_features, trip_days = travel_generate_request_to_user_features(request)
        if COURSE_RUNTIME is None:
            return fallback_course_response(request.areaCode, trip_days)
        content_ids, steps = COURSE_RUNTIME.generate(
            user_features=user_features,
            trip_days=trip_days,
            desired_poi_count=None,
            duplicate_masking=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        trip_days = parse_int_choice("travelDuration", request.travelDuration, {1, 2, 3})
        return fallback_course_response(request.areaCode, trip_days)
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


@app.post("/recommend", response_model=RecommendResponse)
def suggest_travel_spots(request: TravelSpotSuggestionsRequest) -> RecommendResponse:
    try:
        return recommend_for_backend(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
