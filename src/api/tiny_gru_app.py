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
from pydantic import BaseModel, Field

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
    areaCode: str
    travelDuration: str
    travelPersona: str
    ageGroup: str
    gender: str
    travelerStyle: str
    preferredArea: str
    residenceArea: str
    hasChild: bool
    hasElderly: bool
    hasDisabled: bool
    companionCount: int = Field(ge=1)


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
    if request.gender not in {"남", "여"}:
        raise ValueError("gender must be one of ['남', '여']")

    persona_codes = split_codes(request.travelPersona)
    if not persona_codes:
        raise ValueError("travelPersona must contain at least one code")
    style_codes = split_codes(request.travelerStyle)
    if len(style_codes) != 1:
        raise ValueError("travelerStyle must contain exactly one code")
    preferred_area = normalize_code_list(request.preferredArea)
    if not preferred_area:
        raise ValueError("preferredArea must contain at least one code")
    residence_area = str(request.residenceArea).strip()
    if not residence_area:
        raise ValueError("residenceArea must not be empty")

    style = style_codes[0]
    user_features = {
        "area_code": str(request.areaCode).strip(),
        "trip_days": trip_days,
        "theme": persona_codes[0],
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
RUNTIME: dict[str, Any] | None = None
COURSE_RUNTIME: OnnxCourseGenerator | None = None
app = FastAPI(title="Tiny GRU POI Recommender")


@app.on_event("startup")
def startup() -> None:
    global RUNTIME, COURSE_RUNTIME
    RUNTIME = load_runtime(ARTIFACT_DIR)
    if (COURSE_ARTIFACT_DIR / "course_user_encoder.onnx").exists() and (COURSE_ARTIFACT_DIR / "course_decoder_step.onnx").exists():
        COURSE_RUNTIME = OnnxCourseGenerator(COURSE_ARTIFACT_DIR)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if RUNTIME is not None else "degraded",
        "tiny_gru_loaded": RUNTIME is not None,
        "course_decoder_loaded": COURSE_RUNTIME is not None,
        "tiny_gru_artifact_dir": str(ARTIFACT_DIR),
        "course_decoder_artifact_dir": str(COURSE_ARTIFACT_DIR),
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
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


@app.post("/generate-course", response_model=GenerateCourseResponse)
def generate_course(request: GenerateCourseRequest) -> GenerateCourseResponse:
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


@app.post("/api/travel/generate", response_model=GenerateCourseResponse)
def generate_travel(request: TravelGenerateRequest) -> GenerateCourseResponse:
    if COURSE_RUNTIME is None:
        raise HTTPException(status_code=503, detail="course decoder runtime is not loaded")
    try:
        user_features, trip_days = travel_generate_request_to_user_features(request)
        content_ids, steps = COURSE_RUNTIME.generate(
            user_features=user_features,
            trip_days=trip_days,
            desired_poi_count=None,
            duplicate_masking=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid travel generation request: {exc}") from exc
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
