from __future__ import annotations

import json
import os
import pickle
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
