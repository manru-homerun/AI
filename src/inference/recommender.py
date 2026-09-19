from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.schemas.travel import RecommendItem, RecommendRequest, RecommendResponse


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


def recommend_internal(runtime: dict[str, Any], request: RecommendRequest) -> RecommendResponse:
    content_id_to_token = runtime["content_id_to_token"]
    unk_token = runtime["unk_token"]
    tokens = [
        content_id_to_token.get(str(content_id), content_id_to_token[unk_token])
        for content_id in request.content_id_sequence
    ]
    tokens = tokens[-runtime["max_sequence_len"] :]
    if not tokens:
        raise ValueError("content_id_sequence is empty")

    feature_df = pd.DataFrame([request.user_features])
    user_features = runtime["feature_encoder"].transform(feature_df).astype(np.float32)
    sequences = np.asarray([tokens], dtype=np.int64)
    lengths = np.asarray([len(tokens)], dtype=np.int64)
    logits = runtime["session"].run(
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
            content_id=runtime["token_to_content_id"].get(int(token_id), str(token_id)),
            token_id=int(token_id),
            score=float(logits[token_id]),
        )
        for token_id in order
    ]
    return RecommendResponse(recommendations=recommendations)


def recommend_backend_area_limited(
    runtime: dict[str, Any],
    content_id_sequence: list[str],
    user_features: dict[str, Any],
    allowed_content_ids: set[str],
    top_k: int,
) -> list[RecommendItem]:
    content_id_to_token = runtime["content_id_to_token"]
    unk_token = runtime["unk_token"]
    tokens = [
        content_id_to_token.get(str(content_id), content_id_to_token[unk_token])
        for content_id in content_id_sequence
    ]
    tokens = tokens[-runtime["max_sequence_len"] :]
    feature_df = pd.DataFrame([user_features])
    encoded_user_features = runtime["feature_encoder"].transform(feature_df).astype(np.float32)
    sequences = np.asarray([tokens], dtype=np.int64)
    lengths = np.asarray([len(tokens)], dtype=np.int64)
    logits = runtime["session"].run(
        None,
        {
            "user_features": encoded_user_features,
            "sequences": sequences,
            "lengths": lengths,
        },
    )[0][0]
    order = np.argsort(-logits)
    seen = {str(content_id) for content_id in content_id_sequence}
    recommendations: list[RecommendItem] = []
    for token_id in order:
        content_id = runtime["token_to_content_id"].get(int(token_id), str(token_id))
        if content_id in seen or content_id not in allowed_content_ids:
            continue
        recommendations.append(
            RecommendItem(
                content_id=content_id,
                token_id=int(token_id),
                score=float(logits[token_id]),
            )
        )
        if len(recommendations) == top_k:
            break
    return recommendations
