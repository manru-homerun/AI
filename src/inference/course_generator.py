from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "conditional_gru_decoder_experiment"


@dataclass(frozen=True)
class GeneratedCourseStep:
    rank: int
    day_index: int
    slot_index: int
    content_id: str
    token_id: int
    score: float


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def build_step_feature_rows(trip_days: int, desired_poi_count: int) -> list[dict[str, int]]:
    trip_days = max(int(trip_days), 1)
    desired_poi_count = max(int(desired_poi_count), 1)
    poi_per_day = max(int(math.ceil(desired_poi_count / trip_days)), 1)
    rows = []
    for step_idx in range(desired_poi_count):
        day_index = min(step_idx // poi_per_day + 1, trip_days)
        slot_index = step_idx % poi_per_day + 1
        rows.append(
            {
                "day_index": day_index,
                "slot_index": slot_index,
                "absolute_step_index": step_idx + 1,
                "remaining_poi_count": desired_poi_count - step_idx,
                "remaining_days": max(trip_days - day_index + 1, 0),
                "is_day_start": int(slot_index == 1),
            }
        )
    return rows


class OnnxCourseGenerator:
    def __init__(self, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> None:
        import onnxruntime as ort

        self.artifact_dir = artifact_dir.resolve()
        self.vocab = load_json(self.artifact_dir / "content_id_vocab.json")
        self.train_config = load_json(self.artifact_dir / "train_config.json")
        with open(self.artifact_dir / "static_feature_encoder.pkl", "rb") as fp:
            self.static_feature_encoder = pickle.load(fp)
        with open(self.artifact_dir / "step_feature_encoder.pkl", "rb") as fp:
            self.step_feature_encoder = pickle.load(fp)
        self.user_session = ort.InferenceSession(str(self.artifact_dir / "course_user_encoder.onnx"), providers=["CPUExecutionProvider"])
        self.decoder_session = ort.InferenceSession(str(self.artifact_dir / "course_decoder_step.onnx"), providers=["CPUExecutionProvider"])
        self.content_id_to_token = self.vocab["content_id_to_token"]
        self.token_to_content_id = {int(key): value for key, value in self.vocab["token_to_content_id"].items()}
        self.pad_token_id = int(self.content_id_to_token[self.vocab["pad_token"]])
        self.unk_token_id = int(self.content_id_to_token[self.vocab["unk_token"]])
        self.start_token_id = int(self.content_id_to_token[self.vocab["start_token"]])
        self.special_token_ids = (self.pad_token_id, self.unk_token_id, self.start_token_id)
        self.static_feature_columns = list(self.train_config["static_feature_columns"])
        self.step_feature_columns = list(self.train_config["step_feature_columns"])

    def encode_static_features(
        self,
        user_features: dict[str, Any],
        trip_days: int,
        desired_poi_count: int,
    ) -> np.ndarray:
        row = dict(user_features)
        row["trip_days"] = trip_days
        row["desired_poi_count"] = desired_poi_count
        row["poi_per_day"] = max(int(math.ceil(desired_poi_count / max(trip_days, 1))), 1)
        feature_df = pd.DataFrame([row], columns=self.static_feature_columns)
        return self.static_feature_encoder.transform(feature_df).astype(np.float32)

    def encode_step_features(self, trip_days: int, desired_poi_count: int) -> tuple[list[dict[str, int]], np.ndarray]:
        rows = build_step_feature_rows(trip_days, desired_poi_count)
        step_df = pd.DataFrame(rows, columns=self.step_feature_columns)
        return rows, self.step_feature_encoder.transform(step_df.astype(np.float32)).astype(np.float32)

    def generate(
        self,
        user_features: dict[str, Any],
        trip_days: int,
        desired_poi_count: int | None = None,
        duplicate_masking: bool = True,
    ) -> tuple[list[str], list[GeneratedCourseStep]]:
        trip_days = max(int(trip_days), 1)
        desired = int(desired_poi_count) if desired_poi_count is not None else trip_days * 3
        desired = max(desired, 1)
        static_features = self.encode_static_features(user_features, trip_days, desired)
        step_rows, step_features = self.encode_step_features(trip_days, desired)
        hidden = self.user_session.run(None, {"static_features": static_features})[0].astype(np.float32)
        current_token = np.asarray([self.start_token_id], dtype=np.int64)
        generated_tokens: list[int] = []
        steps: list[GeneratedCourseStep] = []
        for step_idx in range(desired):
            logits, hidden = self.decoder_session.run(
                None,
                {
                    "current_token": current_token,
                    "hidden_state": hidden.astype(np.float32),
                    "step_features": step_features[step_idx : step_idx + 1],
                },
            )
            scores = logits[0].astype(np.float32)
            scores[list(self.special_token_ids)] = -1e9
            if duplicate_masking and generated_tokens:
                scores[generated_tokens] = -1e9
            if float(scores.max()) <= -1e8:
                scores = logits[0].astype(np.float32)
                scores[list(self.special_token_ids)] = -1e9
            token_id = int(np.argmax(scores))
            generated_tokens.append(token_id)
            current_token = np.asarray([token_id], dtype=np.int64)
            content_id = self.token_to_content_id.get(token_id, str(token_id))
            steps.append(
                GeneratedCourseStep(
                    rank=step_idx + 1,
                    day_index=int(step_rows[step_idx]["day_index"]),
                    slot_index=int(step_rows[step_idx]["slot_index"]),
                    content_id=content_id,
                    token_id=token_id,
                    score=float(scores[token_id]),
                )
            )
        return [step.content_id for step in steps], steps
