from __future__ import annotations

import json
import math
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.core.config import COURSE_POIS_PER_DAY, ROOT
from src.schemas.travel import RecommendItem


DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "shared_next_poi_gru_experiment" / "shared-next-poi-gru-v1"
MISSING_CONTEXT_TOKEN = "__MISSING_CONTEXT__"
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
SIDO_TO_SERVICE_AREA_CODE = {
    "11": "11000",
    "26": "26000",
    "27": "27000",
    "28": "28000",
    "29": "12000",
    "30": "30000",
    "41": "41110",
    "48": "48120",
}


@dataclass(frozen=True)
class GeneratedCourseStep:
    rank: int
    day_index: int
    slot_index: int
    content_id: str
    token_id: int
    score: float


class ContextFeatureEncoder:
    """Runtime-compatible class for notebook-created feature encoder pickles."""

    def _scalar_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        scalar_frame = df[self.scalar_context_columns]
        if hasattr(scalar_frame, "map"):
            normalized = scalar_frame.map(normalize_context)
        else:
            normalized = scalar_frame.applymap(normalize_context)
        return normalized.astype(str)

    def _preferred_values(self, df: pd.DataFrame) -> list[list[str]]:
        return [normalize_preferred_area_list(value) for value in df[self.preferred_area_column]]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        scalar_features = self.scalar_encoder.transform(self._scalar_frame(df)).astype(np.float32)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"unknown class\(es\).*will be ignored", category=UserWarning)
            preferred_features = self.preferred_encoder.transform(self._preferred_values(df)).astype(np.float32)
        return np.hstack([scalar_features, preferred_features]).astype(np.float32)


class _SharedFeatureUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if name == "ContextFeatureEncoder" and module in {"__main__", "builtins"}:
            return ContextFeatureEncoder
        return super().find_class(module, name)


def load_pickle_compat(path: Path) -> Any:
    with open(path, "rb") as fp:
        return _SharedFeatureUnpickler(fp).load()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def normalize_context(value: Any) -> str:
    if value is None or pd.isna(value):
        return MISSING_CONTEXT_TOKEN
    text = str(value).strip()
    return text if text else MISSING_CONTEXT_TOKEN


def normalize_content_id(value: Any) -> str:
    text = normalize_context(value)
    if text == MISSING_CONTEXT_TOKEN:
        return ""
    return text[:-2] if text.endswith(".0") else text


def normalize_area_code_5digit(value: Any) -> str:
    text = normalize_context(value)
    if text == MISSING_CONTEXT_TOKEN:
        return MISSING_CONTEXT_TOKEN
    text = text[:-2] if text.endswith(".0") else text
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 5:
        return digits
    if len(digits) == 2:
        return SIDO_TO_SERVICE_AREA_CODE.get(digits, f"{digits}000")
    return MISSING_CONTEXT_TOKEN


def normalize_preferred_area_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return [MISSING_CONTEXT_TOKEN]
    if isinstance(value, str):
        raw_values = [part for part in re.split(r"[;,\s]+", value.strip()) if part]
    elif isinstance(value, Iterable):
        raw_values = list(value)
    else:
        raw_values = [value]
    normalized: list[str] = []
    for item in raw_values:
        code = normalize_area_code_5digit(item)
        if code not in normalized:
            normalized.append(code)
    return normalized or [MISSING_CONTEXT_TOKEN]


def normalize_service_persona(value: Any) -> str:
    text = normalize_context(value)
    if text == MISSING_CONTEXT_TOKEN:
        return MISSING_CONTEXT_TOKEN
    text = text[:-2] if text.endswith(".0") else text
    try:
        parsed = int(text)
    except ValueError:
        return MISSING_CONTEXT_TOKEN
    return str(parsed) if 1 <= parsed <= 7 else MISSING_CONTEXT_TOKEN


def cap_companion_count(value: Any) -> str:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return MISSING_CONTEXT_TOKEN
    return str(min(max(parsed, 0), 2))


def split_codes(value: Any) -> list[str]:
    if value is None:
        return []
    return [part for part in re.split(r"[;,\s]+", str(value).strip()) if part]


def backend_features_to_shared_request(user_features: Mapping[str, Any]) -> dict[str, Any]:
    style_codes = split_codes(user_features.get("p0_style") or user_features.get("travelerStyle"))
    return {
        "areaCode": user_features.get("area_code", user_features.get("areaCode", MISSING_CONTEXT_TOKEN)),
        "travelPersona": user_features.get("theme", user_features.get("travelPersona", MISSING_CONTEXT_TOKEN)),
        "companionCount": user_features.get(
            "companion_count",
            user_features.get("companionCount", MISSING_CONTEXT_TOKEN),
        ),
        "preferredArea": user_features.get(
            "p0_preferred",
            user_features.get("preferredArea", [MISSING_CONTEXT_TOKEN]),
        ),
        "residenceArea": user_features.get(
            "p0_home",
            user_features.get("residenceArea", MISSING_CONTEXT_TOKEN),
        ),
        "gender": user_features.get("p0_gender", user_features.get("gender", MISSING_CONTEXT_TOKEN)),
        "ageGroup": user_features.get("p0_age", user_features.get("ageGroup", MISSING_CONTEXT_TOKEN)),
        "travelerStyle": style_codes[0] if style_codes else user_features.get("travelerStyle", MISSING_CONTEXT_TOKEN),
    }


class OnnxSharedNextPoiRuntime:
    def __init__(self, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> None:
        import onnxruntime as ort

        self.artifact_dir = artifact_dir.resolve()
        self.vocab = load_json(self.artifact_dir / "content_id_vocab.json")
        self.train_config = load_json(self.artifact_dir / "train_config.json")
        self.model_known_pois = load_json(self.artifact_dir / "model_known_pois.json")
        self.feature_encoder = load_pickle_compat(self.artifact_dir / "feature_encoder.pkl")
        self.session = ort.InferenceSession(str(self.artifact_dir / "model.onnx"), providers=["CPUExecutionProvider"])

        self.content_id_to_token: dict[str, int] = {
            str(content_id): int(token_id)
            for content_id, token_id in self.vocab["content_id_to_token"].items()
        }
        self.token_to_content_id: dict[int, str] = {
            int(token_id): str(content_id)
            for token_id, content_id in self.vocab["token_to_content_id"].items()
        }
        self.pad_token_id = int(self.train_config.get("pad_token_id", self.content_id_to_token[PAD_TOKEN]))
        self.bos_token_id = int(self.train_config.get("bos_token_id", self.content_id_to_token[BOS_TOKEN]))
        self.special_token_ids = {self.pad_token_id, self.bos_token_id}
        self.max_sequence_len = int(self.train_config["max_sequence_len"])

        scalar_columns = list(self.train_config.get("scalar_context_columns", []))
        preferred_column = str(self.train_config.get("preferred_area_column", "preferred_area_codes"))
        if isinstance(self.feature_encoder, ContextFeatureEncoder):
            self.feature_encoder.scalar_context_columns = scalar_columns
            self.feature_encoder.preferred_area_column = preferred_column

        self.poi_master = pd.read_csv(self.artifact_dir / "poi_master.csv", dtype={"content_id": str, "area_code": str})
        self.poi_master["content_id"] = self.poi_master["content_id"].map(normalize_content_id)
        self.poi_master["area_code"] = self.poi_master["area_code"].map(normalize_area_code_5digit)
        self.poi_master["token_id"] = self.poi_master["token_id"].astype(int)
        self.poi_master["visit_count"] = pd.to_numeric(self.poi_master.get("visit_count", 1), errors="coerce").fillna(1).astype(int)
        self.master_content_ids = set(self.poi_master["content_id"].astype(str))
        self.content_id_to_area_code = dict(zip(self.poi_master["content_id"].astype(str), self.poi_master["area_code"].astype(str)))
        self.content_id_to_visit_count = dict(zip(self.poi_master["content_id"].astype(str), self.poi_master["visit_count"].astype(int)))

        self.train_known_input_content_ids = {
            normalize_content_id(content_id)
            for content_id in self.model_known_pois.get("train_known_input_content_ids", [])
        }
        self.train_output_candidate_content_ids = {
            normalize_content_id(content_id)
            for content_id in self.model_known_pois.get("train_output_candidate_content_ids", [])
        }
        vocab_size = len(self.content_id_to_token)
        self.token_area_code = np.full(vocab_size, "", dtype=object)
        self.valid_token_mask = np.zeros(vocab_size, dtype=bool)
        for row in self.poi_master.itertuples(index=False):
            token_id = int(row.token_id)
            self.token_area_code[token_id] = str(row.area_code)
            self.valid_token_mask[token_id] = True
        self.valid_token_mask[list(self.special_token_ids)] = False
        self.train_output_candidate_token_mask = np.zeros(vocab_size, dtype=bool)
        for content_id in self.train_output_candidate_content_ids:
            token_id = self.content_id_to_token.get(content_id)
            if token_id is not None:
                self.train_output_candidate_token_mask[int(token_id)] = True
        self.train_output_candidate_token_mask[list(self.special_token_ids)] = False

    def request_to_feature_frame(self, user_features: Mapping[str, Any]) -> pd.DataFrame:
        request = backend_features_to_shared_request(user_features)
        row = {
            "companion_count_bucket": cap_companion_count(request.get("companionCount")),
            "travel_persona": normalize_service_persona(request.get("travelPersona")),
            "residence_area_code": normalize_area_code_5digit(request.get("residenceArea")),
            "gender": normalize_context(request.get("gender")),
            "age_grp": normalize_context(request.get("ageGroup")),
            "traveler_style": normalize_context(request.get("travelerStyle")),
            "preferred_area_codes": normalize_preferred_area_list(request.get("preferredArea")),
        }
        return pd.DataFrame([row])

    def encode_user_features(self, user_features: Mapping[str, Any]) -> np.ndarray:
        return self.feature_encoder.transform(self.request_to_feature_frame(user_features)).astype(np.float32)

    def project_known_sequence(self, content_id_sequence: Iterable[Any]) -> list[str]:
        known: list[str] = []
        for value in content_id_sequence:
            content_id = normalize_content_id(value)
            if content_id in self.train_known_input_content_ids:
                known.append(content_id)
        return known

    def tokens_for_prefix(self, prefix_content_ids: Iterable[Any]) -> list[int]:
        known_tokens = [
            int(self.content_id_to_token[content_id])
            for content_id in self.project_known_sequence(prefix_content_ids)
        ]
        max_known_tokens = max(self.max_sequence_len - 1, 0)
        return [self.bos_token_id, *known_tokens[-max_known_tokens:]]

    def run_logits(self, user_features: Mapping[str, Any], prefix_content_ids: list[str]) -> np.ndarray:
        prefix_tokens = self.tokens_for_prefix(prefix_content_ids)
        logits = self.session.run(
            None,
            {
                "user_features": self.encode_user_features(user_features),
                "sequences": np.asarray([prefix_tokens], dtype=np.int64),
                "lengths": np.asarray([len(prefix_tokens)], dtype=np.int64),
            },
        )[0][0]
        return logits.astype(np.float32)

    def area_token_mask(self, area_code: str) -> np.ndarray:
        normalized = normalize_area_code_5digit(area_code)
        return self.valid_token_mask & (self.token_area_code == normalized)

    def apply_candidate_mask(
        self,
        logits: np.ndarray,
        area_code: str,
        excluded_content_ids: Iterable[Any] = (),
        restrict_to_train_outputs: bool = True,
    ) -> np.ndarray:
        scores = np.asarray(logits, dtype=np.float32).copy()
        candidate_mask = self.area_token_mask(area_code)
        if restrict_to_train_outputs:
            candidate_mask = candidate_mask & self.train_output_candidate_token_mask
        scores[~candidate_mask] = -np.inf
        scores[list(self.special_token_ids)] = -np.inf
        for content_id in excluded_content_ids:
            token_id = self.content_id_to_token.get(normalize_content_id(content_id))
            if token_id is not None:
                scores[int(token_id)] = -np.inf
        return scores

    def top_items_from_scores(self, scores: np.ndarray, top_k: int) -> list[RecommendItem]:
        finite = np.flatnonzero(np.isfinite(scores))
        order = finite[np.argsort(-scores[finite])]
        return [
            RecommendItem(
                content_id=self.token_to_content_id[int(token_id)],
                token_id=int(token_id),
                score=float(scores[int(token_id)]),
            )
            for token_id in order[:top_k]
        ]

    def fallback_top_k(self, area_code: str, excluded_content_ids: Iterable[Any] = (), top_k: int = 4) -> list[RecommendItem]:
        excluded = {normalize_content_id(value) for value in excluded_content_ids}
        normalized_area_code = normalize_area_code_5digit(area_code)
        rows = self.poi_master.loc[self.poi_master["area_code"].eq(normalized_area_code)].copy()
        rows = rows.sort_values(["visit_count", "content_id"], ascending=[False, True], kind="mergesort")
        rows = rows.loc[~rows["content_id"].isin(excluded)].head(top_k)
        return [
            RecommendItem(
                content_id=str(row.content_id),
                token_id=int(row.token_id),
                score=1.0 - rank * 0.05,
            )
            for rank, row in enumerate(rows.itertuples(index=False))
        ]

    def recommend(
        self,
        *,
        user_features: Mapping[str, Any],
        area_code: str,
        content_id_sequence: list[str],
        top_k: int,
    ) -> list[RecommendItem]:
        original_sequence = [normalize_content_id(content_id) for content_id in content_id_sequence]
        known_sequence = self.project_known_sequence(original_sequence)
        if not known_sequence:
            return []
        scores = self.apply_candidate_mask(
            self.run_logits(user_features, known_sequence),
            area_code,
            original_sequence,
        )
        recommendations = self.top_items_from_scores(scores, top_k)
        if len(recommendations) < top_k:
            excluded = [*original_sequence, *[item.content_id for item in recommendations]]
            recommendations.extend(self.fallback_top_k(area_code, excluded, top_k - len(recommendations)))
        return recommendations[:top_k]

    def validate_required_content_ids(self, area_code: str, content_ids: Iterable[Any], target_length: int) -> list[str]:
        normalized_area_code = normalize_area_code_5digit(area_code)
        required = {normalize_content_id(content_id) for content_id in content_ids}
        required.discard("")
        if len(required) > target_length:
            raise ValueError("contentIdList cannot contain more unique items than target course length")
        for content_id in required:
            if content_id not in self.master_content_ids:
                raise ValueError(f"contentIdList contains a POI outside the model candidate set: {content_id}")
            if self.content_id_to_area_code.get(content_id) != normalized_area_code:
                raise ValueError(f"contentIdList contains a POI outside areaCode {normalized_area_code}: {content_id}")
        return sorted(required)

    def rank_required_candidates(self, required_ids: Iterable[str], scores: np.ndarray | None = None) -> list[str]:
        candidates = sorted({normalize_content_id(content_id) for content_id in required_ids})
        if scores is not None:
            scored_candidates = []
            for content_id in candidates:
                token_id = self.content_id_to_token.get(content_id)
                if token_id is not None and np.isfinite(scores[int(token_id)]):
                    scored_candidates.append(
                        (
                            float(scores[int(token_id)]),
                            int(self.content_id_to_visit_count.get(content_id, 0)),
                            content_id,
                        )
                    )
            if scored_candidates:
                return [item[-1] for item in sorted(scored_candidates, key=lambda item: (-item[0], -item[1], item[2]))]
        return sorted(candidates, key=lambda content_id: (-int(self.content_id_to_visit_count.get(content_id, 0)), content_id))

    def generate(
        self,
        user_features: Mapping[str, Any],
        trip_days: int,
        desired_poi_count: int | None = None,
        duplicate_masking: bool = True,
        forced_content_ids: list[str] | None = None,
        allowed_content_ids: list[str] | None = None,
    ) -> tuple[list[str], list[GeneratedCourseStep]]:
        trip_days = max(int(trip_days), 1)
        desired = int(desired_poi_count) if desired_poi_count is not None else trip_days * COURSE_POIS_PER_DAY
        desired = max(desired, 1)
        area_code = normalize_area_code_5digit(user_features.get("area_code", user_features.get("areaCode", "")))
        required = self.validate_required_content_ids(area_code, forced_content_ids or [], desired)
        allowed_set = {normalize_content_id(content_id) for content_id in allowed_content_ids or []}
        if allowed_set:
            required_outside_allowed = set(required) - allowed_set
            if required_outside_allowed:
                raise ValueError("contentIdList contains POIs outside allowed_content_ids")

        all_generated: list[str] = []
        steps: list[GeneratedCourseStep] = []
        for day_index in range(1, int(math.ceil(desired / COURSE_POIS_PER_DAY)) + 1):
            current_day_sequence: list[str] = []
            for slot_index in range(1, COURSE_POIS_PER_DAY + 1):
                if len(all_generated) >= desired:
                    break
                remaining_slots = desired - len(all_generated)
                missing_required = sorted(set(required) - set(all_generated))
                excluded = all_generated if duplicate_masking else []
                scores = self.apply_candidate_mask(
                    self.run_logits(user_features, current_day_sequence),
                    area_code,
                    excluded,
                )
                if allowed_set:
                    for token_id, content_id in self.token_to_content_id.items():
                        if content_id not in allowed_set:
                            scores[int(token_id)] = -np.inf
                if missing_required and len(missing_required) >= remaining_slots:
                    next_content_id = self.rank_required_candidates(missing_required, scores)[0]
                    token_id = int(self.content_id_to_token[next_content_id])
                    selected_score = float(scores[token_id]) if np.isfinite(scores[token_id]) else 0.0
                else:
                    next_items = self.top_items_from_scores(scores, 1)
                    if next_items:
                        next_content_id = next_items[0].content_id
                        token_id = int(next_items[0].token_id)
                        selected_score = float(next_items[0].score)
                    else:
                        fallback_items = self.fallback_top_k(area_code, excluded, 1)
                        if not fallback_items:
                            raise ValueError("no valid content_id candidates are available")
                        next_content_id = fallback_items[0].content_id
                        token_id = int(fallback_items[0].token_id)
                        selected_score = float(fallback_items[0].score)
                all_generated.append(next_content_id)
                current_day_sequence.append(next_content_id)
                steps.append(
                    GeneratedCourseStep(
                        rank=len(all_generated),
                        day_index=day_index,
                        slot_index=slot_index,
                        content_id=next_content_id,
                        token_id=token_id,
                        score=selected_score,
                    )
                )
        return all_generated, steps
