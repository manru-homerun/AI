from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TINY_GRU_ARTIFACT_DIR = ROOT / "artifacts" / "tiny_gru_onnx_experiment"
DEFAULT_COURSE_DECODER_ARTIFACT_DIR = ROOT / "artifacts" / "conditional_gru_decoder_experiment"
BACKEND_RECOMMENDATION_TOP_K = 4
BACKEND_RECOMMENDATION_CANDIDATE_K = 10
COURSE_POIS_PER_DAY = 6


@dataclass(frozen=True)
class Settings:
    tiny_gru_artifact_dir: Path
    course_decoder_artifact_dir: Path
    backend_recommendation_top_k: int = BACKEND_RECOMMENDATION_TOP_K
    backend_recommendation_candidate_k: int = BACKEND_RECOMMENDATION_CANDIDATE_K


def load_settings() -> Settings:
    return Settings(
        tiny_gru_artifact_dir=Path(
            os.environ.get("TINY_GRU_ARTIFACT_DIR", DEFAULT_TINY_GRU_ARTIFACT_DIR)
        ).resolve(),
        course_decoder_artifact_dir=Path(
            os.environ.get("COURSE_DECODER_ARTIFACT_DIR", DEFAULT_COURSE_DECODER_ARTIFACT_DIR)
        ).resolve(),
    )


SETTINGS = load_settings()
