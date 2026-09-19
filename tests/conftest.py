from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import tiny_gru_app


AREA_FALLBACK_IDS = tiny_gru_app.FALLBACK_CONTENT_IDS_BY_AREA


@pytest.fixture
def backend_payload() -> dict:
    return {
        "areaCode": "11000",
        "contentIdList": AREA_FALLBACK_IDS["11000"][:2],
        "travelDuration": "2",
        "travelPersona": 3,
        "ageGroup": "30",
        "gender": "남",
        "travelerStyle": "4",
        "preferredArea": ["50110", "26350"],
        "residenceArea": "11000",
        "hasChild": 0,
        "hasElderly": 0,
        "hasDisabled": 0,
        "companionCount": 1,
    }


class DummyFeatureEncoder:
    def transform(self, feature_df):
        return np.zeros((len(feature_df), 1), dtype=np.float32)


class DummyRecommendSession:
    def __init__(self, logits: np.ndarray) -> None:
        self.logits = logits.astype(np.float32)

    def run(self, *_args, **_kwargs):
        return [self.logits.reshape(1, -1)]
