from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from conftest import AREA_FALLBACK_IDS, DummyFeatureEncoder, DummyRecommendSession
from src.api import tiny_gru_app


def test_suggest_travel_spots_returns_four_fallback_items(monkeypatch, backend_payload) -> None:
    monkeypatch.setattr(tiny_gru_app, "RUNTIME", None)
    client = TestClient(tiny_gru_app.app)
    payload = {
        **backend_payload,
        "contentIdSequence": AREA_FALLBACK_IDS["11000"][:2],
    }
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    recommendations = response.json()["recommendations"]
    assert len(recommendations) == 4
    assert {item["content_id"] for item in recommendations}.isdisjoint(payload["contentIdSequence"])
    assert {item["content_id"] for item in recommendations}.issubset(AREA_FALLBACK_IDS["11000"])


def test_suggest_travel_spots_fallback_uses_area_specific_content_ids(monkeypatch, backend_payload) -> None:
    monkeypatch.setattr(tiny_gru_app, "RUNTIME", None)
    client = TestClient(tiny_gru_app.app)

    for area_code, area_content_ids in AREA_FALLBACK_IDS.items():
        payload = {
            **backend_payload,
            "areaCode": area_code,
            "contentIdSequence": area_content_ids[:2],
        }
        payload.pop("contentIdList")
        response = client.post("/recommend", json=payload)

        assert response.status_code == 200
        recommendations = response.json()["recommendations"]
        recommended_ids = [item["content_id"] for item in recommendations]
        assert len(recommended_ids) == 4
        assert set(recommended_ids).issubset(area_content_ids)
        assert set(recommended_ids).isdisjoint(area_content_ids[:2])
        assert recommended_ids == area_content_ids[2:6]


def test_suggest_travel_spots_runtime_filters_to_area(monkeypatch, backend_payload) -> None:
    seoul_ids = AREA_FALLBACK_IDS["11000"]
    busan_ids = AREA_FALLBACK_IDS["26000"]
    content_ids = [*busan_ids[:4], *seoul_ids[:6]]
    token_to_content_id = {index: content_id for index, content_id in enumerate(content_ids)}
    content_id_to_token = {content_id: index for index, content_id in token_to_content_id.items()}
    content_id_to_token["<UNK>"] = len(content_id_to_token)
    logits = np.arange(len(content_id_to_token), 0, -1, dtype=np.float32)
    monkeypatch.setattr(
        tiny_gru_app,
        "RUNTIME",
        {
            "session": DummyRecommendSession(logits),
            "feature_encoder": DummyFeatureEncoder(),
            "content_id_to_token": content_id_to_token,
            "token_to_content_id": token_to_content_id,
            "unk_token": "<UNK>",
            "max_sequence_len": 8,
        },
    )
    client = TestClient(tiny_gru_app.app)
    payload = {**backend_payload, "contentIdSequence": seoul_ids[:2]}
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    recommended_ids = [item["content_id"] for item in response.json()["recommendations"]]
    assert len(recommended_ids) == 4
    assert set(recommended_ids).issubset(seoul_ids)
    assert set(recommended_ids).isdisjoint(payload["contentIdSequence"])
