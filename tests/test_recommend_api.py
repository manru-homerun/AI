from __future__ import annotations

import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import AREA_FALLBACK_IDS, DummyFeatureEncoder, DummyRecommendSession
from src.api import tiny_gru_app
from src.inference import runtime as runtime_state


def test_suggest_travel_spots_returns_four_fallback_items(monkeypatch, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "RUNTIME", None)
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


def test_suggest_travel_spots_logs_model_unavailable_fallback(monkeypatch, caplog, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "RUNTIME", None)
    caplog.set_level(logging.WARNING)
    client = TestClient(tiny_gru_app.app)
    payload = {
        **backend_payload,
        "contentIdSequence": AREA_FALLBACK_IDS["11000"][:2],
    }
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    assert "fallback_reason=model_unavailable" in caplog.text


def test_suggest_travel_spots_accepts_empty_content_id_sequence(monkeypatch, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "RUNTIME", DummyRecommendSession(np.ones(8, dtype=np.float32)))
    client = TestClient(tiny_gru_app.app)
    payload = {
        **backend_payload,
        "companionCount": 0,
        "contentIdSequence": [],
        "gender": "남",
        "residenceArea": "11",
    }
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    recommended_ids = [item["content_id"] for item in response.json()["recommendations"]]
    assert recommended_ids == AREA_FALLBACK_IDS["11000"][:4]


def test_suggest_travel_spots_fallback_uses_area_specific_content_ids(monkeypatch, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "RUNTIME", None)
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
        runtime_state,
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


def test_suggest_travel_spots_rejects_invalid_travel_duration(monkeypatch, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "RUNTIME", None)
    client = TestClient(tiny_gru_app.app)
    payload = {**backend_payload, "travelDuration": "4", "contentIdSequence": AREA_FALLBACK_IDS["11000"][:2]}
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 400


def test_suggest_travel_spots_logs_inference_failure_before_fallback(monkeypatch, caplog, backend_payload) -> None:
    seoul_ids = AREA_FALLBACK_IDS["11000"]
    content_id_to_token = {content_id: index for index, content_id in enumerate(seoul_ids)}
    content_id_to_token["<UNK>"] = len(content_id_to_token)
    monkeypatch.setattr(
        runtime_state,
        "RUNTIME",
        {
            "session": DummyRecommendSession(np.zeros(len(content_id_to_token), dtype=np.float32)),
            "feature_encoder": object(),
            "content_id_to_token": content_id_to_token,
            "token_to_content_id": {index: content_id for content_id, index in content_id_to_token.items()},
            "unk_token": "<UNK>",
            "max_sequence_len": 8,
        },
    )
    client = TestClient(tiny_gru_app.app)
    payload = {**backend_payload, "contentIdSequence": seoul_ids[:2]}
    payload.pop("contentIdList")
    caplog.set_level(logging.ERROR)

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    assert "recommendation inference failed; using fallback recommendation response" in caplog.text
    assert "fallback_reason=inference_error" in caplog.text


def test_recommend_internal_does_not_convert_unexpected_errors_to_400(monkeypatch) -> None:
    monkeypatch.setattr(runtime_state, "RUNTIME", {"session": object()})

    def raise_unexpected(*_args, **_kwargs):
        raise RuntimeError("unexpected recommender failure")

    monkeypatch.setattr(tiny_gru_app, "_recommend_internal", raise_unexpected)

    with pytest.raises(RuntimeError):
        tiny_gru_app.recommend_internal(
            tiny_gru_app.RecommendRequest(content_id_sequence=["1"], user_features={}, top_k=1)
        )
