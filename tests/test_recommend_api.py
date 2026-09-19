from __future__ import annotations

import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import AREA_FALLBACK_IDS, DummyFeatureEncoder, DummyRecommendSession
from src.api import tiny_gru_app
from src.fallback.travel import (
    CENTRAL_TOURISM_BASE_YM,
    CENTRAL_TOURISM_API_URL,
    TOURAPI_AREA_PARAMS_BY_BACKEND_AREA,
    build_central_tourism_recommendation_payload,
)
from src.inference import runtime as runtime_state
from src.services import travel_service


CENTRAL_TOURISM_RECOMMENDATIONS = [
    {"content_id": "central-1", "token_id": 3, "score": 1.0},
    {"content_id": "central-2", "token_id": 4, "score": 0.95},
    {"content_id": "central-3", "token_id": 5, "score": 0.9},
    {"content_id": "central-4", "token_id": 6, "score": 0.85},
]


def test_central_tourism_api_url_uses_current_locgo_service() -> None:
    assert CENTRAL_TOURISM_API_URL == "https://apis.data.go.kr/B551011/LocgoHubTarService1/areaBasedList1"


def test_central_tourism_uses_manual_area_params_and_hub_tats_code(monkeypatch) -> None:
    assert CENTRAL_TOURISM_BASE_YM == "202504"
    assert TOURAPI_AREA_PARAMS_BY_BACKEND_AREA == {
        "11000": {"areaCd": "11", "signguCd": "11710"},
        "41110": {"areaCd": "41", "signguCd": "41111"},
        "28000": {"areaCd": "28", "signguCd": "28177"},
        "30000": {"areaCd": "30", "signguCd": "30140"},
        "27000": {"areaCd": "27", "signguCd": "27260"},
        "12000": {"areaCd": "29", "signguCd": "29170"},
        "26000": {"areaCd": "26", "signguCd": "26260"},
        "48120": {"areaCd": "48", "signguCd": "48127"},
    }

    monkeypatch.setattr(
        "src.fallback.travel.fetch_central_tourism_items",
        lambda area_code, top_k: [
            {"hubTatsCd": f"hub-{index}", "hubTatsNm": f"spot-{index}"}
            for index in range(1, top_k + 1)
        ],
    )

    recommendations = build_central_tourism_recommendation_payload("11000", 4)

    assert [item["content_id"] for item in recommendations] == ["hub-1", "hub-2", "hub-3", "hub-4"]


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
    record = next(item for item in caplog.records if item.event == "recommend_fallback")
    assert record.endpoint == "/recommend"
    assert record.area_code == backend_payload["areaCode"]
    assert record.trip_days == 2
    assert record.runtime == "tiny_gru"
    assert record.fallback_reason == "model_unavailable"


def test_suggest_travel_spots_empty_sequence_uses_central_tourism_fallback(
    monkeypatch, caplog, backend_payload
) -> None:
    monkeypatch.setattr(runtime_state, "RUNTIME", DummyRecommendSession(np.ones(8, dtype=np.float32)))
    monkeypatch.setattr(
        travel_service,
        "build_central_tourism_recommendation_payload",
        lambda area_code, top_k: CENTRAL_TOURISM_RECOMMENDATIONS[:top_k],
    )
    caplog.set_level(logging.INFO)
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
    assert recommended_ids == ["central-1", "central-2", "central-3", "central-4"]
    record = next(item for item in caplog.records if item.event == "recommend_fallback")
    assert record.fallback_reason == "empty_sequence_central_tourism"


def test_suggest_travel_spots_empty_sequence_falls_back_to_static_when_central_tourism_fails(
    monkeypatch, backend_payload
) -> None:
    monkeypatch.setattr(runtime_state, "RUNTIME", DummyRecommendSession(np.ones(8, dtype=np.float32)))

    def fail_central_tourism(*_args, **_kwargs):
        raise RuntimeError("central tourism unavailable")

    monkeypatch.setattr(travel_service, "build_central_tourism_recommendation_payload", fail_central_tourism)
    client = TestClient(tiny_gru_app.app)
    payload = {
        **backend_payload,
        "contentIdSequence": [],
    }
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    recommended_ids = [item["content_id"] for item in response.json()["recommendations"]]
    assert recommended_ids == AREA_FALLBACK_IDS["11000"][:4]


def test_suggest_travel_spots_full_course_sequence_uses_central_tourism_fallback(
    monkeypatch, caplog, backend_payload
) -> None:
    monkeypatch.setattr(
        runtime_state,
        "RUNTIME",
        {
            "session": DummyRecommendSession(np.ones(8, dtype=np.float32)),
            "feature_encoder": DummyFeatureEncoder(),
            "content_id_to_token": {"<UNK>": 0},
            "token_to_content_id": {},
            "unk_token": "<UNK>",
            "max_sequence_len": 8,
        },
    )
    monkeypatch.setattr(
        travel_service,
        "build_central_tourism_recommendation_payload",
        lambda area_code, top_k: CENTRAL_TOURISM_RECOMMENDATIONS[:top_k],
    )

    def fail_model_recommendation(*_args, **_kwargs):
        raise AssertionError("model recommendation should be skipped")

    monkeypatch.setattr(travel_service, "recommend_backend_area_limited", fail_model_recommendation)
    caplog.set_level(logging.INFO)
    client = TestClient(tiny_gru_app.app)
    payload = {**backend_payload, "contentIdSequence": [str(index) for index in range(12)]}
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    recommended_ids = [item["content_id"] for item in response.json()["recommendations"]]
    assert recommended_ids == ["central-1", "central-2", "central-3", "central-4"]
    record = next(item for item in caplog.records if item.event == "recommend_fallback")
    assert record.fallback_reason == "full_course_sequence"


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
    record = next(item for item in caplog.records if item.event == "recommend_inference_failure")
    assert record.endpoint == "/recommend"
    assert record.area_code == backend_payload["areaCode"]
    assert record.trip_days == 2
    assert record.runtime == "tiny_gru"
    assert record.fallback_reason == "inference_error"


def test_recommend_internal_does_not_convert_unexpected_errors_to_400(monkeypatch) -> None:
    monkeypatch.setattr(runtime_state, "RUNTIME", {"session": object()})

    def raise_unexpected(*_args, **_kwargs):
        raise RuntimeError("unexpected recommender failure")

    monkeypatch.setattr(tiny_gru_app, "_recommend_internal", raise_unexpected)

    with pytest.raises(RuntimeError):
        tiny_gru_app.recommend_internal(
            tiny_gru_app.RecommendRequest(content_id_sequence=["1"], user_features={}, top_k=1)
        )
