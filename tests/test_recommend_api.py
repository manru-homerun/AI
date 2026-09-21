from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from conftest import AREA_FALLBACK_IDS, DummySharedRecommendRuntime
from src.api import tiny_gru_app
from src.fallback.travel import (
    CENTRAL_TOURISM_BASE_YM,
    CENTRAL_TOURISM_API_URL,
    TOURAPI_AREA_PARAMS_BY_BACKEND_AREA,
    build_central_tourism_recommendation_payload,
)
from src.inference import runtime as runtime_state
from src.services import travel_service
from src.services import accessibility


CENTRAL_TOURISM_RECOMMENDATIONS = [
    {"content_id": "central-1", "token_id": 3, "score": 1.0},
    {"content_id": "central-2", "token_id": 4, "score": 0.95},
    {"content_id": "central-3", "token_id": 5, "score": 0.9},
    {"content_id": "central-4", "token_id": 6, "score": 0.85},
]


class TrackingSharedRecommendRuntime(DummySharedRecommendRuntime):
    def __init__(self, recommendations=None) -> None:
        super().__init__(recommendations=recommendations)
        self.last_user_features = None

    def recommend(self, *, user_features, area_code, content_id_sequence, top_k):
        self.last_user_features = user_features
        return super().recommend(
            user_features=user_features,
            area_code=area_code,
            content_id_sequence=content_id_sequence,
            top_k=top_k,
        )


def test_central_tourism_api_url_uses_current_locgo_service() -> None:
    assert CENTRAL_TOURISM_API_URL == "https://apis.data.go.kr/B551011/LocgoHubTarService1/areaBasedList1"


def test_central_tourism_uses_manual_area_params_and_tourapi_content_id(monkeypatch) -> None:
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
            {"contentid": f"{1000 + index}", "hubTatsCd": f"hub-{index}", "hubTatsNm": f"spot-{index}"}
            for index in range(1, top_k + 1)
        ],
    )

    recommendations = build_central_tourism_recommendation_payload("11000", 4)

    assert [item["content_id"] for item in recommendations] == ["1001", "1002", "1003", "1004"]


def test_central_tourism_rejects_hub_tats_code_as_content_id(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.fallback.travel.fetch_central_tourism_items",
        lambda area_code, top_k: [
            {"hubTatsCd": f"hub-{index}", "hubTatsNm": f"spot-{index}"}
            for index in range(1, top_k + 1)
        ],
    )

    with pytest.raises(RuntimeError, match="too few items"):
        build_central_tourism_recommendation_payload("11000", 4)


def test_suggest_travel_spots_returns_four_fallback_items(monkeypatch, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)
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


def test_suggest_travel_spots_truncates_overlong_preferred_area(
    monkeypatch, caplog, backend_payload
) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)
    caplog.set_level(logging.WARNING)
    client = TestClient(tiny_gru_app.app)
    payload = {
        **backend_payload,
        "contentIdSequence": AREA_FALLBACK_IDS["11000"][:2],
        "preferredArea": [
            "11000",
            "26000",
            "27000",
            "28000",
            "29000",
            "30000",
            "31000",
            "41000",
            "42000",
            "43000",
            "44000",
            "45000",
            "46000",
            "47000",
            "48000",
            "50000",
        ],
    }
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    record = next(item for item in caplog.records if item.event == "preferred_area_truncated")
    assert record.original_count == 16
    assert record.normalized_count == 3


def test_suggest_travel_spots_logs_model_unavailable_fallback(monkeypatch, caplog, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)
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
    assert record.runtime == "shared_next_poi_gru"
    assert record.fallback_reason == "model_unavailable"


def test_suggest_travel_spots_empty_sequence_uses_central_tourism_fallback(
    monkeypatch, caplog, backend_payload
) -> None:
    runtime = DummySharedRecommendRuntime()
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", runtime)
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
    assert runtime.call_count == 0
    record = next(item for item in caplog.records if item.event == "recommend_fallback")
    assert record.fallback_reason == "empty_sequence_central_tourism"


def test_suggest_travel_spots_empty_sequence_falls_back_to_static_when_central_tourism_fails(
    monkeypatch, backend_payload
) -> None:
    runtime = DummySharedRecommendRuntime()
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", runtime)

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
    assert runtime.call_count == 0


def test_suggest_travel_spots_full_course_sequence_uses_central_tourism_fallback(
    monkeypatch, caplog, backend_payload
) -> None:
    runtime = DummySharedRecommendRuntime()
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", runtime)
    monkeypatch.setattr(
        travel_service,
        "build_central_tourism_recommendation_payload",
        lambda area_code, top_k: CENTRAL_TOURISM_RECOMMENDATIONS[:top_k],
    )

    caplog.set_level(logging.INFO)
    client = TestClient(tiny_gru_app.app)
    payload = {**backend_payload, "contentIdSequence": [str(index) for index in range(12)]}
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    recommended_ids = [item["content_id"] for item in response.json()["recommendations"]]
    assert recommended_ids == ["central-1", "central-2", "central-3", "central-4"]
    assert runtime.call_count == 0
    record = next(item for item in caplog.records if item.event == "recommend_fallback")
    assert record.fallback_reason == "full_course_sequence"


def test_suggest_travel_spots_fallback_uses_area_specific_content_ids(monkeypatch, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)
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


def test_suggest_travel_spots_uses_shared_runtime_recommendations(monkeypatch, backend_payload) -> None:
    seoul_ids = AREA_FALLBACK_IDS["11000"]
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", DummySharedRecommendRuntime(recommendations=seoul_ids))
    client = TestClient(tiny_gru_app.app)
    payload = {**backend_payload, "contentIdSequence": seoul_ids[:2]}
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    recommended_ids = [item["content_id"] for item in response.json()["recommendations"]]
    assert len(recommended_ids) == 4
    assert set(recommended_ids).issubset(seoul_ids)
    assert set(recommended_ids).isdisjoint(payload["contentIdSequence"])


@pytest.mark.parametrize(
    ("age_group", "expected_age"),
    [
        ("10", 20),
        ("59", 50),
        ("70", 60),
    ],
)
def test_suggest_travel_spots_normalizes_age_group_before_runtime(
    monkeypatch, backend_payload, age_group, expected_age
) -> None:
    seoul_ids = AREA_FALLBACK_IDS["11000"]
    runtime = TrackingSharedRecommendRuntime(recommendations=seoul_ids)
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", runtime)
    client = TestClient(tiny_gru_app.app)
    payload = {**backend_payload, "ageGroup": age_group, "contentIdSequence": seoul_ids[:2]}
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    assert runtime.call_count == 1
    assert runtime.last_user_features["p0_age"] == expected_age


def test_barrierfree_detail_api_uses_content_id_params(monkeypatch) -> None:
    captured_url = ""

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return (
                b'{"response":{"header":{"resultCode":"0000"},'
                b'"body":{"items":{"item":[{"wheelchair":"ok"}]}}}}'
            )

    def fake_urlopen(url, timeout):
        nonlocal captured_url
        captured_url = url
        assert timeout == accessibility.BARRIERFREE_TIMEOUT_SECONDS
        return FakeResponse()

    monkeypatch.setattr(accessibility, "urlopen", fake_urlopen)

    items = accessibility.fetch_barrierfree_detail_items("12345", service_key="plain-key")

    assert items == [{"wheelchair": "ok"}]
    assert accessibility.BARRIERFREE_DETAIL_BASE_URL in captured_url
    assert "contentId=12345" in captured_url
    assert "MobileOS=ETC" in captured_url
    assert "MobileApp=kor_travel_recommendation" in captured_url
    assert "numOfRows=10" in captured_url


def test_accessibility_feature_detection_requires_requested_features() -> None:
    items = [
        {
            "parking": "장애인 주차장 있음",
            "exit": "출입구까지 경사로 설치",
            "babysparechair": "유모차 대여 가능",
        }
    ]

    features = accessibility.accessibility_features_from_items(items)

    assert {"disabled", "elderly", "child"}.issubset(features)


def test_accessibility_feature_detection_uses_barrierfree_field_names() -> None:
    items = [
        {
            "wheelchair": "대여 가능",
            "elevator": "승강기 이용 가능",
            "parking": "장애인 주차장 있음",
            "route": "출입구까지 경사로 설치",
            "stroller": "대여 가능",
            "helpdog": "없음",
        }
    ]

    features = accessibility.accessibility_features_from_items(items)

    assert "disabled" in features
    assert "elderly" in features
    assert "child" in features


def test_suggest_travel_spots_skips_accessibility_filter_when_flags_are_false(
    monkeypatch, backend_payload
) -> None:
    seoul_ids = AREA_FALLBACK_IDS["11000"]
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", DummySharedRecommendRuntime(recommendations=seoul_ids))

    def fail_if_called(_content_id):
        raise AssertionError("accessibility API should not be called")

    monkeypatch.setattr(accessibility, "fetch_barrierfree_detail_items", fail_if_called)
    client = TestClient(tiny_gru_app.app)
    payload = {**backend_payload, "contentIdSequence": seoul_ids[:2]}
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    recommended_ids = [item["content_id"] for item in response.json()["recommendations"]]
    assert recommended_ids == seoul_ids[2:6]


def test_suggest_travel_spots_filters_candidate_pool_by_disabled_accessibility(
    monkeypatch, backend_payload
) -> None:
    seoul_ids = AREA_FALLBACK_IDS["11000"]
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", DummySharedRecommendRuntime(recommendations=seoul_ids))
    accessible_ids = {seoul_ids[2], seoul_ids[4]}
    called_ids: list[str] = []

    def fake_fetch(content_id):
        called_ids.append(content_id)
        if content_id in accessible_ids:
            return [{"wheelchair": "휠체어 접근 가능, 장애인 화장실 있음"}]
        return [{"contentid": content_id}]

    monkeypatch.setattr(accessibility, "fetch_barrierfree_detail_items", fake_fetch)
    client = TestClient(tiny_gru_app.app)
    payload = {
        **backend_payload,
        "contentIdSequence": seoul_ids[:2],
        "hasDisabled": True,
    }
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    recommended_ids = [item["content_id"] for item in response.json()["recommendations"]]
    assert recommended_ids == [seoul_ids[2], seoul_ids[4]]
    assert called_ids == seoul_ids[2:] + seoul_ids[:2]


def test_suggest_travel_spots_returns_empty_when_no_accessibility_candidates(
    monkeypatch, backend_payload
) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)
    monkeypatch.setattr(accessibility, "fetch_barrierfree_detail_items", lambda _content_id: [])
    client = TestClient(tiny_gru_app.app)
    payload = {
        **backend_payload,
        "contentIdSequence": AREA_FALLBACK_IDS["11000"][:2],
        "hasDisabled": True,
    }
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    assert response.json()["recommendations"] == []


def test_suggest_travel_spots_falls_back_for_invalid_travel_duration(monkeypatch, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)
    client = TestClient(tiny_gru_app.app)
    payload = {**backend_payload, "travelDuration": "4", "contentIdSequence": AREA_FALLBACK_IDS["11000"][:2]}
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    recommended_ids = [item["content_id"] for item in response.json()["recommendations"]]
    assert recommended_ids == AREA_FALLBACK_IDS["11000"][2:6]


def test_suggest_travel_spots_validation_error_uses_central_tourism(
    monkeypatch, backend_payload
) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)
    monkeypatch.setattr(
        travel_service,
        "build_central_tourism_recommendation_payload",
        lambda area_code, top_k: CENTRAL_TOURISM_RECOMMENDATIONS[:top_k],
    )
    client = TestClient(tiny_gru_app.app)
    payload = {
        **backend_payload,
        "travelPersona": 8,
        "contentIdSequence": AREA_FALLBACK_IDS["11000"][:2],
    }
    payload.pop("contentIdList")

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    recommended_ids = [item["content_id"] for item in response.json()["recommendations"]]
    assert recommended_ids == ["central-1", "central-2", "central-3", "central-4"]


def test_suggest_travel_spots_logs_inference_failure_before_fallback(monkeypatch, caplog, backend_payload) -> None:
    seoul_ids = AREA_FALLBACK_IDS["11000"]
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", DummySharedRecommendRuntime(fail=True))
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
    assert record.runtime == "shared_next_poi_gru"
    assert record.fallback_reason == "inference_error"


def test_recommend_internal_does_not_convert_unexpected_errors_to_400(monkeypatch) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", DummySharedRecommendRuntime(fail=True))

    with pytest.raises(RuntimeError):
        tiny_gru_app.recommend_internal(
            tiny_gru_app.RecommendRequest(content_id_sequence=["1"], user_features={}, top_k=1)
        )
