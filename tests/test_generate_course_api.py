from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from conftest import AREA_FALLBACK_IDS
from src.api import tiny_gru_app
from src.core.config import COURSE_POIS_PER_DAY
from src.inference import runtime as runtime_state
from src.services import travel_service


class DummyCourseRuntime:
    def generate(
        self,
        user_features,
        trip_days,
        desired_poi_count=None,
        duplicate_masking=True,
        forced_content_ids=None,
        allowed_content_ids=None,
    ):
        desired = int(desired_poi_count or trip_days * COURSE_POIS_PER_DAY)
        forced = list(forced_content_ids or [])
        area_code = str(user_features.get("area_code", "11000"))
        candidate_ids = allowed_content_ids or AREA_FALLBACK_IDS[area_code]
        allowed = [content_id for content_id in candidate_ids if content_id not in forced]
        content_ids = [*forced, *allowed[: desired - len(forced)]]
        steps = [
            tiny_gru_app.GenerateCourseStep(
                rank=index + 1,
                day_index=index // COURSE_POIS_PER_DAY + 1,
                slot_index=index % COURSE_POIS_PER_DAY + 1,
                content_id=content_id,
                token_id=index + 3,
                score=1.0 - (index * 0.01),
            )
            for index, content_id in enumerate(content_ids)
        ]
        return content_ids, steps


class FailingCourseRuntime:
    def generate(self, *_args, **_kwargs):
        raise RuntimeError("course decoder exploded")


class TrackingCourseRuntime(DummyCourseRuntime):
    def __init__(self) -> None:
        self.call_count = 0
        self.last_user_features = None

    def generate(self, *args, **kwargs):
        self.call_count += 1
        self.last_user_features = kwargs.get("user_features", args[0] if args else None)
        return super().generate(*args, **kwargs)


def test_generate_travel_returns_fallback_with_content_id_list_prefix(monkeypatch, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)
    client = TestClient(tiny_gru_app.app)

    response = client.post("/generate-course", json=backend_payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["content_id_sequence"]) == 12
    assert body["content_id_sequence"][:2] == backend_payload["contentIdList"]
    assert body["steps"][0]["day_index"] == 1
    assert body["steps"][0]["slot_index"] == 1
    assert body["steps"][5]["day_index"] == 1
    assert body["steps"][5]["slot_index"] == 6
    assert body["steps"][6]["day_index"] == 2
    assert body["steps"][6]["slot_index"] == 1
    assert set(body["content_id_sequence"][2:]).issubset(AREA_FALLBACK_IDS["11000"])


def test_generate_travel_logs_model_unavailable_fallback(monkeypatch, caplog, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)
    caplog.set_level(logging.WARNING)
    client = TestClient(tiny_gru_app.app)

    response = client.post("/generate-course", json=backend_payload)

    assert response.status_code == 200
    assert "fallback_reason=model_unavailable" in caplog.text
    record = next(item for item in caplog.records if item.event == "course_fallback")
    assert record.endpoint == "/generate-course"
    assert record.area_code == backend_payload["areaCode"]
    assert record.trip_days == 2
    assert record.runtime == "shared_next_poi_gru"
    assert record.fallback_reason == "model_unavailable"


def test_generate_travel_logs_inference_error_fallback(monkeypatch, caplog, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", FailingCourseRuntime())
    caplog.set_level(logging.ERROR)
    client = TestClient(tiny_gru_app.app)

    response = client.post("/generate-course", json=backend_payload)

    assert response.status_code == 200
    assert "fallback_reason=inference_error" in caplog.text
    record = next(item for item in caplog.records if item.event == "course_inference_failure")
    assert record.endpoint == "/generate-course"
    assert record.area_code == backend_payload["areaCode"]
    assert record.trip_days == 2
    assert record.runtime == "shared_next_poi_gru"
    assert record.fallback_reason == "inference_error"


def test_generate_travel_fallback_uses_area_specific_content_ids(monkeypatch, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)
    client = TestClient(tiny_gru_app.app)

    for area_code, area_content_ids in AREA_FALLBACK_IDS.items():
        content_id_list = area_content_ids[:2]
        response = client.post(
            "/generate-course",
            json={**backend_payload, "areaCode": area_code, "contentIdList": content_id_list},
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body["content_id_sequence"]) == 12
        assert body["content_id_sequence"][:2] == content_id_list
        assert set(body["content_id_sequence"][2:]).issubset(area_content_ids)
        assert body["content_id_sequence"] == area_content_ids[:12]


def test_generate_travel_runtime_uses_area_specific_content_ids(monkeypatch, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", DummyCourseRuntime())
    client = TestClient(tiny_gru_app.app)

    response = client.post("/generate-course", json=backend_payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["content_id_sequence"]) == 12
    content_ids = response.json()["content_id_sequence"]
    assert content_ids[:2] == backend_payload["contentIdList"]
    assert set(content_ids[2:]).issubset(AREA_FALLBACK_IDS["11000"])


@pytest.mark.parametrize(
    ("age_group", "expected_age"),
    [
        ("-1", 20),
        ("0", 20),
        ("10", 20),
        ("19", 20),
        ("20", 20),
        ("20.0", 20),
        ("25", 20),
        ("29", 20),
        ("59", 50),
        ("60", 60),
        ("70", 60),
        ("90", 60),
    ],
)
def test_generate_travel_normalizes_age_group_before_runtime(
    monkeypatch, backend_payload, age_group, expected_age
) -> None:
    runtime = TrackingCourseRuntime()
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", runtime)
    client = TestClient(tiny_gru_app.app)

    response = client.post("/generate-course", json={**backend_payload, "ageGroup": age_group})

    assert response.status_code == 200
    assert runtime.call_count == 1
    assert runtime.last_user_features["p0_age"] == expected_age


def test_generate_travel_rejects_non_numeric_age_group(backend_payload) -> None:
    client = TestClient(tiny_gru_app.app)

    response = client.post("/generate-course", json={**backend_payload, "ageGroup": "abc"})

    assert response.status_code == 400


def test_content_id_list_validation(backend_payload) -> None:
    client = TestClient(tiny_gru_app.app)

    empty_response = client.post("/generate-course", json={**backend_payload, "contentIdList": []})
    bad_response = client.post("/generate-course", json={**backend_payload, "contentIdList": ["2815426", "abc"]})
    exactly_full_response = client.post(
        "/generate-course",
        json={**backend_payload, "contentIdList": [str(index) for index in range(12)]},
    )
    too_long_response = client.post(
        "/generate-course",
        json={**backend_payload, "contentIdList": [str(index) for index in range(13)]},
    )

    assert empty_response.status_code == 200
    assert empty_response.json()["content_id_sequence"] == AREA_FALLBACK_IDS["11000"][:12]
    assert bad_response.status_code == 422
    assert exactly_full_response.status_code == 200
    assert exactly_full_response.json()["content_id_sequence"] == [str(index) for index in range(12)]
    assert too_long_response.status_code == 200
    assert too_long_response.json()["content_id_sequence"] == [str(index) for index in range(13)]
    assert [step["content_id"] for step in too_long_response.json()["steps"]] == [str(index) for index in range(13)]


def test_generate_travel_short_circuits_one_day_over_requested_content_ids(
    monkeypatch, caplog, backend_payload
) -> None:
    runtime = TrackingCourseRuntime()
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", runtime)
    caplog.set_level(logging.INFO)
    client = TestClient(tiny_gru_app.app)
    content_ids = [str(index) for index in range(7)]

    response = client.post(
        "/generate-course",
        json={**backend_payload, "travelDuration": "1", "contentIdList": content_ids},
    )

    assert response.status_code == 200
    assert runtime.call_count == 0
    body = response.json()
    assert body["content_id_sequence"] == content_ids
    assert [step["content_id"] for step in body["steps"]] == content_ids
    assert [step["day_index"] for step in body["steps"]] == [1] * 7
    assert [step["slot_index"] for step in body["steps"]] == list(range(1, 8))
    assert {step["token_id"] for step in body["steps"]} == {-1}
    assert {step["score"] for step in body["steps"]} == {0.0}
    record = next(item for item in caplog.records if item.event == "course_short_circuit_over_requested_content_ids")
    assert record.input_count == 7
    assert record.target_count == 6


def test_generate_travel_uses_existing_logic_at_exact_target(monkeypatch, backend_payload) -> None:
    runtime = TrackingCourseRuntime()
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", runtime)
    client = TestClient(tiny_gru_app.app)
    content_ids = [str(index) for index in range(6)]

    response = client.post(
        "/generate-course",
        json={**backend_payload, "travelDuration": "1", "contentIdList": content_ids},
    )

    assert response.status_code == 200
    assert runtime.call_count == 1


def test_generate_travel_short_circuit_keeps_all_items_with_last_day_overflow(
    monkeypatch, backend_payload
) -> None:
    runtime = TrackingCourseRuntime()
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", runtime)
    client = TestClient(tiny_gru_app.app)
    content_ids = [str(index) for index in range(13)]

    response = client.post(
        "/generate-course",
        json={**backend_payload, "travelDuration": "2", "contentIdList": content_ids},
    )

    assert response.status_code == 200
    assert runtime.call_count == 0
    body = response.json()
    assert body["content_id_sequence"] == content_ids
    assert [step["content_id"] for step in body["steps"]] == content_ids
    assert [step["day_index"] for step in body["steps"]] == [1] * 6 + [2] * 7
    assert [step["slot_index"] for step in body["steps"]] == [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7]


def test_generate_travel_short_circuit_skips_fallback_when_runtime_unavailable(
    monkeypatch, backend_payload
) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)

    def fail_fallback(*_args, **_kwargs):
        raise AssertionError("fallback should not run for over-requested contentIdList")

    monkeypatch.setattr(travel_service, "fallback_course_response_or_500", fail_fallback)
    client = TestClient(tiny_gru_app.app)

    response = client.post(
        "/generate-course",
        json={**backend_payload, "travelDuration": "1", "contentIdList": [str(index) for index in range(7)]},
    )

    assert response.status_code == 200


def test_generate_travel_short_circuit_skips_fallback_with_accessibility_flags(
    monkeypatch, backend_payload
) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)

    def fail_fallback(*_args, **_kwargs):
        raise AssertionError("fallback should not run for over-requested contentIdList")

    monkeypatch.setattr(travel_service, "fallback_course_response_or_500", fail_fallback)
    client = TestClient(tiny_gru_app.app)

    response = client.post(
        "/generate-course",
        json={
            **backend_payload,
            "travelDuration": "1",
            "contentIdList": [str(index) for index in range(7)],
            "hasDisabled": True,
            "hasChild": True,
            "hasElderly": True,
        },
    )

    assert response.status_code == 200


def test_companion_count_zero_is_accepted(monkeypatch, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)
    client = TestClient(tiny_gru_app.app)

    response = client.post("/generate-course", json={**backend_payload, "companionCount": 0})

    assert response.status_code == 200


def test_preferred_area_truncates_to_three_five_digit_strings(caplog, backend_payload) -> None:
    client = TestClient(tiny_gru_app.app)
    caplog.set_level(logging.WARNING)

    missing_response = client.post("/generate-course", json={**backend_payload, "preferredArea": []})
    overlong_response = client.post(
        "/generate-course",
        json={
            **backend_payload,
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
        },
    )
    bad_code_response = client.post("/generate-course", json={**backend_payload, "preferredArea": ["5011A"]})
    bad_late_code_response = client.post(
        "/generate-course",
        json={**backend_payload, "preferredArea": ["50110", "26350", "11110", "bad"]},
    )

    assert missing_response.status_code == 422
    assert overlong_response.status_code == 200
    assert bad_code_response.status_code == 422
    assert bad_late_code_response.status_code == 422
    record = next(item for item in caplog.records if item.event == "preferred_area_truncated")
    assert record.original_count == 16
    assert record.normalized_count == 3


def test_gender_must_use_backend_korean_values(monkeypatch, backend_payload) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)
    client = TestClient(tiny_gru_app.app)

    male_response = client.post("/generate-course", json={**backend_payload, "gender": "남"})
    female_response = client.post("/generate-course", json={**backend_payload, "gender": "여"})
    legacy_response = client.post("/generate-course", json={**backend_payload, "gender": "M"})

    assert male_response.status_code == 200
    assert female_response.status_code == 200
    assert legacy_response.status_code == 422


def test_travel_persona_must_be_between_one_and_seven(backend_payload) -> None:
    client = TestClient(tiny_gru_app.app)

    response = client.post("/generate-course", json={**backend_payload, "travelPersona": 8})

    assert response.status_code == 422


def test_travel_duration_must_be_one_to_three(backend_payload) -> None:
    client = TestClient(tiny_gru_app.app)

    response = client.post("/generate-course", json={**backend_payload, "travelDuration": "4"})

    assert response.status_code == 400


def test_area_code_must_be_supported(backend_payload) -> None:
    client = TestClient(tiny_gru_app.app)

    response = client.post("/generate-course", json={**backend_payload, "areaCode": "99999"})

    assert response.status_code == 422


def test_health_and_ready_report_degraded_with_503_when_runtimes_are_missing(monkeypatch) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)
    monkeypatch.setattr(runtime_state, "RUNTIME", None)
    monkeypatch.setattr(runtime_state, "COURSE_RUNTIME", None)
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME_ERROR", "sensitive /tmp/shared failure")
    monkeypatch.setattr(runtime_state, "RUNTIME_ERROR", "sensitive /tmp/model failure")
    monkeypatch.setattr(runtime_state, "COURSE_RUNTIME_ERROR", "sensitive /tmp/course failure")
    client = TestClient(tiny_gru_app.app)

    for path in ("/health", "/ready"):
        response = client.get(path)

        assert response.status_code == 503
        body = response.json()
        assert body == {
            "status": "degraded",
            "shared_gru_loaded": False,
            "tiny_gru_loaded": False,
            "course_decoder_loaded": False,
        }
        assert "sensitive" not in response.text
        assert "/tmp" not in response.text


def test_health_and_ready_report_ok_when_runtimes_are_loaded(monkeypatch) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", DummyCourseRuntime())
    monkeypatch.setattr(runtime_state, "RUNTIME", DummyCourseRuntime())
    monkeypatch.setattr(runtime_state, "COURSE_RUNTIME", DummyCourseRuntime())
    client = TestClient(tiny_gru_app.app)

    for path in ("/health", "/ready"):
        response = client.get(path)

        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_live_reports_ok_when_runtimes_are_missing(monkeypatch) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", None)
    monkeypatch.setattr(runtime_state, "RUNTIME", None)
    monkeypatch.setattr(runtime_state, "COURSE_RUNTIME", None)
    client = TestClient(tiny_gru_app.app)

    response = client.get("/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_course_internal_does_not_convert_unexpected_errors_to_400(monkeypatch) -> None:
    monkeypatch.setattr(runtime_state, "SHARED_RUNTIME", FailingCourseRuntime())

    with pytest.raises(RuntimeError):
        tiny_gru_app.generate_course_internal(
            tiny_gru_app.GenerateCourseRequest(user_features={}, trip_days=1, desired_poi_count=3)
        )


def test_swagger_examples_use_backend_contract_defaults() -> None:
    client = TestClient(tiny_gru_app.app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    openapi = response.json()
    assert sorted(openapi["paths"]) == ["/generate-course", "/health", "/live", "/ready", "/recommend"]
    schemas = openapi["components"]["schemas"]
    generate_example = schemas["TravelGenerateRequest"]["example"]
    suggestions_example = schemas["TravelSpotSuggestionsRequest"]["example"]
    assert generate_example["areaCode"] == "11000"
    assert generate_example["contentIdList"] == ["2815426", "2773265"]
    assert generate_example["travelPersona"] == 3
    assert generate_example["preferredArea"] == ["50110", "26350"]
    assert suggestions_example["contentIdSequence"] == ["2815426", "2773265"]
    assert "contentIdList" not in suggestions_example
    assert "top_k" not in suggestions_example
    assert "RecommendRequest" not in schemas
    assert "GenerateCourseRequest" not in schemas
