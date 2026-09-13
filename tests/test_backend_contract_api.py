from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import tiny_gru_app


AREA_FALLBACK_IDS = tiny_gru_app.FALLBACK_CONTENT_IDS_BY_AREA


def backend_payload() -> dict:
    return {
        "areaCode": "11000",
        "travelDuration": "2",
        "travelPersona": 3,
        "ageGroup": "30",
        "gender": "남",
        "travelerStyle": "4",
        "preferredArea": ["50110", "26350"],
        "residenceArea": "11",
        "hasChild": False,
        "hasElderly": False,
        "hasDisabled": False,
        "companionCount": 1,
    }


def test_generate_travel_returns_fallback_when_course_runtime_missing(monkeypatch) -> None:
    monkeypatch.setattr(tiny_gru_app, "COURSE_RUNTIME", None)
    client = TestClient(tiny_gru_app.app)

    response = client.post("/generate-course", json=backend_payload())

    assert response.status_code == 200
    body = response.json()
    assert len(body["content_id_sequence"]) == 6
    assert len(body["steps"]) == 6
    assert body["steps"][0]["day_index"] == 1
    assert body["steps"][3]["day_index"] == 2
    assert set(body["content_id_sequence"]).issubset(AREA_FALLBACK_IDS["11000"])


def test_suggest_travel_spots_returns_four_fallback_items(monkeypatch) -> None:
    monkeypatch.setattr(tiny_gru_app, "RUNTIME", None)
    client = TestClient(tiny_gru_app.app)
    payload = {
        **backend_payload(),
        "contentIdSequence": AREA_FALLBACK_IDS["11000"][:2],
    }

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    recommendations = response.json()["recommendations"]
    assert len(recommendations) == 4
    assert {item["content_id"] for item in recommendations}.isdisjoint(payload["contentIdSequence"])
    assert {item["content_id"] for item in recommendations}.issubset(AREA_FALLBACK_IDS["11000"])


def test_generate_travel_fallback_uses_area_specific_content_ids(monkeypatch) -> None:
    monkeypatch.setattr(tiny_gru_app, "COURSE_RUNTIME", None)
    client = TestClient(tiny_gru_app.app)

    for area_code, area_content_ids in AREA_FALLBACK_IDS.items():
        response = client.post(
            "/generate-course",
            json={**backend_payload(), "areaCode": area_code},
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body["content_id_sequence"]) == 6
        assert set(body["content_id_sequence"]).issubset(area_content_ids)
        assert body["content_id_sequence"] == area_content_ids[:6]


def test_suggest_travel_spots_fallback_uses_area_specific_content_ids(monkeypatch) -> None:
    monkeypatch.setattr(tiny_gru_app, "RUNTIME", None)
    client = TestClient(tiny_gru_app.app)

    for area_code, area_content_ids in AREA_FALLBACK_IDS.items():
        response = client.post(
            "/recommend",
            json={
                **backend_payload(),
                "areaCode": area_code,
                "contentIdSequence": area_content_ids[:2],
            },
        )

        assert response.status_code == 200
        recommendations = response.json()["recommendations"]
        recommended_ids = [item["content_id"] for item in recommendations]
        assert len(recommended_ids) == 4
        assert set(recommended_ids).issubset(area_content_ids)
        assert set(recommended_ids).isdisjoint(area_content_ids[:2])
        assert recommended_ids == area_content_ids[2:6]


def test_preferred_area_requires_one_to_three_five_digit_strings() -> None:
    client = TestClient(tiny_gru_app.app)

    missing_response = client.post(
        "/generate-course",
        json={**backend_payload(), "preferredArea": []},
    )
    too_many_response = client.post(
        "/generate-course",
        json={**backend_payload(), "preferredArea": ["50110", "26350", "11110", "22220"]},
    )
    bad_code_response = client.post(
        "/generate-course",
        json={**backend_payload(), "preferredArea": ["5011A"]},
    )

    assert missing_response.status_code == 422
    assert too_many_response.status_code == 422
    assert bad_code_response.status_code == 422


def test_travel_persona_must_be_between_one_and_seven() -> None:
    client = TestClient(tiny_gru_app.app)

    response = client.post(
        "/generate-course",
        json={**backend_payload(), "travelPersona": 8},
    )

    assert response.status_code == 422


def test_area_code_must_be_supported() -> None:
    client = TestClient(tiny_gru_app.app)

    response = client.post(
        "/generate-course",
        json={**backend_payload(), "areaCode": "99999"},
    )

    assert response.status_code == 422


def test_health_reports_degraded_when_runtimes_are_missing(monkeypatch) -> None:
    monkeypatch.setattr(tiny_gru_app, "RUNTIME", None)
    monkeypatch.setattr(tiny_gru_app, "COURSE_RUNTIME", None)
    client = TestClient(tiny_gru_app.app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["tiny_gru_loaded"] is False
    assert body["course_decoder_loaded"] is False


def test_swagger_examples_use_backend_contract_defaults() -> None:
    client = TestClient(tiny_gru_app.app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    openapi = response.json()
    assert sorted(openapi["paths"]) == ["/generate-course", "/health", "/recommend"]
    schemas = openapi["components"]["schemas"]
    generate_example = schemas["TravelGenerateRequest"]["example"]
    suggestions_example = schemas["TravelSpotSuggestionsRequest"]["example"]
    assert generate_example["areaCode"] == "11000"
    assert generate_example["travelPersona"] == 3
    assert generate_example["preferredArea"] == ["50110", "26350"]
    assert suggestions_example["contentIdSequence"] == ["2815426", "2773265"]
    assert "top_k" not in suggestions_example
    assert "RecommendRequest" not in schemas
    assert "GenerateCourseRequest" not in schemas
