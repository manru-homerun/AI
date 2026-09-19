from __future__ import annotations

import json
import os
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from src.core.config import BACKEND_RECOMMENDATION_TOP_K, COURSE_POIS_PER_DAY, ROOT


CENTRAL_TOURISM_API_URL = "https://apis.data.go.kr/B551011/LocgoHubTarService1/areaBasedList1"
CENTRAL_TOURISM_MOBILE_APP = "kor_travel_recommendation"
CENTRAL_TOURISM_TIMEOUT_SECONDS = 3.0
CENTRAL_TOURISM_BASE_YM = "202504"

TOURAPI_AREA_PARAMS_BY_BACKEND_AREA = {
    "11000": {"areaCd": "11", "signguCd": "11710"},
    "41110": {"areaCd": "41", "signguCd": "41111"},
    "28000": {"areaCd": "28", "signguCd": "28177"},
    "30000": {"areaCd": "30", "signguCd": "30140"},
    "27000": {"areaCd": "27", "signguCd": "27260"},
    "12000": {"areaCd": "29", "signguCd": "29170"},
    "26000": {"areaCd": "26", "signguCd": "26260"},
    "48120": {"areaCd": "48", "signguCd": "48127"},
}


FALLBACK_CONTENT_IDS_BY_AREA = {
    "11000": [
        "2815426",
        "2773265",
        "3076141",
        "2758179",
        "3060919",
        "130289",
        "250469",
        "2930884",
        "129854",
        "1750737",
        "809490",
        "2930839",
    ],
    "41110": [
        "2868656",
        "2747132",
        "2892936",
        "3355253",
        "4076762",
        "2662855",
        "2944481",
        "2829013",
        "2753679",
        "2893042",
        "2613658",
        "1064469",
    ],
    "28000": [
        "2458348",
        "2994418",
        "1030642",
        "2612802",
        "2734015",
        "947611",
        "2767580",
        "1113230",
        "2834112",
        "3097744",
        "852304",
        "2837034",
    ],
    "30000": [
        "2580239",
        "1720749",
        "2900942",
        "1807489",
        "129785",
        "2662681",
        "2899345",
        "3454325",
        "2721465",
        "2752861",
        "2580604",
        "2738011",
    ],
    "27000": [
        "2864490",
        "130132",
        "1956986",
        "651687",
        "1871383",
        "2930650",
        "637751",
        "1611891",
        "2756646",
        "2785275",
        "126130",
        "2470055",
    ],
    "12000": [
        "2488192",
        "4065124",
        "1621360",
        "2755010",
        "2783851",
        "127539",
        "2779116",
        "3056157",
        "129761",
        "130065",
        "129782",
        "2033294",
    ],
    "26000": [
        "127488",
        "3345026",
        "2456767",
        "4011128",
        "4011143",
        "2931381",
        "2609623",
        "2785272",
        "2869241",
        "127925",
        "126078",
        "2991028",
    ],
    "48120": [
        "2864150",
        "2575790",
        "2627175",
        "1622590",
        "2844199",
        "2838789",
        "1622326",
        "2864877",
        "3456409",
        "3397899",
        "2863958",
        "3386455",
    ],
}


def fallback_content_ids_for_area(area_code: str) -> list[str]:
    return FALLBACK_CONTENT_IDS_BY_AREA[str(area_code).strip()]


def read_env_key(env_path: str, key_name: str) -> str | None:
    path = ROOT / env_path
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == key_name:
            return value.strip().strip('"').strip("'")
    return None


def data_openapi_key() -> str | None:
    return os.environ.get("DATA_OPENAPI_KEY") or read_env_key(".env", "DATA_OPENAPI_KEY")


def looks_url_encoded(value: str) -> bool:
    return bool(value) and "%2" in value.lower()


def build_openapi_url(base_url: str, params: dict[str, Any]) -> str:
    service_key = str(params["serviceKey"])
    other_params = {key: value for key, value in params.items() if key != "serviceKey" and value is not None}
    service_key_param = (
        f"serviceKey={service_key}"
        if looks_url_encoded(service_key)
        else urlencode({"serviceKey": service_key})
    )
    return f"{base_url}?{service_key_param}&{urlencode(other_params)}"


def parse_openapi_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    response = payload.get("response", {})
    body = response.get("body", {})
    items = body.get("items", {})
    raw_item = items.get("item", []) if isinstance(items, dict) else []
    if isinstance(raw_item, dict):
        raw_item = [raw_item]
    if not isinstance(raw_item, list):
        return []
    return [item for item in raw_item if isinstance(item, dict)]


def central_tourism_content_id(item: dict[str, Any]) -> str | None:
    for key in (
        "contentid",
        "contentId",
        "hubTatsCd",
        "rlteTatsId",
        "rlteTatsNo",
        "hubTatsId",
        "hubTatsNo",
        "baseYmd",
        "title",
        "rlteTatsNm",
        "hubTatsNm",
    ):
        value = item.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def fetch_central_tourism_items(
    area_code: str,
    top_k: int = BACKEND_RECOMMENDATION_TOP_K,
    service_key: str | None = None,
    base_url: str | None = None,
    timeout: float = CENTRAL_TOURISM_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    api_key = service_key or data_openapi_key()
    if not api_key:
        raise RuntimeError("DATA_OPENAPI_KEY is not configured")

    params: dict[str, Any] = {
        "serviceKey": api_key,
        "MobileOS": "ETC",
        "MobileApp": CENTRAL_TOURISM_MOBILE_APP,
        "_type": "json",
        "numOfRows": top_k,
        "pageNo": 1,
        "baseYm": os.environ.get("CENTRAL_TOURISM_BASE_YM", CENTRAL_TOURISM_BASE_YM),
        **TOURAPI_AREA_PARAMS_BY_BACKEND_AREA[str(area_code).strip()],
    }
    api_url = base_url or os.environ.get("CENTRAL_TOURISM_API_URL", CENTRAL_TOURISM_API_URL)
    url = build_openapi_url(api_url, params)

    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, socket.timeout, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"central tourism API request failed: {type(exc).__name__}") from exc

    header = payload.get("response", {}).get("header", {})
    result_code = str(header.get("resultCode", ""))
    if result_code and result_code != "0000":
        raise RuntimeError("central tourism API returned an error")
    return parse_openapi_items(payload)


def build_central_tourism_recommendation_payload(
    area_code: str,
    top_k: int = BACKEND_RECOMMENDATION_TOP_K,
) -> list[dict[str, Any]]:
    items = fetch_central_tourism_items(area_code, top_k)
    recommendations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        content_id = central_tourism_content_id(item)
        if content_id is None or content_id in seen:
            continue
        recommendations.append(
            {
                "content_id": content_id,
                "token_id": len(recommendations) + 3,
                "score": 1.0 - (len(recommendations) * 0.05),
            }
        )
        seen.add(content_id)
        if len(recommendations) == top_k:
            break
    if len(recommendations) < top_k:
        raise RuntimeError("central tourism API returned too few items")
    return recommendations


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_values.append(normalized)
    return unique_values


def validate_forced_content_ids_fit(forced_content_ids: list[str], desired_poi_count: int) -> list[str]:
    unique_content_ids = unique_preserve_order(forced_content_ids)
    if len(unique_content_ids) > desired_poi_count:
        raise ValueError("contentIdList cannot contain more unique items than the generated course length")
    return unique_content_ids


def build_area_limited_course_ids(
    area_code: str,
    desired_poi_count: int,
    forced_content_ids: list[str] | None = None,
) -> list[str]:
    content_ids = unique_preserve_order(forced_content_ids or [])
    area_content_ids = fallback_content_ids_for_area(area_code)
    seen = set(content_ids)
    while len(content_ids) < desired_poi_count:
        before_count = len(content_ids)
        for area_content_id in area_content_ids:
            if area_content_id in seen:
                continue
            content_ids.append(area_content_id)
            seen.add(area_content_id)
            if len(content_ids) == desired_poi_count:
                break
        if len(content_ids) == before_count:
            for area_content_id in area_content_ids:
                content_ids.append(area_content_id)
                if len(content_ids) == desired_poi_count:
                    break
    return content_ids


def build_fallback_course_payload(
    area_code: str,
    trip_days: int,
    forced_content_ids: list[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    desired_poi_count = max(int(trip_days), 1) * COURSE_POIS_PER_DAY
    forced_content_ids = validate_forced_content_ids_fit(forced_content_ids or [], desired_poi_count)
    content_ids = build_area_limited_course_ids(area_code, desired_poi_count, forced_content_ids)
    steps = [
        {
            "rank": index + 1,
            "day_index": index // COURSE_POIS_PER_DAY + 1,
            "slot_index": index % COURSE_POIS_PER_DAY + 1,
            "content_id": content_id,
            "token_id": index + 3,
            "score": 1.0 - (index * 0.01),
        }
        for index, content_id in enumerate(content_ids)
    ]
    return content_ids, steps


def build_fallback_recommendation_payload(
    area_code: str,
    content_id_sequence: list[str],
    top_k: int = BACKEND_RECOMMENDATION_TOP_K,
) -> list[dict[str, Any]]:
    area_content_ids = fallback_content_ids_for_area(area_code)
    seen = {str(content_id) for content_id in content_id_sequence}
    candidates = [content_id for content_id in area_content_ids if content_id not in seen]
    if len(candidates) < top_k:
        candidates.extend(content_id for content_id in area_content_ids if content_id in seen)
    return [
        {
            "content_id": content_id,
            "token_id": index + 3,
            "score": 1.0 - (index * 0.05),
        }
        for index, content_id in enumerate(candidates[:top_k])
    ]
