from __future__ import annotations

import json
import socket
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from src.fallback.travel import build_openapi_url, data_openapi_key, parse_openapi_items
from src.schemas.travel import RecommendItem


BARRIERFREE_DETAIL_BASE_URL = "https://apis.data.go.kr/B551011/KorWithService2/detailWithTour2"
BARRIERFREE_MOBILE_APP = "kor_travel_recommendation"
BARRIERFREE_TIMEOUT_SECONDS = 5.0

DISABLED_FIELDS = {
    "parking",
    "publictransport",
    "route",
    "ticketoffice",
    "promotion",
    "wheelchair",
    "exit",
    "elevator",
    "restroom",
    "guidesystem",
    "blindhandicapetc",
    "signguide",
    "videoguide",
    "hearingroom",
    "hearinghandicapetc",
    "handicapetc",
    "braileblock",
    "helpdog",
    "guidehuman",
    "audioguide",
    "bigprint",
    "brailepromotion",
}
ELDERLY_FIELDS = {
    "publictransport",
    "route",
    "wheelchair",
    "exit",
    "elevator",
    "restroom",
}
CHILD_FIELDS = {
    "stroller",
    "lactationroom",
    "babysparechair",
    "infantsfamilyetc",
}


def requested_accessibility_features(*, has_disabled: bool, has_elderly: bool, has_child: bool) -> set[str]:
    features: set[str] = set()
    if has_disabled:
        features.add("disabled")
    if has_elderly:
        features.add("elderly")
    if has_child:
        features.add("child")
    return features


def fetch_barrierfree_detail_items(
    content_id: str,
    service_key: str | None = None,
    base_url: str = BARRIERFREE_DETAIL_BASE_URL,
    timeout: float = BARRIERFREE_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    api_key = service_key or data_openapi_key()
    if not api_key:
        raise RuntimeError("DATA_OPENAPI_KEY is not configured")

    params: dict[str, Any] = {
        "serviceKey": api_key,
        "MobileOS": "ETC",
        "MobileApp": BARRIERFREE_MOBILE_APP,
        "_type": "json",
        "numOfRows": 10,
        "pageNo": 1,
        "contentId": str(content_id),
    }
    url = build_openapi_url(base_url, params)
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, socket.timeout, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"barrier-free API request failed: {type(exc).__name__}") from exc

    header = payload.get("response", {}).get("header", {})
    result_code = str(header.get("resultCode", ""))
    if result_code and result_code != "0000":
        raise RuntimeError("barrier-free API returned an error")
    return parse_openapi_items(payload)


def _has_accessibility_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_accessibility_value(nested_value) for nested_value in value.values())
    if isinstance(value, list):
        return any(_has_accessibility_value(nested_value) for nested_value in value)
    return bool(str(value).strip())


def _features_from_field_names(items: list[dict[str, Any]]) -> set[str]:
    features: set[str] = set()
    for item in items:
        for field_name, value in item.items():
            normalized_field_name = str(field_name).strip().lower()
            if normalized_field_name in DISABLED_FIELDS and _has_accessibility_value(value):
                features.add("disabled")
            if normalized_field_name in ELDERLY_FIELDS and _has_accessibility_value(value):
                features.add("elderly")
            if normalized_field_name in CHILD_FIELDS and _has_accessibility_value(value):
                features.add("child")
    return features


def accessibility_features_from_items(items: list[dict[str, Any]]) -> set[str]:
    return _features_from_field_names(items)


def filter_recommendations_by_accessibility(
    recommendations: list[RecommendItem],
    required_features: set[str],
    *,
    log_extra: Mapping[str, Any] | None = None,
    logger: Any = None,
) -> list[RecommendItem]:
    if not required_features:
        return recommendations

    filtered: list[RecommendItem] = []
    for item in recommendations:
        try:
            features = accessibility_features_from_items(fetch_barrierfree_detail_items(item.content_id))
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "barrier-free filtering dropped candidate after API failure",
                    extra={
                        **(dict(log_extra or {})),
                        "event": "recommend_accessibility_filter_failure",
                        "content_id": item.content_id,
                        "fallback_reason": "accessibility_api_error",
                        "error_type": type(exc).__name__,
                    },
                )
            continue
        if required_features.issubset(features):
            filtered.append(item)
    return filtered
