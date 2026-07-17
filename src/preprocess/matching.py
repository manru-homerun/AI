from __future__ import annotations

import difflib
import math
import re
from typing import Any

import pandas as pd

from .config import MATCH_THRESHOLD
from .sequences import clean_code


def normalize_keyword(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[\[\]\(\){}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_match(value: Any) -> str:
    text = normalize_keyword(value)
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", text)


def normalize_place_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def address_tokens(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        for token in re.split(r"\s+", str(value)):
            cleaned = re.sub(r"[^0-9a-zA-Z가-힣]", "", token)
            if len(cleaned) >= 2:
                tokens.add(cleaned)
    return tokens


def string_similarity(left: Any, right: Any) -> float:
    left_norm = normalize_for_match(left)
    right_norm = normalize_for_match(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        shorter = min(len(left_norm), len(right_norm))
        longer = max(len(left_norm), len(right_norm))
        return max(0.85, shorter / longer)
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio()


def safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def haversine_km(left_lon: float, left_lat: float, right_lon: float, right_lat: float) -> float:
    radius_km = 6371.0088
    lon1, lat1, lon2, lat2 = map(math.radians, [left_lon, left_lat, right_lon, right_lat])
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def distance_score(distance_km: float | None) -> float:
    if distance_km is None:
        return 0.0
    if distance_km <= 0.2:
        return 1.0
    if distance_km >= 5.0:
        return 0.0
    return 1.0 - ((distance_km - 0.2) / 4.8)


def candidate_legal_codes(candidate: dict[str, Any]) -> set[str]:
    codes: set[str] = set()
    for key, value in candidate.items():
        key_lower = key.lower()
        if "ldong" not in key_lower and "legal" not in key_lower and "bcode" not in key_lower:
            continue
        code = clean_code(value)
        if pd.notna(code):
            codes.add(str(code))
    return codes


def score_candidate(row: pd.Series, candidate: dict[str, Any]) -> float:
    title = candidate.get("title", "")
    name_score = string_similarity(row["visit_area_nm"], title)

    source_tokens = address_tokens(row.get("ROAD_NM_ADDR"), row.get("LOTNO_ADDR"))
    candidate_tokens = address_tokens(candidate.get("addr1"), candidate.get("addr2"))
    if source_tokens and candidate_tokens:
        address_score = len(source_tokens.intersection(candidate_tokens)) / len(
            source_tokens.union(candidate_tokens)
        )
    else:
        address_score = 0.0

    row_lon = safe_float(row.get("X_COORD"))
    row_lat = safe_float(row.get("Y_COORD"))
    candidate_lon = safe_float(candidate.get("mapx"))
    candidate_lat = safe_float(candidate.get("mapy"))
    distance_km = None
    if None not in (row_lon, row_lat, candidate_lon, candidate_lat):
        distance_km = haversine_km(row_lon, row_lat, candidate_lon, candidate_lat)

    legal_code = clean_code(row.get("LEGAL_DONG_CD"))
    sgg_code = clean_code(row.get("SGG_CD"))
    legal_candidates = candidate_legal_codes(candidate)
    code_score = 0.0
    for candidate_code in legal_candidates:
        if pd.notna(legal_code) and str(legal_code).startswith(candidate_code):
            code_score = 1.0
        if pd.notna(sgg_code) and str(sgg_code).startswith(candidate_code):
            code_score = 1.0

    return round(
        (0.45 * name_score)
        + (0.20 * address_score)
        + (0.30 * distance_score(distance_km))
        + (0.05 * code_score),
        4,
    )


def location_fallback_key(row: pd.Series, radius: Any) -> str | None:
    row_lon = safe_float(row.get("X_COORD"))
    row_lat = safe_float(row.get("Y_COORD"))
    if row_lon is None or row_lat is None:
        return None
    try:
        radius_int = int(radius)
    except (TypeError, ValueError):
        return None
    return f"{row_lon:.6f},{row_lat:.6f},r{radius_int}"


def title_contains_place_name(row: pd.Series, candidate: dict[str, Any]) -> bool:
    source = normalize_for_match(row.get("visit_area_nm"))
    title = normalize_for_match(candidate.get("title"))
    return bool(source and title and (source in title or title in source))


def candidate_distance_m(candidate: dict[str, Any]) -> float:
    distance = safe_float(candidate.get("dist"))
    if distance is None:
        return float("inf")
    return distance


def best_location_fallback_match(
    row: pd.Series,
    cache_record: dict[str, Any],
    location_radius: int,
) -> tuple[Any, Any, str, Any]:
    fallbacks = cache_record.get("location_fallbacks") or {}
    fallback_key = location_fallback_key(row, location_radius)
    fallback_record = fallbacks.get(fallback_key) if fallback_key else None

    if fallback_record is None:
        return pd.NA, pd.NA, "no_result", pd.NA
    if fallback_record.get("status") == "api_error":
        return pd.NA, pd.NA, "location_api_error", pd.NA

    candidates = fallback_record.get("items") or []
    if not candidates:
        return pd.NA, pd.NA, "location_no_result", pd.NA

    scored = [
        (
            title_contains_place_name(row, candidate),
            score_candidate(row, candidate),
            -candidate_distance_m(candidate),
            candidate,
        )
        for candidate in candidates
    ]
    _, score, _, candidate = max(scored, key=lambda item: item[:3])
    if score < MATCH_THRESHOLD:
        return pd.NA, pd.NA, "location_low_score", score

    content_id = candidate.get("contentid")
    content_type_id = candidate.get("contenttypeid")
    if not content_id or not content_type_id:
        return pd.NA, pd.NA, "location_api_error", score
    return str(content_id), str(content_type_id), "matched_location", score


def best_tourapi_match(
    row: pd.Series,
    keyword_cache: dict[str, dict[str, Any]],
    location_radius: int,
) -> tuple[Any, Any, str, Any]:
    if pd.isna(row.get("X_COORD")) or pd.isna(row.get("Y_COORD")):
        return pd.NA, pd.NA, "missing_coord", pd.NA

    keyword_norm = normalize_keyword(row.get("visit_area_nm"))
    if not keyword_norm:
        return pd.NA, pd.NA, "no_keyword", pd.NA

    cache_record = keyword_cache.get(keyword_norm)
    if cache_record is None:
        return pd.NA, pd.NA, "not_queried", pd.NA
    if cache_record.get("status") == "api_error":
        return pd.NA, pd.NA, "api_error", pd.NA

    candidates = cache_record.get("items") or []
    if not candidates:
        return best_location_fallback_match(row, cache_record, location_radius)

    scored = [(score_candidate(row, candidate), candidate) for candidate in candidates]
    score, candidate = max(scored, key=lambda item: item[0])
    if score < MATCH_THRESHOLD:
        return pd.NA, pd.NA, "low_score", score

    content_id = candidate.get("contentid")
    content_type_id = candidate.get("contenttypeid")
    if not content_id or not content_type_id:
        return pd.NA, pd.NA, "api_error", score
    return str(content_id), str(content_type_id), "matched", score
