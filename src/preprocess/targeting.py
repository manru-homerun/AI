from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PRIVATE_PLACE_PATTERNS, TOURAPI_TARGET_CONFIGS
from .matching import normalize_place_name
from .sequences import clean_code


def load_private_place_patterns(pattern_path: Path | None) -> list[re.Pattern[str]]:
    raw_patterns = list(PRIVATE_PLACE_PATTERNS)
    if pattern_path is not None:
        if not pattern_path.exists():
            raise FileNotFoundError(f"Private-place pattern file not found: {pattern_path}")
        raw_patterns = [
            line.strip()
            for line in pattern_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in raw_patterns]


def is_private_place_name(value: Any, patterns: list[re.Pattern[str]]) -> bool:
    name = normalize_place_name(value)
    return bool(name) and any(pattern.search(name) for pattern in patterns)


def target_code_prefixes(target_config: dict[str, Any]) -> list[str]:
    return target_config.get("code_prefixes") or [target_config["code_prefix"]]


def target_address_keywords(target_config: dict[str, Any]) -> list[str]:
    return target_config.get("address_keywords") or [target_config["address_keyword"]]


def is_target_visit(row: pd.Series, target_config: dict[str, Any]) -> bool:
    code_prefixes = target_code_prefixes(target_config)
    legal_code = clean_code(row.get("LEGAL_DONG_CD"))
    sgg_code = clean_code(row.get("SGG_CD"))
    if pd.notna(legal_code) and str(legal_code).startswith(tuple(code_prefixes)):
        return True
    if pd.notna(sgg_code) and str(sgg_code).startswith(tuple(code_prefixes)):
        return True

    address = " ".join(
        str(value)
        for value in (row.get("ROAD_NM_ADDR"), row.get("LOTNO_ADDR"))
        if pd.notna(value)
    )
    return any(keyword in address for keyword in target_address_keywords(target_config))


def endpoint_mask(travel_seq: pd.DataFrame) -> pd.Series:
    visit_order = pd.to_numeric(travel_seq["visit_order"], errors="coerce")
    first_order = visit_order.groupby(travel_seq["travel_id"]).transform("min")
    last_order = visit_order.groupby(travel_seq["travel_id"]).transform("max")
    return visit_order.eq(first_order) | visit_order.eq(last_order)


def target_trip_ids_from_middle_visits(
    travel_seq: pd.DataFrame,
    target_config: dict[str, Any],
) -> set[str]:
    endpoints = endpoint_mask(travel_seq)
    middle = travel_seq.loc[~endpoints].copy()
    if middle.empty:
        return set()

    target_flags = middle.apply(lambda row: is_target_visit(row, target_config), axis=1)
    trip_flags = target_flags.groupby(middle["travel_id"]).all()
    return set(trip_flags[trip_flags].index.astype(str))


def mark_tourapi_scope(
    travel_seq: pd.DataFrame,
    tourapi_target: str,
    private_place_patterns: list[re.Pattern[str]],
) -> pd.Series:
    if tourapi_target == "all":
        status = pd.Series("not_queried", index=travel_seq.index, dtype="object")
        status.loc[travel_seq["CONTENT_ID"].notna()] = travel_seq.loc[
            travel_seq["CONTENT_ID"].notna(),
            "TOURAPI_MATCH_STATUS",
        ].fillna("matched")
        return status

    if tourapi_target not in TOURAPI_TARGET_CONFIGS:
        raise ValueError(f"Unsupported TourAPI target: {tourapi_target}")

    target_config = TOURAPI_TARGET_CONFIGS[tourapi_target]
    target_trip_ids = target_trip_ids_from_middle_visits(travel_seq, target_config)
    endpoints = endpoint_mask(travel_seq)
    in_target_trip = travel_seq["travel_id"].astype("string").isin(target_trip_ids)
    private_place = travel_seq["visit_area_nm"].apply(
        lambda value: is_private_place_name(value, private_place_patterns)
    )

    status = pd.Series(target_config["out_of_scope_status"], index=travel_seq.index, dtype="object")
    status.loc[in_target_trip & endpoints] = "excluded_endpoint"
    status.loc[in_target_trip & ~endpoints & private_place] = "excluded_private_place"
    status.loc[in_target_trip & ~endpoints & ~private_place] = "not_queried"

    matched = travel_seq["CONTENT_ID"].notna()
    status.loc[matched] = travel_seq.loc[matched, "TOURAPI_MATCH_STATUS"].fillna("matched")
    return status
