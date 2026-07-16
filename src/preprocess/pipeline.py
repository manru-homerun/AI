from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import TOURAPI_TARGET_CONFIGS, TRAVEL_SEQ_COLUMNS
from .input_features import build_input
from .io import load_all_tables, read_env_keys, save_outputs
from .matching import best_tourapi_match
from .sequences import build_travel_seq
from .targeting import load_private_place_patterns, mark_tourapi_scope
from .tourapi import populate_tourapi_cache, populate_tourapi_location_fallbacks
from .validation import validate_outputs


@dataclass(frozen=True)
class PipelineOptions:
    raw_root: Path
    output_root: Path
    env_path: Path
    api_key_name: str
    api_key_names: list[str] | None
    cache_path: Path | None
    tourapi_base_url: str
    tourapi_location_base_url: str
    call_tourapi: bool
    tourapi_target: str
    private_place_pattern_path: Path | None
    refresh_tourapi_cache: bool
    max_api_calls: int | None
    allow_partial_api: bool
    retry_api_errors: bool
    retry_empty_results: bool
    skip_tourapi: bool
    request_sleep: float
    request_timeout: float
    request_retries: int
    retry_sleep: float
    parallel_workers: int
    num_rows: int
    location_radius: int


def restore_existing_matches(travel_seq: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    if not existing_path.exists():
        return travel_seq

    required_columns = [
        "travel_id",
        "visit_area_id",
        "CONTENT_ID",
        "CONTENT_TYPE_ID",
        "TOURAPI_MATCH_STATUS",
        "TOURAPI_MATCH_SCORE",
    ]
    existing = pd.read_csv(existing_path, encoding="utf-8-sig", dtype="string")
    if any(column not in existing.columns for column in required_columns):
        return travel_seq

    existing = existing[required_columns].copy()
    existing = existing[existing["CONTENT_ID"].notna()].copy()
    if existing.empty:
        return travel_seq

    existing = existing.drop_duplicates(["travel_id", "visit_area_id"], keep="last")
    restored = travel_seq.merge(
        existing,
        on=["travel_id", "visit_area_id"],
        how="left",
        suffixes=("", "_existing"),
        validate="many_to_one",
    )

    for column in [
        "CONTENT_ID",
        "CONTENT_TYPE_ID",
        "TOURAPI_MATCH_STATUS",
        "TOURAPI_MATCH_SCORE",
    ]:
        existing_column = f"{column}_existing"
        has_existing = restored[existing_column].notna()
        restored.loc[has_existing, column] = restored.loc[has_existing, existing_column]
        restored = restored.drop(columns=[existing_column])

    matched_without_status = (
        restored["CONTENT_ID"].notna() & restored["TOURAPI_MATCH_STATUS"].isna()
    )
    restored.loc[matched_without_status, "TOURAPI_MATCH_STATUS"] = "matched"
    return restored[TRAVEL_SEQ_COLUMNS]


def default_cache_path(output_root: Path, tourapi_target: str) -> Path:
    target_config = TOURAPI_TARGET_CONFIGS.get(tourapi_target)
    if target_config is None:
        return output_root / "tourapi_keyword_cache.jsonl"
    return output_root / target_config["cache_filename"]


def attach_tourapi_matches(
    travel_seq: pd.DataFrame,
    cache_path: Path,
    service_keys: list[tuple[str, str]],
    service_key_names: list[str],
    base_url: str,
    location_base_url: str,
    tourapi_target: str,
    private_place_pattern_path: Path | None,
    max_api_calls: int | None,
    allow_partial_api: bool,
    refresh_cache: bool,
    request_sleep: float,
    timeout: float,
    num_rows: int,
    retries: int,
    retry_sleep: float,
    parallel_workers: int,
    retry_api_errors: bool,
    retry_empty_results: bool,
    location_radius: int,
    skip_tourapi: bool,
) -> pd.DataFrame:
    travel_seq = travel_seq.copy()
    private_place_patterns = load_private_place_patterns(private_place_pattern_path)
    travel_seq["TOURAPI_MATCH_STATUS"] = mark_tourapi_scope(
        travel_seq=travel_seq,
        tourapi_target=tourapi_target,
        private_place_patterns=private_place_patterns,
    )

    if skip_tourapi:
        return travel_seq[TRAVEL_SEQ_COLUMNS]

    if not service_keys:
        raise RuntimeError(f"None of these API keys were found in .env: {service_key_names}")

    query_mask = travel_seq["TOURAPI_MATCH_STATUS"].eq("not_queried")
    query_mask &= travel_seq["X_COORD"].notna() & travel_seq["Y_COORD"].notna()

    keyword_cache = populate_tourapi_cache(
        keywords=travel_seq.loc[query_mask, "visit_area_nm"],
        cache_path=cache_path,
        service_keys=service_keys,
        base_url=base_url,
        max_api_calls=max_api_calls,
        allow_partial_api=allow_partial_api,
        refresh_cache=refresh_cache,
        request_sleep=request_sleep,
        timeout=timeout,
        num_rows=num_rows,
        retries=retries,
        retry_sleep=retry_sleep,
        parallel_workers=parallel_workers,
        retry_api_errors=retry_api_errors,
    )
    if retry_empty_results:
        keyword_cache = populate_tourapi_location_fallbacks(
            travel_seq=travel_seq.loc[query_mask],
            cache_path=cache_path,
            cache=keyword_cache,
            service_keys=service_keys,
            base_url=location_base_url,
            max_api_calls=max_api_calls,
            allow_partial_api=allow_partial_api,
            request_sleep=request_sleep,
            timeout=timeout,
            num_rows=num_rows,
            retries=retries,
            retry_sleep=retry_sleep,
            parallel_workers=parallel_workers,
            radius=location_radius,
        )

    matches = travel_seq.loc[travel_seq["TOURAPI_MATCH_STATUS"].eq("not_queried")].apply(
        lambda row: best_tourapi_match(row, keyword_cache, location_radius),
        axis=1,
    )
    match_df = pd.DataFrame(
        matches.tolist(),
        columns=[
            "CONTENT_ID",
            "CONTENT_TYPE_ID",
            "TOURAPI_MATCH_STATUS",
            "TOURAPI_MATCH_SCORE",
        ],
        index=matches.index,
    )

    for column in match_df.columns:
        travel_seq.loc[match_df.index, column] = match_df[column]
    return travel_seq[TRAVEL_SEQ_COLUMNS]


def _trip_days(travel: pd.DataFrame) -> pd.Series:
    start = pd.to_datetime(travel["TRAVEL_START_YMD"], errors="coerce")
    end = pd.to_datetime(travel["TRAVEL_END_YMD"], errors="coerce")
    return (end - start).dt.days + 1


def process_total(options: PipelineOptions | Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = load_all_tables(options.raw_root)
    travel = tables["travel"].copy()
    traveller = tables["traveller"].copy()
    visit = tables["visit"].copy()

    travel["trip_days"] = _trip_days(travel)
    travel = travel[travel["trip_days"].between(1, 3)].copy()
    travel["trip_days"] = travel["trip_days"].astype("int64")

    input_df = build_input(travel, traveller)
    travel_seq = build_travel_seq(visit, travel)
    travel_seq = restore_existing_matches(
        travel_seq,
        options.output_root / "total_travel_seq.csv",
    )

    seq_trip_ids = set(travel_seq["travel_id"].dropna())
    input_df = input_df[input_df["trip_id"].isin(seq_trip_ids)].copy()

    service_key_names = options.api_key_names or [options.api_key_name]
    service_keys = read_env_keys(options.env_path, service_key_names)
    cache_path = options.cache_path or default_cache_path(
        options.output_root,
        options.tourapi_target,
    )
    travel_seq = attach_tourapi_matches(
        travel_seq=travel_seq,
        cache_path=cache_path,
        service_keys=service_keys,
        service_key_names=service_key_names,
        base_url=options.tourapi_base_url,
        location_base_url=options.tourapi_location_base_url,
        tourapi_target=options.tourapi_target,
        private_place_pattern_path=options.private_place_pattern_path,
        max_api_calls=options.max_api_calls,
        allow_partial_api=options.allow_partial_api,
        refresh_cache=options.refresh_tourapi_cache,
        request_sleep=options.request_sleep,
        timeout=options.request_timeout,
        num_rows=options.num_rows,
        retries=options.request_retries,
        retry_sleep=options.retry_sleep,
        parallel_workers=options.parallel_workers,
        retry_api_errors=options.retry_api_errors,
        retry_empty_results=options.retry_empty_results,
        location_radius=options.location_radius,
        skip_tourapi=options.skip_tourapi or not options.call_tourapi,
    )

    validate_outputs(input_df, travel_seq)
    save_outputs(input_df, travel_seq, options.output_root)
    return input_df, travel_seq
