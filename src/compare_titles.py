from __future__ import annotations

import argparse
import glob
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

from preprocess.config import MATCH_THRESHOLD
from preprocess.io import read_env_keys
from preprocess.matching import normalize_for_match, string_similarity
from preprocess.tourapi import (
    build_tourapi_url,
    is_key_fallback_error,
    parse_tourapi_items,
)

DETAIL_COMMON_BASE_URL = "https://apis.data.go.kr/B551011/KorService2/detailCommon2"
MOBILE_APP = "kor_travel_recommendation"
AUDIT_COLUMNS = [
    "source_cache_file",
    "keyword_norm",
    "keyword",
    "source_kind",
    "fallback_key",
    "contentid",
    "candidate_contenttypeid",
    "detail_contenttypeid",
    "candidate_title",
    "detail_title",
    "keyword_detail_similarity",
    "keyword_candidate_similarity",
    "detail_contains_keyword",
    "keyword_contains_detail",
    "contenttypeid_matches",
    "audit_status",
    "candidate_addr1",
    "detail_addr1",
    "candidate_dist",
    "api_key_name",
    "api_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit TourAPI cached content IDs by fetching official detail titles."
    )
    parser.add_argument(
        "--cache-glob",
        default="data/processed/*_tourapi_keyword_cache.jsonl",
        help="Glob for TourAPI keyword cache JSONL files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/tourapi_content_title_audit.csv"),
    )
    parser.add_argument(
        "--detail-cache-path",
        type=Path,
        default=Path("data/processed/tourapi_detail_common_cache.jsonl"),
    )
    parser.add_argument("--env-path", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-name", default="DATA_OPENAPI_KEY")
    parser.add_argument("--api-key-names", nargs="+", default=None)
    parser.add_argument("--detail-base-url", default=DETAIL_COMMON_BASE_URL)
    parser.add_argument("--request-sleep", type=float, default=0.05)
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--parallel-workers", type=int, default=1)
    return parser.parse_args()


def load_latest_keyword_records(cache_path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            keyword_norm = record.get("keyword_norm")
            if keyword_norm:
                latest[str(keyword_norm)] = record
    return latest


def candidate_row(
    cache_path: Path,
    record: dict[str, Any],
    source_kind: str,
    fallback_key: str,
    item: dict[str, Any],
) -> dict[str, Any] | None:
    content_id = item.get("contentid")
    if not content_id:
        return None
    return {
        "source_cache_file": str(cache_path),
        "keyword_norm": record.get("keyword_norm"),
        "keyword": record.get("keyword"),
        "source_kind": source_kind,
        "fallback_key": fallback_key,
        "contentid": str(content_id),
        "candidate_contenttypeid": str(item.get("contenttypeid") or ""),
        "candidate_title": item.get("title"),
        "candidate_addr1": item.get("addr1"),
        "candidate_mapx": item.get("mapx"),
        "candidate_mapy": item.get("mapy"),
        "candidate_dist": item.get("dist"),
    }


def extract_candidate_rows(cache_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for cache_path in cache_paths:
        for record in load_latest_keyword_records(cache_path).values():
            for item in record.get("items") or []:
                row = candidate_row(cache_path, record, "keyword_item", "", item)
                if row is None:
                    continue
                dedupe_key = (
                    row["source_cache_file"],
                    str(row["keyword_norm"]),
                    row["source_kind"],
                    row["fallback_key"],
                    row["contentid"],
                )
                if dedupe_key not in seen:
                    seen.add(dedupe_key)
                    rows.append(row)

            for fallback_key, fallback_record in (record.get("location_fallbacks") or {}).items():
                for item in fallback_record.get("items") or []:
                    row = candidate_row(
                        cache_path,
                        record,
                        "location_fallback_item",
                        str(fallback_key),
                        item,
                    )
                    if row is None:
                        continue
                    dedupe_key = (
                        row["source_cache_file"],
                        str(row["keyword_norm"]),
                        row["source_kind"],
                        row["fallback_key"],
                        row["contentid"],
                    )
                    if dedupe_key not in seen:
                        seen.add(dedupe_key)
                        rows.append(row)
    return rows


def load_detail_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return cache

    with cache_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            content_id = record.get("contentid")
            if content_id:
                cache[str(content_id)] = record
    return cache


def append_detail_cache(cache_path: Path, record: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def fetch_detail_common(
    content_id: str,
    service_key: str,
    base_url: str,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    params = {
        "serviceKey": service_key,
        "MobileOS": "ETC",
        "MobileApp": MOBILE_APP,
        "_type": "json",
        "numOfRows": 10,
        "pageNo": 1,
        "contentId": content_id,
    }
    url = build_tourapi_url(base_url, params)
    payload: dict[str, Any] | None = None
    last_error = None
    for attempt in range(retries + 1):
        try:
            with urlopen(url, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except (HTTPError, URLError, TimeoutError, socket.timeout, OSError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))
                continue
            return {
                "contentid": content_id,
                "status": "api_error",
                "items": [],
                "error": last_error,
            }

    if payload is None:
        return {
            "contentid": content_id,
            "status": "api_error",
            "items": [],
            "error": last_error or "empty response",
        }

    header = payload.get("response", {}).get("header", {})
    result_code = str(header.get("resultCode", ""))
    if result_code and result_code != "0000":
        return {
            "contentid": content_id,
            "status": "api_error",
            "items": [],
            "error": header.get("resultMsg", f"resultCode={result_code}"),
        }

    return {
        "contentid": content_id,
        "status": "ok",
        "items": parse_tourapi_items(payload),
        "error": None,
    }


def fetch_detail_common_with_keys(
    content_id: str,
    service_keys: list[tuple[str, str]],
    start_key_index: int,
    base_url: str,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    last_record: dict[str, Any] | None = None
    key_count = len(service_keys)
    if key_count == 0:
        return {
            "contentid": content_id,
            "status": "api_error",
            "items": [],
            "error": "no API keys provided",
        }

    rotated_keys = service_keys[start_key_index % key_count :] + service_keys[: start_key_index % key_count]
    for key_index, (key_name, service_key) in enumerate(rotated_keys):
        record = fetch_detail_common(
            content_id=content_id,
            service_key=service_key,
            base_url=base_url,
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
        )
        record["api_key_name"] = key_name
        last_record = record
        if is_key_fallback_error(record) and key_index < key_count - 1:
            continue
        return record

    return last_record


def populate_detail_cache(
    content_ids: list[str],
    detail_cache_path: Path,
    detail_cache: dict[str, dict[str, Any]],
    service_keys: list[tuple[str, str]],
    base_url: str,
    timeout: float,
    retries: int,
    retry_sleep: float,
    request_sleep: float,
    parallel_workers: int,
) -> dict[str, dict[str, Any]]:
    pending = [content_id for content_id in content_ids if content_id not in detail_cache]
    if not pending:
        return detail_cache

    def fetch_record(content_id: str, key_index: int) -> dict[str, Any]:
        return fetch_detail_common_with_keys(
            content_id=content_id,
            service_keys=service_keys,
            start_key_index=key_index,
            base_url=base_url,
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
        )

    if parallel_workers == 1:
        for key_index, content_id in enumerate(pending):
            record = fetch_record(content_id, key_index)
            detail_cache[content_id] = record
            append_detail_cache(detail_cache_path, record)
            if request_sleep > 0:
                time.sleep(request_sleep)
        return detail_cache

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        future_to_content_id = {}
        for key_index, content_id in enumerate(pending):
            future = executor.submit(fetch_record, content_id, key_index)
            future_to_content_id[future] = content_id
            if request_sleep > 0:
                time.sleep(request_sleep)

        for future in as_completed(future_to_content_id):
            content_id = future_to_content_id[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "contentid": content_id,
                    "status": "api_error",
                    "items": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            detail_cache[content_id] = record
            append_detail_cache(detail_cache_path, record)
    return detail_cache


def first_detail_item(detail_record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not detail_record or detail_record.get("status") == "api_error":
        return None
    items = detail_record.get("items") or []
    if not items:
        return None
    return items[0]


def contains_flags(keyword: Any, detail_title: Any) -> tuple[bool, bool]:
    keyword_norm = normalize_for_match(keyword)
    detail_norm = normalize_for_match(detail_title)
    detail_contains_keyword = bool(keyword_norm and detail_norm and keyword_norm in detail_norm)
    keyword_contains_detail = bool(keyword_norm and detail_norm and detail_norm in keyword_norm)
    return detail_contains_keyword, keyword_contains_detail


def audit_status(
    keyword: Any,
    detail_title: Any,
    detail_record: dict[str, Any] | None,
) -> str:
    if detail_record is None or detail_record.get("status") == "api_error":
        return "detail_api_error"
    if first_detail_item(detail_record) is None:
        return "detail_no_result"

    keyword_norm = normalize_for_match(keyword)
    detail_norm = normalize_for_match(detail_title)
    if keyword_norm and detail_norm and keyword_norm == detail_norm:
        return "exact_match"
    detail_contains_keyword, keyword_contains_detail = contains_flags(keyword, detail_title)
    if detail_contains_keyword or keyword_contains_detail:
        return "contains_match"
    if string_similarity(keyword, detail_title) >= MATCH_THRESHOLD:
        return "similar_match"
    return "weak_match"


def build_audit_rows(
    candidate_rows: list[dict[str, Any]],
    detail_cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        content_id = str(candidate["contentid"])
        detail_record = detail_cache.get(content_id)
        detail_item = first_detail_item(detail_record)
        detail_title = detail_item.get("title") if detail_item else None
        detail_content_type_id = str(detail_item.get("contenttypeid") or "") if detail_item else ""
        detail_contains_keyword, keyword_contains_detail = contains_flags(
            candidate.get("keyword"),
            detail_title,
        )
        candidate_content_type_id = str(candidate.get("candidate_contenttypeid") or "")
        rows.append(
            {
                "source_cache_file": candidate.get("source_cache_file"),
                "keyword_norm": candidate.get("keyword_norm"),
                "keyword": candidate.get("keyword"),
                "source_kind": candidate.get("source_kind"),
                "fallback_key": candidate.get("fallback_key"),
                "contentid": content_id,
                "candidate_contenttypeid": candidate_content_type_id,
                "detail_contenttypeid": detail_content_type_id,
                "candidate_title": candidate.get("candidate_title"),
                "detail_title": detail_title,
                "keyword_detail_similarity": round(
                    string_similarity(candidate.get("keyword"), detail_title), 4
                ),
                "keyword_candidate_similarity": round(
                    string_similarity(candidate.get("keyword"), candidate.get("candidate_title")),
                    4,
                ),
                "detail_contains_keyword": detail_contains_keyword,
                "keyword_contains_detail": keyword_contains_detail,
                "contenttypeid_matches": bool(
                    candidate_content_type_id
                    and detail_content_type_id
                    and candidate_content_type_id == detail_content_type_id
                ),
                "audit_status": audit_status(
                    candidate.get("keyword"),
                    detail_title,
                    detail_record,
                ),
                "candidate_addr1": candidate.get("candidate_addr1"),
                "detail_addr1": detail_item.get("addr1") if detail_item else None,
                "candidate_dist": candidate.get("candidate_dist"),
                "api_key_name": detail_record.get("api_key_name") if detail_record else None,
                "api_error": detail_record.get("error") if detail_record else "missing detail cache",
            }
        )
    return rows


def print_summary(df: pd.DataFrame) -> None:
    print(f"audit rows: {len(df):,}")
    print("audit_status:")
    if df.empty:
        print("(empty)")
    else:
        print(df["audit_status"].value_counts(dropna=False).to_string())

    problem_statuses = {"weak_match", "detail_api_error", "detail_no_result"}
    problems = df[df["audit_status"].isin(problem_statuses)].copy()
    if problems.empty:
        return
    print("\nproblem samples:")
    columns = [
        "keyword",
        "candidate_title",
        "detail_title",
        "contentid",
        "keyword_detail_similarity",
        "audit_status",
    ]
    print(
        problems.sort_values(
            ["audit_status", "keyword_detail_similarity"],
            na_position="first",
        )
        .head(30)[columns]
        .to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    if args.parallel_workers < 1:
        raise ValueError("--parallel-workers must be at least 1")

    cache_paths = [Path(path) for path in sorted(glob.glob(args.cache_glob))]
    if not cache_paths:
        raise FileNotFoundError(f"No cache files matched: {args.cache_glob}")

    service_key_names = args.api_key_names or [args.api_key_name]
    service_keys = read_env_keys(args.env_path, service_key_names)
    if not service_keys:
        raise RuntimeError(f"None of these API keys were found in .env: {service_key_names}")

    candidate_rows = extract_candidate_rows(cache_paths)
    content_ids = sorted({str(row["contentid"]) for row in candidate_rows})
    print(
        f"cache files={len(cache_paths):,}, "
        f"candidate rows={len(candidate_rows):,}, "
        f"unique content IDs={len(content_ids):,}"
    )

    detail_cache = load_detail_cache(args.detail_cache_path)
    detail_cache = populate_detail_cache(
        content_ids=content_ids,
        detail_cache_path=args.detail_cache_path,
        detail_cache=detail_cache,
        service_keys=service_keys,
        base_url=args.detail_base_url,
        timeout=args.request_timeout,
        retries=args.request_retries,
        retry_sleep=args.retry_sleep,
        request_sleep=args.request_sleep,
        parallel_workers=args.parallel_workers,
    )

    audit_rows = build_audit_rows(candidate_rows, detail_cache)
    df = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    print(f"wrote: {args.output_path}")
    print_summary(df)


if __name__ == "__main__":
    main()
