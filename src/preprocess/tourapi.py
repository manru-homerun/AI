from __future__ import annotations

import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from .config import MAX_PARALLEL_WORKERS
from .matching import normalize_keyword

KEY_FALLBACK_ERROR_MARKERS = (
    "401",
    "unauthorized",
    "429",
    "too many requests",
    "limited number of service requests",
    "limited_number_of_service_requests",
    "quota",
)


def looks_url_encoded(value: str) -> bool:
    return bool(value) and "%2" in value.lower()


def build_tourapi_url(base_url: str, params: dict[str, Any]) -> str:
    service_key = str(params["serviceKey"])
    other_params = {key: value for key, value in params.items() if key != "serviceKey"}
    service_key_param = (
        f"serviceKey={service_key}"
        if looks_url_encoded(service_key)
        else urlencode({"serviceKey": service_key})
    )
    return f"{base_url}?{service_key_param}&{urlencode(other_params)}"


def parse_tourapi_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    response = payload.get("response", {})
    body = response.get("body", {})
    items = body.get("items", {})
    raw_item = items.get("item", []) if isinstance(items, dict) else []
    if isinstance(raw_item, dict):
        raw_item = [raw_item]
    if not isinstance(raw_item, list):
        return []
    return [item for item in raw_item if isinstance(item, dict)]


def fetch_tourapi_keyword(
    keyword: str,
    service_key: str,
    base_url: str,
    timeout: float,
    num_rows: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    params = {
        "serviceKey": service_key,
        "MobileOS": "ETC",
        "MobileApp": "kor_travel_recommendation",
        "_type": "json",
        "numOfRows": num_rows,
        "pageNo": 1,
        "keyword": keyword,
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
                "status": "api_error",
                "keyword": keyword,
                "items": [],
                "error": last_error,
            }

    if payload is None:
        return {
            "status": "api_error",
            "keyword": keyword,
            "items": [],
            "error": last_error or "empty response",
        }

    header = payload.get("response", {}).get("header", {})
    result_code = str(header.get("resultCode", ""))
    if result_code and result_code != "0000":
        return {
            "status": "api_error",
            "keyword": keyword,
            "items": [],
            "error": header.get("resultMsg", f"resultCode={result_code}"),
        }

    return {
        "status": "ok",
        "keyword": keyword,
        "items": parse_tourapi_items(payload),
        "error": None,
    }


def is_key_fallback_error(record: dict[str, Any]) -> bool:
    if record.get("status") != "api_error":
        return False
    error = str(record.get("error") or "").lower()
    return any(marker in error for marker in KEY_FALLBACK_ERROR_MARKERS)


def fetch_tourapi_keyword_with_keys(
    keyword: str,
    service_keys: list[tuple[str, str]],
    base_url: str,
    timeout: float,
    num_rows: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    last_record: dict[str, Any] | None = None
    for key_index, (key_name, service_key) in enumerate(service_keys):
        record = fetch_tourapi_keyword(
            keyword=keyword,
            service_key=service_key,
            base_url=base_url,
            timeout=timeout,
            num_rows=num_rows,
            retries=retries,
            retry_sleep=retry_sleep,
        )
        record["api_key_name"] = key_name
        last_record = record
        if is_key_fallback_error(record) and key_index < len(service_keys) - 1:
            continue
        return record

    return last_record or {
        "status": "api_error",
        "keyword": keyword,
        "items": [],
        "error": "no API keys provided",
    }


def load_keyword_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
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
            keyword_norm = record.get("keyword_norm")
            if keyword_norm:
                cache[keyword_norm] = record
    return cache


def append_keyword_cache(cache_path: Path, record: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def populate_tourapi_cache(
    keywords: pd.Series,
    cache_path: Path,
    service_keys: list[tuple[str, str]],
    base_url: str,
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
) -> dict[str, dict[str, Any]]:
    if parallel_workers < 1:
        raise ValueError("--parallel-workers must be at least 1")
    if parallel_workers > MAX_PARALLEL_WORKERS:
        raise ValueError(f"--parallel-workers must be <= {MAX_PARALLEL_WORKERS}")

    if refresh_cache and cache_path.exists():
        cache_path.unlink()
    if not service_keys:
        raise RuntimeError("No TourAPI service keys were provided.")

    cache = load_keyword_cache(cache_path)
    raw_keywords = (
        keywords.dropna()
        .astype("string")
        .drop_duplicates()
        .sort_values(kind="mergesort")
        .tolist()
    )

    pending: list[tuple[str, str]] = []
    seen_norms: set[str] = set()
    exhausted = False
    for keyword in raw_keywords:
        keyword_norm = normalize_keyword(keyword)
        if not keyword_norm or keyword_norm in seen_norms:
            continue
        cached_record = cache.get(keyword_norm)
        if cached_record is not None and not (
            retry_api_errors and cached_record.get("status") == "api_error"
        ):
            continue
        if max_api_calls is not None and len(pending) >= max_api_calls:
            exhausted = True
            break
        pending.append((keyword_norm, str(keyword)))
        seen_norms.add(keyword_norm)

    def fetch_record(keyword_norm: str, keyword: str) -> dict[str, Any]:
        result = fetch_tourapi_keyword_with_keys(
            keyword=keyword,
            service_keys=service_keys,
            base_url=base_url,
            timeout=timeout,
            num_rows=num_rows,
            retries=retries,
            retry_sleep=retry_sleep,
        )
        return {
            "keyword_norm": keyword_norm,
            "keyword": keyword,
            **result,
        }

    if parallel_workers == 1:
        for keyword_norm, keyword in pending:
            record = fetch_record(keyword_norm, keyword)
            cache[keyword_norm] = record
            append_keyword_cache(cache_path, record)
            if request_sleep > 0:
                time.sleep(request_sleep)
    else:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_keyword = {}
            for keyword_norm, keyword in pending:
                future = executor.submit(fetch_record, keyword_norm, keyword)
                future_to_keyword[future] = (keyword_norm, keyword)
                if request_sleep > 0:
                    time.sleep(request_sleep)

            for future in as_completed(future_to_keyword):
                keyword_norm, keyword = future_to_keyword[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "keyword_norm": keyword_norm,
                        "keyword": keyword,
                        "status": "api_error",
                        "items": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                cache[keyword_norm] = record
                append_keyword_cache(cache_path, record)

    if exhausted and not allow_partial_api:
        raise RuntimeError(
            "TourAPI call limit reached before all keywords were cached. "
            "Increase --max-api-calls or pass --allow-partial-api."
        )

    return cache
