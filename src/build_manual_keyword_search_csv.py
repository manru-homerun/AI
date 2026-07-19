from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


AREA_NAMES = {
    "busan": "부산",
    "changwon": "창원",
    "daegu": "대구",
    "daejeon": "대전",
    "gwangju": "광주",
    "incheon": "인천",
    "seoul": "서울",
    "suwon": "수원",
}

OUTPUT_COLUMNS = [
    "area",
    "status",
    "error",
    "keyword",
    "title",
    "contentid",
    "contenttypeid",
    "addr1",
    "addr2",
    "mapx",
    "mapy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a manual TourAPI keyword review CSV from regional JSONL caches."
    )
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/manual_keyword_search_sample.csv"),
    )
    parser.add_argument(
        "--areas",
        nargs="+",
        choices=AREA_NAMES.keys(),
        default=list(AREA_NAMES.keys()),
    )
    return parser.parse_args()


def latest_cache_records(cache_path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            keyword_norm = str(record.get("keyword_norm") or record.get("keyword") or "")
            if keyword_norm:
                latest[keyword_norm] = record
    return latest


def empty_item_row(area: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "area": area,
        "status": record.get("status"),
        "error": record.get("error"),
        "keyword": record.get("keyword"),
        "title": "",
        "contentid": "",
        "contenttypeid": "",
        "addr1": "",
        "addr2": "",
        "mapx": "",
        "mapy": "",
    }


def item_row(area: str, record: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    return {
        "area": area,
        "status": record.get("status"),
        "error": record.get("error"),
        "keyword": record.get("keyword"),
        "title": item.get("title"),
        "contentid": item.get("contentid"),
        "contenttypeid": item.get("contenttypeid"),
        "addr1": item.get("addr1"),
        "addr2": item.get("addr2"),
        "mapx": item.get("mapx"),
        "mapy": item.get("mapy"),
    }


def build_rows(processed_root: Path, areas: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for area_slug in areas:
        area_name = AREA_NAMES[area_slug]
        cache_path = processed_root / f"{area_slug}_tourapi_keyword_cache.jsonl"
        records = latest_cache_records(cache_path)
        sorted_records = sorted(
            records.values(),
            key=lambda record: str(record.get("keyword") or record.get("keyword_norm") or ""),
        )

        for record in sorted_records:
            keyword = str(record.get("keyword") or "")
            items = record.get("items") or []
            if not items:
                dedupe_key = (area_name, keyword, "", "")
                if dedupe_key not in seen:
                    seen.add(dedupe_key)
                    rows.append(empty_item_row(area_name, record))
                continue

            for item in items:
                contentid = str(item.get("contentid") or "")
                title = str(item.get("title") or "")
                dedupe_key = (area_name, keyword, contentid, title)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                rows.append(item_row(area_name, record, item))

    return rows


def write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(args.processed_root, args.areas)
    write_rows(rows, args.output_path)

    print(f"wrote: {args.output_path}")
    print(f"rows: {len(rows):,}")
    for area in AREA_NAMES.values():
        count = sum(1 for row in rows if row["area"] == area)
        if count:
            print(f"{area}: {count:,}")


if __name__ == "__main__":
    main()
