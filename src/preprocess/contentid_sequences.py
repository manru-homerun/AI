from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import TOURAPI_TARGET_CONFIGS, TRAVEL_SEQ_COLUMNS


CONTENTID_SEQ_COLUMNS = [
    "travel_id",
    "day_index",
    "visit_order",
    "content_id",
    "content_type_id",
]
THEME_AREAS = [
    "seoul",
    "suwon",
    "incheon",
    "daejeon",
    "daegu",
    "gwangju",
    "busan",
    "changwon",
]
TRAVEL_SEQ_WITH_AREA_COLUMNS = ["area", *TRAVEL_SEQ_COLUMNS]
MAPPING_USED_COLUMNS = [
    "keyword",
    "contentid",
    "contenttypeid",
    "title",
    "area",
    "matched_visit_rows",
]
REVIEW_COLUMNS = [
    "travel_id",
    "day_index",
    "reason",
    "original_rows",
    "contentid_rows",
    "duplicate_contentids",
    "ambiguous_keywords",
]


@dataclass(frozen=True)
class MappingRecord:
    keyword: str
    contentid: str
    contenttypeid: str
    title: str
    area: str


def normalize_key(value: Any) -> str:
    return str(value or "").strip()


def clean_code(value: Any) -> str:
    return "".join(character for character in normalize_key(value) if character.isdigit())


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        rows = [{key: value or "" for key, value in row.items() if key is not None} for row in reader]
        return reader.fieldnames, rows


def write_csv_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def required_columns(path: Path, columns: list[str], required: list[str]) -> None:
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def target_code_prefixes(target_config: dict[str, Any]) -> list[str]:
    return target_config.get("code_prefixes") or [target_config["code_prefix"]]


def target_address_keywords(target_config: dict[str, Any]) -> list[str]:
    return target_config.get("address_keywords") or [target_config["address_keyword"]]


def row_theme_area(row: dict[str, str]) -> str:
    legal_code = clean_code(row.get("LEGAL_DONG_CD"))
    sgg_code = clean_code(row.get("SGG_CD"))
    address = " ".join(
        value
        for value in [
            normalize_key(row.get("ROAD_NM_ADDR")),
            normalize_key(row.get("LOTNO_ADDR")),
        ]
        if value
    )

    for area in THEME_AREAS:
        target_config = TOURAPI_TARGET_CONFIGS[area]
        code_prefixes = tuple(target_code_prefixes(target_config))
        if legal_code.startswith(code_prefixes) or sgg_code.startswith(code_prefixes):
            return area
        if any(keyword in address for keyword in target_address_keywords(target_config)):
            return area
    return ""


def travel_theme_areas(
    seq_rows: list[dict[str, str]],
    area_threshold: float,
) -> dict[str, str]:
    total_counts: Counter[str] = Counter()
    area_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in seq_rows:
        travel_id = normalize_key(row.get("travel_id"))
        if not travel_id:
            continue
        total_counts[travel_id] += 1
        area = row_theme_area(row)
        if area:
            area_counts[travel_id][area] += 1

    themes: dict[str, str] = {}
    for travel_id, total_count in total_counts.items():
        if total_count == 0 or not area_counts[travel_id]:
            themes[travel_id] = ""
            continue

        area, area_count = area_counts[travel_id].most_common(1)[0]
        themes[travel_id] = area if area_count / total_count >= area_threshold else ""
    return themes


def attach_travel_areas(
    seq_rows: list[dict[str, str]],
    area_threshold: float,
) -> tuple[list[dict[str, str]], Counter[str]]:
    theme_by_travel = travel_theme_areas(seq_rows, area_threshold)
    output_rows: list[dict[str, str]] = []
    area_summary: Counter[str] = Counter()

    for row in seq_rows:
        area = theme_by_travel.get(normalize_key(row.get("travel_id")), "")
        new_row = {"area": area, **row}
        new_row["area"] = area
        output_rows.append(new_row)
        if area:
            area_summary[area] += 1
    return output_rows, area_summary


def load_mapping_records(mapping_path: Path) -> tuple[dict[str, MappingRecord], set[str]]:
    columns, rows = read_csv_rows(mapping_path)
    required_columns(
        mapping_path,
        columns,
        ["keyword", "contentid", "contenttypeid", "title", "area"],
    )

    grouped: dict[str, list[MappingRecord]] = defaultdict(list)
    for row in rows:
        keyword = normalize_key(row.get("keyword"))
        contentid = normalize_key(row.get("contentid"))
        contenttypeid = normalize_key(row.get("contenttypeid"))
        if not keyword or not contentid or not contenttypeid:
            continue
        grouped[keyword].append(
            MappingRecord(
                keyword=keyword,
                contentid=contentid,
                contenttypeid=contenttypeid,
                title=normalize_key(row.get("title")),
                area=normalize_key(row.get("area")),
            )
        )

    unique: dict[str, MappingRecord] = {}
    ambiguous: set[str] = set()
    for keyword, records in grouped.items():
        content_pairs = {(record.contentid, record.contenttypeid) for record in records}
        if len(content_pairs) == 1:
            unique[keyword] = records[-1]
        else:
            ambiguous.add(keyword)
    return unique, ambiguous


def attach_contentids(
    seq_rows: list[dict[str, str]],
    mappings: dict[str, MappingRecord],
    ambiguous_keywords: set[str],
) -> tuple[list[dict[str, str]], Counter[str], dict[tuple[str, str], set[str]]]:
    matched_counts: Counter[str] = Counter()
    ambiguous_by_day: dict[tuple[str, str], set[str]] = defaultdict(set)
    output_rows: list[dict[str, str]] = []

    for row in seq_rows:
        new_row = dict(row)
        keyword = normalize_key(row.get("visit_area_nm"))
        has_existing = bool(normalize_key(row.get("CONTENT_ID")))

        if keyword in ambiguous_keywords and not has_existing:
            ambiguous_by_day[(row.get("travel_id", ""), row.get("day_index", ""))].add(keyword)

        if not has_existing and keyword in mappings:
            mapping = mappings[keyword]
            new_row["CONTENT_ID"] = mapping.contentid
            new_row["CONTENT_TYPE_ID"] = mapping.contenttypeid
            matched_counts[keyword] += 1

            if not normalize_key(new_row.get("TOURAPI_MATCH_STATUS")):
                new_row["TOURAPI_MATCH_STATUS"] = "manual_keyword_match"
        elif has_existing and keyword in mappings:
            matched_counts[keyword] += 1

        output_rows.append(new_row)

    return output_rows, matched_counts, ambiguous_by_day


def build_mapping_used_rows(
    mappings: dict[str, MappingRecord],
    matched_counts: Counter[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for keyword in sorted(matched_counts):
        mapping = mappings.get(keyword)
        if mapping is None:
            continue
        rows.append(
            {
                "keyword": mapping.keyword,
                "contentid": mapping.contentid,
                "contenttypeid": mapping.contenttypeid,
                "title": mapping.title,
                "area": mapping.area,
                "matched_visit_rows": matched_counts[keyword],
            }
        )
    return rows


def build_contentid_seq_rows(
    seq_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[str]], dict[tuple[str, str], int]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    original_counts: Counter[tuple[str, str]] = Counter()
    for row in seq_rows:
        key = (row.get("travel_id", ""), row.get("day_index", ""))
        original_counts[key] += 1
        if normalize_key(row.get("CONTENT_ID")):
            grouped[key].append(row)

    output_rows: list[dict[str, Any]] = []
    duplicate_by_day: dict[tuple[str, str], list[str]] = {}
    content_counts: dict[tuple[str, str], int] = {}

    for key in sorted(original_counts):
        seen: set[str] = set()
        duplicates: list[str] = []
        kept_rows: list[dict[str, str]] = []

        for row in grouped.get(key, []):
            content_id = normalize_key(row.get("CONTENT_ID"))
            if content_id in seen:
                duplicates.append(content_id)
                continue
            seen.add(content_id)
            kept_rows.append(row)

        for index, row in enumerate(kept_rows, start=1):
            output_rows.append(
                {
                    "travel_id": row.get("travel_id", ""),
                    "day_index": row.get("day_index", ""),
                    "visit_order": index,
                    "content_id": normalize_key(row.get("CONTENT_ID")),
                    "content_type_id": normalize_key(row.get("CONTENT_TYPE_ID")),
                }
            )

        content_counts[key] = len(kept_rows)
        if duplicates:
            duplicate_by_day[key] = sorted(set(duplicates))

    return output_rows, duplicate_by_day, dict(original_counts)


def build_review_rows(
    original_counts: dict[tuple[str, str], int],
    content_counts: dict[tuple[str, str], int],
    duplicate_by_day: dict[tuple[str, str], list[str]],
    ambiguous_by_day: dict[tuple[str, str], set[str]],
    min_contentids_per_day: int,
) -> list[dict[str, Any]]:
    review_rows: list[dict[str, Any]] = []
    all_keys = sorted(set(original_counts) | set(content_counts) | set(ambiguous_by_day))

    for travel_id, day_index in all_keys:
        key = (travel_id, day_index)
        reasons: list[str] = []
        if content_counts.get(key, 0) < min_contentids_per_day:
            reasons.append("too_few_contentids")
        if key in duplicate_by_day:
            reasons.append("duplicate_contentid")
        if key in ambiguous_by_day:
            reasons.append("ambiguous_keyword")
        if not reasons:
            continue

        review_rows.append(
            {
                "travel_id": travel_id,
                "day_index": day_index,
                "reason": "|".join(reasons),
                "original_rows": original_counts.get(key, 0),
                "contentid_rows": content_counts.get(key, 0),
                "duplicate_contentids": "|".join(duplicate_by_day.get(key, [])),
                "ambiguous_keywords": "|".join(sorted(ambiguous_by_day.get(key, set()))),
            }
        )
    return review_rows


def validate_outputs(
    original_columns: list[str],
    original_rows: list[dict[str, str]],
    with_contentid_rows: list[dict[str, str]],
    contentid_seq_rows: list[dict[str, Any]],
    area_threshold: float,
) -> None:
    if original_columns != TRAVEL_SEQ_COLUMNS:
        raise AssertionError("total_travel_seq columns do not match expected order")
    if len(original_rows) != len(with_contentid_rows):
        raise AssertionError("total_travel_seq_with_contentid row count changed")

    if list(with_contentid_rows[0].keys())[:1] != ["area"] and with_contentid_rows:
        raise AssertionError("total_travel_seq_with_contentid must start with area")

    for row in with_contentid_rows:
        if normalize_key(row.get("CONTENT_ID")) and not normalize_key(row.get("CONTENT_TYPE_ID")):
            raise AssertionError("Rows with CONTENT_ID must also have CONTENT_TYPE_ID")

    expected_area_by_travel = travel_theme_areas(original_rows, area_threshold)
    actual_area_by_travel: dict[str, str] = {}
    for row in with_contentid_rows:
        travel_id = normalize_key(row.get("travel_id"))
        area = normalize_key(row.get("area"))
        if travel_id in actual_area_by_travel and actual_area_by_travel[travel_id] != area:
            raise AssertionError(f"area is not consistent within travel_id={travel_id!r}")
        actual_area_by_travel[travel_id] = area
    if actual_area_by_travel != expected_area_by_travel:
        raise AssertionError("area values do not match the travel-level threshold policy")

    for row in contentid_seq_rows:
        if not normalize_key(row.get("content_id")):
            raise AssertionError("total_contentid_seq contains empty content_id")

    day_orders: dict[tuple[str, str], list[int]] = defaultdict(list)
    day_contentids: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in contentid_seq_rows:
        key = (str(row["travel_id"]), str(row["day_index"]))
        day_orders[key].append(int(row["visit_order"]))
        day_contentids[key].append(str(row["content_id"]))

    for key, orders in day_orders.items():
        expected = list(range(1, len(orders) + 1))
        if orders != expected:
            raise AssertionError(f"Non-contiguous visit_order for {key}: {orders[:10]}")
        contentids = day_contentids[key]
        if len(contentids) != len(set(contentids)):
            raise AssertionError(f"Duplicate content_id remains for {key}")


def process_contentid_sequences(
    seq_path: Path,
    mapping_path: Path,
    output_root: Path,
    min_contentids_per_day: int,
    area_threshold: float,
) -> dict[str, int]:
    seq_columns, seq_rows = read_csv_rows(seq_path)
    required_columns(seq_path, seq_columns, TRAVEL_SEQ_COLUMNS)

    mappings, ambiguous_keywords = load_mapping_records(mapping_path)
    with_contentid_rows, matched_counts, ambiguous_by_day = attach_contentids(
        seq_rows,
        mappings,
        ambiguous_keywords,
    )
    with_contentid_rows, area_summary = attach_travel_areas(
        with_contentid_rows,
        area_threshold,
    )
    contentid_seq_rows, duplicate_by_day, original_counts = build_contentid_seq_rows(
        with_contentid_rows
    )
    review_rows = build_review_rows(
        original_counts=original_counts,
        content_counts={
            key: count
            for key, count in Counter(
                (row["travel_id"], row["day_index"]) for row in contentid_seq_rows
            ).items()
        },
        duplicate_by_day=duplicate_by_day,
        ambiguous_by_day=ambiguous_by_day,
        min_contentids_per_day=min_contentids_per_day,
    )
    mapping_used_rows = build_mapping_used_rows(mappings, matched_counts)

    write_csv_rows(
        output_root / "total_travel_seq_with_contentid.csv",
        TRAVEL_SEQ_WITH_AREA_COLUMNS,
        with_contentid_rows,
    )
    write_csv_rows(
        output_root / "total_contentid_seq.csv",
        CONTENTID_SEQ_COLUMNS,
        contentid_seq_rows,
    )
    write_csv_rows(
        output_root / "contentid_mapping_used.csv",
        MAPPING_USED_COLUMNS,
        mapping_used_rows,
    )
    write_csv_rows(
        output_root / "contentid_seq_review.csv",
        REVIEW_COLUMNS,
        review_rows,
    )

    validate_outputs(
        seq_columns,
        seq_rows,
        with_contentid_rows,
        contentid_seq_rows,
        area_threshold,
    )

    return {
        "source_rows": len(seq_rows),
        "with_contentid_rows": len(with_contentid_rows),
        "contentid_seq_rows": len(contentid_seq_rows),
        "mapping_used_rows": len(mapping_used_rows),
        "review_rows": len(review_rows),
        "ambiguous_keywords": len(ambiguous_keywords),
        "duplicate_days": len(duplicate_by_day),
        "area_assigned_rows": sum(area_summary.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build contentID-based travel sequence CSVs from total_travel_seq."
    )
    parser.add_argument(
        "--seq-path",
        type=Path,
        default=Path("data/processed/total_travel_seq.csv"),
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=Path("data/processed/manual_keyword_search_sample.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-contentids-per-day", type=int, default=2)
    parser.add_argument("--area-threshold", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = process_contentid_sequences(
        seq_path=args.seq_path,
        mapping_path=args.mapping_path,
        output_root=args.output_root,
        min_contentids_per_day=args.min_contentids_per_day,
        area_threshold=args.area_threshold,
    )
    for key, value in summary.items():
        print(f"{key}: {value:,}")


if __name__ == "__main__":
    main()
