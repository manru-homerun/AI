from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from preprocess.config import TOURAPI_LOCATION_BASE_URL, TOURAPI_LOCATION_RADIUS
from preprocess.io import read_env_keys
from preprocess.tourapi import fetch_tourapi_location_with_keys


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
        description="Search TourAPI by location and print rows for manual review CSV."
    )
    parser.add_argument("keyword", help="Original place name to write in the keyword column.")
    parser.add_argument("mapx", type=float, help="Longitude/X coordinate for TourAPI mapX.")
    parser.add_argument("mapy", type=float, help="Latitude/Y coordinate for TourAPI mapY.")
    parser.add_argument(
        "radius",
        nargs="?",
        type=int,
        default=TOURAPI_LOCATION_RADIUS,
        help="Search radius in meters.",
    )
    parser.add_argument("--area", default="", help="Area value to print in the output rows.")
    parser.add_argument("--env-path", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-name", default="DATA_OPENAPI_KEY")
    parser.add_argument("--api-key-names", nargs="+", default=None)
    parser.add_argument("--base-url", default=TOURAPI_LOCATION_BASE_URL)
    parser.add_argument("--num-rows", type=int, default=10)
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--print-header", action="store_true")
    return parser.parse_args()


def item_row(
    area: str,
    keyword: str,
    item: dict[str, Any],
    status: str,
    error: str | None,
) -> dict[str, Any]:
    return {
        "area": area,
        "status": status,
        "error": error,
        "keyword": keyword,
        "title": item.get("title"),
        "contentid": item.get("contentid"),
        "contenttypeid": item.get("contenttypeid"),
        "addr1": item.get("addr1"),
        "addr2": item.get("addr2"),
        "mapx": item.get("mapx"),
        "mapy": item.get("mapy"),
    }


def search_location(
    keyword: str,
    service_keys: list[tuple[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    record = fetch_tourapi_location_with_keys(
        map_x=args.mapx,
        map_y=args.mapy,
        radius=args.radius,
        service_keys=service_keys,
        base_url=args.base_url,
        timeout=args.request_timeout,
        num_rows=args.num_rows,
        retries=args.request_retries,
        retry_sleep=args.retry_sleep,
    )
    items = record.get("items") or []
    if not items:
        return [
            item_row(
                area=args.area,
                keyword=keyword,
                item={},
                status=record.get("status", "unknown"),
                error=record.get("error"),
            )
        ]
    return [
        item_row(
            area=args.area,
            keyword=keyword,
            item=item,
            status=record.get("status", "unknown"),
            error=record.get("error"),
        )
        for item in items
    ]


def print_rows(rows: list[dict[str, Any]], print_header: bool) -> None:
    import sys

    writer = csv.DictWriter(sys.stdout, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
    if print_header:
        writer.writeheader()
    writer.writerows(rows)


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.radius < 1:
        raise ValueError("radius must be at least 1 meter.")

    key_names = args.api_key_names or [args.api_key_name]
    service_keys = read_env_keys(args.env_path, key_names)
    if not service_keys:
        raise RuntimeError(f"None of these API keys were found in .env: {key_names}")

    rows = search_location(args.keyword, service_keys, args)
    print_rows(rows, args.print_header)
    if args.output_path is not None:
        write_csv(rows, args.output_path)
        print(f"\nwrote: {args.output_path}")


if __name__ == "__main__":
    main()
