from __future__ import annotations

import argparse
from pathlib import Path

from .config import TOURAPI_BASE_URL, TOURAPI_TARGET_CONFIGS
from .pipeline import PipelineOptions, process_total


def parse_args() -> PipelineOptions:
    parser = argparse.ArgumentParser(description="Build total decoder-model preprocessing CSVs.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--env-path", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-name", default="DATA_OPENAPI_KEY")
    parser.add_argument("--api-key-names", nargs="+", default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--tourapi-base-url", default=TOURAPI_BASE_URL)
    parser.add_argument("--call-tourapi", action="store_true")
    parser.add_argument(
        "--tourapi-target",
        choices=["all", *TOURAPI_TARGET_CONFIGS.keys()],
        default="all",
    )
    parser.add_argument("--private-place-pattern-path", type=Path, default=None)
    parser.add_argument("--refresh-tourapi-cache", action="store_true")
    parser.add_argument("--max-api-calls", type=int, default=None)
    parser.add_argument("--allow-partial-api", action="store_true")
    parser.add_argument("--retry-api-errors", action="store_true")
    parser.add_argument("--skip-tourapi", action="store_true")
    parser.add_argument("--request-sleep", type=float, default=0.05)
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--num-rows", type=int, default=20)
    args = parser.parse_args()
    return PipelineOptions(**vars(args))


def main() -> None:
    options = parse_args()
    input_df, travel_seq = process_total(options)
    print(
        "total: "
        f"input={len(input_df):,} rows, "
        f"travel_seq={len(travel_seq):,} rows, "
        f"matched={travel_seq['CONTENT_ID'].notna().sum():,} rows"
    )
    print(input_df.head(2).to_string(index=False))
    print(travel_seq.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
