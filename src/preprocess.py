from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pandas as pd


REGIONS = ("central", "east", "west")

TABLE_PREFIXES = {
    "travel": "tn_travel_",
    "traveller": "tn_traveller_master_",
    "visit": "tn_visit_area_info_",
}

INPUT_COLUMNS = [
    "trip_id",
    "area_code",
    "trip_days",
    "theme",
    "has_child",
    "has_elderly",
    "has_disabled",
    "companion_count",
    "p0_age",
    "p0_gender",
    "p0_style",
    "p0_home",
    "p0_preferred",
    "p1_age",
    "p1_gender",
    "p1_style",
    "p1_home",
    "p1_preferred",
]

TRAVEL_SEQ_COLUMNS = [
    "travel_id",
    "day_index",
    "visit_order",
    "visit_area_id",
    "visit_area_nm",
]


def read_csv_from_zip(zip_path: Path, filename_prefix: str) -> pd.DataFrame:
    with ZipFile(zip_path) as zf:
        matches = [
            name
            for name in zf.namelist()
            if PurePosixPath(name).name.startswith(filename_prefix)
            and PurePosixPath(name).suffix.lower() == ".csv"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {filename_prefix!r} CSV in {zip_path}, "
                f"found {len(matches)}: {matches}"
            )
        with zf.open(matches[0]) as fp:
            return pd.read_csv(fp, encoding="utf-8-sig", dtype="string")


def load_region_tables(raw_root: Path, region: str) -> dict[str, pd.DataFrame]:
    region_root = raw_root / region
    zip_paths = sorted(region_root.rglob("*_csv.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No *_csv.zip files found under {region_root}")

    tables: dict[str, list[pd.DataFrame]] = {key: [] for key in TABLE_PREFIXES}
    for zip_path in zip_paths:
        for table_name, prefix in TABLE_PREFIXES.items():
            tables[table_name].append(read_csv_from_zip(zip_path, prefix))

    return {
        table_name: pd.concat(parts, ignore_index=True)
        for table_name, parts in tables.items()
    }


def first_code(series: pd.Series) -> pd.Series:
    return series.fillna("").astype("string").str.split(";").str[0].replace("", pd.NA)


def join_non_empty(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    def join_row(row: pd.Series) -> str:
        values = [str(value) for value in row if pd.notna(value) and str(value) != ""]
        return ";".join(values)

    return df[columns].apply(join_row, axis=1).replace("", pd.NA)


def build_input(
    travel: pd.DataFrame,
    traveller: pd.DataFrame,
    region: str,
) -> pd.DataFrame:
    travel_with_profile = travel.merge(traveller, on="TRAVELER_ID", how="left", validate="one_to_one")

    input_df = pd.DataFrame(
        {
            "trip_id": travel_with_profile["TRAVEL_ID"],
            "area_code": region,
            "trip_days": travel_with_profile["trip_days"],
            "theme": first_code(travel_with_profile["TRAVEL_PURPOSE"]),
            "has_child": 0,
            "has_elderly": 0,
            "has_disabled": 0,
            "companion_count": 0,
            "p0_age": travel_with_profile["AGE_GRP"],
            "p0_gender": travel_with_profile["GENDER"],
            "p0_style": join_non_empty(
                travel_with_profile,
                [f"TRAVEL_STYL_{idx}" for idx in range(1, 9)],
            ),
            "p0_home": travel_with_profile["RESIDENCE_SGG_CD"],
            "p0_preferred": join_non_empty(
                travel_with_profile,
                [f"TRAVEL_LIKE_SGG_{idx}" for idx in range(1, 4)],
            ),
            "p1_age": pd.NA,
            "p1_gender": pd.NA,
            "p1_style": pd.NA,
            "p1_home": pd.NA,
            "p1_preferred": pd.NA,
        }
    )
    return input_df[INPUT_COLUMNS]


def build_travel_seq(visit: pd.DataFrame, travel: pd.DataFrame) -> pd.DataFrame:
    seq = visit.merge(
        travel[["TRAVEL_ID", "TRAVEL_START_YMD"]],
        on="TRAVEL_ID",
        how="inner",
        validate="many_to_one",
    ).copy()

    visit_start = pd.to_datetime(seq["VISIT_START_YMD"], errors="coerce")
    travel_start = pd.to_datetime(seq["TRAVEL_START_YMD"], errors="coerce")
    seq["day_index"] = (visit_start - travel_start).dt.days + 1
    seq["source_visit_order"] = pd.to_numeric(seq["VISIT_ORDER"], errors="coerce")
    seq["source_row"] = range(len(seq))

    seq = seq[seq["day_index"].between(1, 3)].copy()
    seq = seq.sort_values(
        ["TRAVEL_ID", "day_index", "source_visit_order", "source_row"],
        kind="mergesort",
    )
    seq["visit_order"] = seq.groupby(["TRAVEL_ID", "day_index"]).cumcount() + 1

    travel_seq = pd.DataFrame(
        {
            "travel_id": seq["TRAVEL_ID"],
            "day_index": seq["day_index"].astype("int64"),
            "visit_order": seq["visit_order"].astype("int64"),
            "visit_area_id": seq["VISIT_AREA_ID"],
            "visit_area_nm": seq["VISIT_AREA_NM"],
        }
    )
    return travel_seq[TRAVEL_SEQ_COLUMNS]


def validate_outputs(input_df: pd.DataFrame, travel_seq: pd.DataFrame, region: str) -> None:
    if list(input_df.columns) != INPUT_COLUMNS:
        raise AssertionError(f"{region}: input columns do not match expected order")
    if list(travel_seq.columns) != TRAVEL_SEQ_COLUMNS:
        raise AssertionError(f"{region}: travel_seq columns do not match expected order")

    input_ids = set(input_df["trip_id"].dropna())
    seq_ids = set(travel_seq["travel_id"].dropna())
    if input_ids != seq_ids:
        missing_seq = sorted(input_ids - seq_ids)[:10]
        missing_input = sorted(seq_ids - input_ids)[:10]
        raise AssertionError(
            f"{region}: trip id mismatch. "
            f"missing_seq={missing_seq}, missing_input={missing_input}"
        )

    if not input_df["trip_days"].between(1, 3).all():
        raise AssertionError(f"{region}: trip_days contains values outside 1..3")
    if not travel_seq["day_index"].between(1, 3).all():
        raise AssertionError(f"{region}: day_index contains values outside 1..3")

    for (travel_id, day_index), group in travel_seq.groupby(["travel_id", "day_index"]):
        expected = list(range(1, len(group) + 1))
        actual = group["visit_order"].tolist()
        if actual != expected:
            raise AssertionError(
                f"{region}: non-contiguous visit_order for "
                f"{travel_id=} {day_index=}: {actual[:10]}"
            )


def process_region(raw_root: Path, output_root: Path, region: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = load_region_tables(raw_root, region)
    travel = tables["travel"].copy()
    traveller = tables["traveller"].copy()
    visit = tables["visit"].copy()

    start = pd.to_datetime(travel["TRAVEL_START_YMD"], errors="coerce")
    end = pd.to_datetime(travel["TRAVEL_END_YMD"], errors="coerce")
    travel["trip_days"] = (end - start).dt.days + 1
    travel = travel[travel["trip_days"].between(1, 3)].copy()
    travel["trip_days"] = travel["trip_days"].astype("int64")

    input_df = build_input(travel, traveller, region)
    travel_seq = build_travel_seq(visit, travel)

    # Keep the model input aligned with trips that have at least one target visit.
    seq_trip_ids = set(travel_seq["travel_id"].dropna())
    input_df = input_df[input_df["trip_id"].isin(seq_trip_ids)].copy()

    validate_outputs(input_df, travel_seq, region)

    output_root.mkdir(parents=True, exist_ok=True)
    input_df.to_csv(output_root / f"{region}_input.csv", index=False, encoding="utf-8-sig")
    travel_seq.to_csv(output_root / f"{region}_travel_seq.csv", index=False, encoding="utf-8-sig")

    return input_df, travel_seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build decoder model preprocessing CSVs.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--regions", nargs="+", default=list(REGIONS), choices=REGIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for region in args.regions:
        input_df, travel_seq = process_region(args.raw_root, args.output_root, region)
        print(
            f"{region}: input={len(input_df):,} rows, "
            f"travel_seq={len(travel_seq):,} rows"
        )
        print(input_df.head(2).to_string(index=False))
        print(travel_seq.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
