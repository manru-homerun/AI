from __future__ import annotations

from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pandas as pd

from .config import REGIONS, TABLE_PREFIXES, ZIP_NAMES


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


def region_from_path(path: Path) -> str:
    for part in path.parts:
        if part in REGIONS:
            return part
    raise ValueError(f"Could not infer region from path: {path}")


def find_label_zip_paths(raw_root: Path) -> list[Path]:
    zip_paths = [
        path
        for path in sorted(raw_root.rglob("*_csv.zip"))
        if path.name in ZIP_NAMES and {"Training", "Validation"}.intersection(path.parts)
    ]
    if not zip_paths:
        raise FileNotFoundError(f"No TL_csv.zip or VL_csv.zip files found under {raw_root}")
    return zip_paths


def load_all_tables(raw_root: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, list[pd.DataFrame]] = {key: [] for key in TABLE_PREFIXES}

    for zip_path in find_label_zip_paths(raw_root):
        region = region_from_path(zip_path)
        for table_name, prefix in TABLE_PREFIXES.items():
            table = read_csv_from_zip(zip_path, prefix)
            table["__region"] = region
            table["__source_zip"] = str(zip_path)
            tables[table_name].append(table)

    return {
        table_name: pd.concat(parts, ignore_index=True)
        for table_name, parts in tables.items()
    }


def read_env_key(env_path: Path, key_name: str = "DATA_OPENAPI_KEY") -> str | None:
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != key_name:
            continue
        return value.strip().strip('"').strip("'")
    return None


def read_env_keys(env_path: Path, key_names: list[str] | tuple[str, ...]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for key_name in key_names:
        value = read_env_key(env_path, key_name)
        if value:
            keys.append((key_name, value))
    return keys


def save_outputs(input_df: pd.DataFrame, travel_seq: pd.DataFrame, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    input_df.to_csv(output_root / "total_input.csv", index=False, encoding="utf-8-sig")
    travel_seq.to_csv(output_root / "total_travel_seq.csv", index=False, encoding="utf-8-sig")
