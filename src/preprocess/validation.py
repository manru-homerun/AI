from __future__ import annotations

import pandas as pd

from .config import INPUT_COLUMNS, TRAVEL_SEQ_COLUMNS


def validate_outputs(input_df: pd.DataFrame, travel_seq: pd.DataFrame) -> None:
    if list(input_df.columns) != INPUT_COLUMNS:
        raise AssertionError("total_input columns do not match expected order")
    if list(travel_seq.columns) != TRAVEL_SEQ_COLUMNS:
        raise AssertionError("total_travel_seq columns do not match expected order")

    input_ids = set(input_df["trip_id"].dropna())
    seq_ids = set(travel_seq["travel_id"].dropna())
    if input_ids != seq_ids:
        missing_seq = sorted(input_ids - seq_ids)[:10]
        missing_input = sorted(seq_ids - input_ids)[:10]
        raise AssertionError(
            "Trip id mismatch between total_input and total_travel_seq. "
            f"missing_seq={missing_seq}, missing_input={missing_input}"
        )

    if not input_df["trip_days"].between(1, 3).all():
        raise AssertionError("trip_days contains values outside 1..3")
    if not travel_seq["day_index"].between(1, 3).all():
        raise AssertionError("day_index contains values outside 1..3")

    for (travel_id, day_index), group in travel_seq.groupby(["travel_id", "day_index"]):
        expected = list(range(1, len(group) + 1))
        actual = group["visit_order"].tolist()
        if actual != expected:
            raise AssertionError(
                "Non-contiguous visit_order for "
                f"travel_id={travel_id!r}, day_index={day_index!r}: {actual[:10]}"
            )

    for column in ["X_COORD", "Y_COORD"]:
        numeric = pd.to_numeric(travel_seq[column].dropna(), errors="coerce")
        if numeric.isna().any():
            raise AssertionError(f"{column} contains non-numeric values")

    content_id_exists = travel_seq["CONTENT_ID"].notna()
    content_type_missing = travel_seq["CONTENT_TYPE_ID"].isna()
    if (content_id_exists & content_type_missing).any():
        raise AssertionError("Rows with CONTENT_ID must also have CONTENT_TYPE_ID")
