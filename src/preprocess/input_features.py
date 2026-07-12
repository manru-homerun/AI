from __future__ import annotations

import pandas as pd

from .companions import companion_count
from .config import INPUT_COLUMNS


def first_code(series: pd.Series) -> pd.Series:
    return series.fillna("").astype("string").str.split(";").str[0].replace("", pd.NA)


def join_non_empty(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    available_columns = [column for column in columns if column in df.columns]

    def join_row(row: pd.Series) -> str:
        values = [str(value) for value in row if pd.notna(value) and str(value) != ""]
        return ";".join(values)

    return df[available_columns].apply(join_row, axis=1).replace("", pd.NA)


def build_input(travel: pd.DataFrame, traveller: pd.DataFrame) -> pd.DataFrame:
    traveller = traveller.drop_duplicates("TRAVELER_ID", keep="first")
    travel_with_profile = travel.merge(
        traveller,
        on="TRAVELER_ID",
        how="left",
        validate="many_to_one",
        suffixes=("", "_traveller"),
    )

    input_df = pd.DataFrame(
        {
            "trip_id": travel_with_profile["TRAVEL_ID"],
            "area_code": travel_with_profile["__region"],
            "trip_days": travel_with_profile["trip_days"],
            "theme": first_code(travel_with_profile["TRAVEL_PURPOSE"]),
            "has_child": 0,
            "has_elderly": 0,
            "has_disabled": 0,
            "companion_count": companion_count(travel_with_profile),
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
