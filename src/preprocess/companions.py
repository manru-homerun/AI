from __future__ import annotations

import pandas as pd


def companion_count(travel_with_profile: pd.DataFrame) -> pd.Series:
    if "TRAVEL_COMPANIONS_NUM" not in travel_with_profile.columns:
        return pd.Series(0, index=travel_with_profile.index, dtype="Int64")
    return pd.to_numeric(
        travel_with_profile["TRAVEL_COMPANIONS_NUM"],
        errors="coerce",
    ).fillna(0).astype("Int64")
