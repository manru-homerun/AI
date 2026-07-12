from __future__ import annotations

import pandas as pd


def companion_count(travel_with_profile: pd.DataFrame) -> pd.Series | int:
    return travel_with_profile.get("TRAVEL_COMPANIONS_NUM", 0)
