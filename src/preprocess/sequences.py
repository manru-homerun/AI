from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .config import TRAVEL_SEQ_COLUMNS, VISIT_LOCATION_COLUMNS


def clean_code(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    digits = re.sub(r"\D", "", str(value))
    return digits if digits else pd.NA


def legal_dong_code(lotno_cd: Any, sgg_cd: Any) -> Any:
    lotno = clean_code(lotno_cd)
    if pd.notna(lotno):
        return str(lotno)

    sgg = clean_code(sgg_cd)
    if pd.notna(sgg) and len(str(sgg)) == 10:
        return str(sgg)
    return pd.NA


def build_travel_seq(visit: pd.DataFrame, travel: pd.DataFrame) -> pd.DataFrame:
    available_visit_columns = [column for column in VISIT_LOCATION_COLUMNS if column in visit.columns]
    seq = visit[available_visit_columns].merge(
        travel[["TRAVEL_ID", "TRAVEL_START_YMD"]],
        on="TRAVEL_ID",
        how="inner",
        validate="many_to_one",
    )

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

    x_coord = pd.to_numeric(seq["X_COORD"], errors="coerce")
    y_coord = pd.to_numeric(seq["Y_COORD"], errors="coerce")

    travel_seq = pd.DataFrame(
        {
            "travel_id": seq["TRAVEL_ID"],
            "day_index": seq["day_index"].astype("int64"),
            "visit_order": seq["visit_order"].astype("int64"),
            "visit_area_id": seq["VISIT_AREA_ID"],
            "visit_area_nm": seq["VISIT_AREA_NM"],
            "X_COORD": x_coord,
            "Y_COORD": y_coord,
            "LEGAL_DONG_CD": [
                legal_dong_code(lotno_cd, sgg_cd)
                for lotno_cd, sgg_cd in zip(seq.get("LOTNO_CD"), seq.get("SGG_CD"))
            ],
            "SGG_CD": seq["SGG_CD"].map(clean_code),
            "ROAD_NM_ADDR": seq["ROAD_NM_ADDR"],
            "LOTNO_ADDR": seq["LOTNO_ADDR"],
            "CONTENT_ID": pd.NA,
            "CONTENT_TYPE_ID": pd.NA,
            "TOURAPI_MATCH_STATUS": "not_queried",
            "TOURAPI_MATCH_SCORE": pd.NA,
        }
    )
    return travel_seq[TRAVEL_SEQ_COLUMNS]
