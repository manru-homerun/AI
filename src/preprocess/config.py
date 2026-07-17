from __future__ import annotations

REGIONS = ("central", "east", "west")
ZIP_NAMES = ("TL_csv.zip", "VL_csv.zip")
TOURAPI_BASE_URL = "https://apis.data.go.kr/B551011/KorService2/searchKeyword2"
TOURAPI_LOCATION_BASE_URL = "https://apis.data.go.kr/B551011/KorService2/locationBasedList2"
TOURAPI_LOCATION_RADIUS = 500
MATCH_THRESHOLD = 0.62
MAX_PARALLEL_WORKERS = 5

TOURAPI_TARGET_CONFIGS = {
    "seoul": {
        "code_prefix": "11",
        "address_keyword": "서울",
        "out_of_scope_status": "out_of_scope_seoul",
        "cache_filename": "seoul_tourapi_keyword_cache.jsonl",
    },
    "daejeon": {
        "code_prefix": "30",
        "address_keyword": "대전",
        "out_of_scope_status": "out_of_scope_daejeon",
        "cache_filename": "daejeon_tourapi_keyword_cache.jsonl",
    },
    "busan": {
        "code_prefix": "26",
        "address_keyword": "부산",
        "out_of_scope_status": "out_of_scope_busan",
        "cache_filename": "busan_tourapi_keyword_cache.jsonl",
    },
    "suwon": {
        "code_prefix": "4111",
        "address_keyword": "수원",
        "out_of_scope_status": "out_of_scope_suwon",
        "cache_filename": "suwon_tourapi_keyword_cache.jsonl",
    },
    "changwon": {
        "code_prefix": "4812",
        "address_keyword": "창원",
        "out_of_scope_status": "out_of_scope_changwon",
        "cache_filename": "changwon_tourapi_keyword_cache.jsonl",
    },
    "incheon": {
        "code_prefix": "28",
        "address_keyword": "인천",
        "out_of_scope_status": "out_of_scope_incheon",
        "cache_filename": "incheon_tourapi_keyword_cache.jsonl",
    },
    "daegu": {
        "code_prefix": "27",
        "address_keyword": "대구",
        "out_of_scope_status": "out_of_scope_daegu",
        "cache_filename": "daegu_tourapi_keyword_cache.jsonl",
    },
    "gwangju": {
        "code_prefixes": ["29110", "29140", "29155", "29170", "29200"],
        "address_keywords": [
            "광주광역시",
            "광주 동구",
            "광주 서구",
            "광주 남구",
            "광주 북구",
            "광주 광산구",
        ],
        "out_of_scope_status": "out_of_scope_gwangju",
        "cache_filename": "gwangju_tourapi_keyword_cache.jsonl",
    },
}

PRIVATE_PLACE_PATTERNS = (
    r"^집$",
    r"^친구\s+친지\s+집$",
    r"^((친구|친지|친척|부모님|어머니|아버지|엄마|아빠|할머니|할머\s*니|할아버지|외할머니|외할아버지|지인|남자친구|여자친구)\s*)+(집|댁)$",
    r"^(친정|본가|시댁|처가|자택)$",
    r"^(사무실|숙소|회사)$",
    r"^(에어비앤비|airbnb)\s*숙소$",
)

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
    "X_COORD",
    "Y_COORD",
    "LEGAL_DONG_CD",
    "SGG_CD",
    "ROAD_NM_ADDR",
    "LOTNO_ADDR",
    "CONTENT_ID",
    "CONTENT_TYPE_ID",
    "TOURAPI_MATCH_STATUS",
    "TOURAPI_MATCH_SCORE",
]

VISIT_LOCATION_COLUMNS = [
    "VISIT_AREA_ID",
    "TRAVEL_ID",
    "VISIT_ORDER",
    "VISIT_AREA_NM",
    "VISIT_START_YMD",
    "ROAD_NM_ADDR",
    "LOTNO_ADDR",
    "X_COORD",
    "Y_COORD",
    "LOTNO_CD",
    "SGG_CD",
]
