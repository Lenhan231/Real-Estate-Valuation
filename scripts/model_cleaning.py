from __future__ import annotations

import pandas as pd

MODEL_DROP_COLS = ["link", "listing_id", "matched_address", "old_address"]

MODEL_TEXT_COLS = [
    "title",
    "street",
    "locality",
    "region",
    "direction",
    "listing_type",
    "property_type",
    "legal_status",
]

MODEL_NUMERIC_COLS = [
    "num_floors",
    "num_bedrooms",
    "road_width_m",
    "width_m",
    "length_m",
    "price_vnd",
    "area_m2",
    "dining_room_bin",
    "kitchen_bin",
    "terrace_bin",
    "car_parking_bin",
    "owner_listing_bin",
    "locality_square",
    "locality_population_density",
    "lat",
    "lon",
    "distance_to_center_km",
    "nearest_school_km",
    "school_count_3km",
    "nearest_hospital_km",
    "hospital_count_5km",
    "nearest_marketplace_km",
    "marketplace_count_3km",
    "nearest_supermarket_km",
    "supermarket_count_3km",
    "nearest_mall_km",
    "mall_count_3km",
    "nearest_bus_stop_km",
    "bus_stop_count_1km",
    "nearest_metro_km",
    "metro_count_5km",
]

REGION_REPLACEMENTS = {
    "tp. hồ chí minh": "hồ chí minh",
    "tp hồ chí minh": "hồ chí minh",
    "thành phố hồ chí minh": "hồ chí minh",
    "hcm": "hồ chí minh",
    "tphcm": "hồ chí minh",
}

DISTANCE_FILL_VALUES = {
    "nearest_metro_km": 5,
    "nearest_mall_km": 3,
    "nearest_supermarket_km": 3,
    "nearest_marketplace_km": 3,
    "nearest_hospital_km": 5,
    "nearest_bus_stop_km": 1,
}

MEDIAN_IMPUTE_COLS = [
    "length_m",
    "width_m",
    "num_bedrooms",
    "num_floors",
    "road_width_m",
]


def clean_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the engineered feature dataset for modeling.

    This is intentionally separate from pipeline/transformation/cleaning.py,
    which is team-owned ETL code. Keep notebook-driven modeling decisions here.
    """

    df = df.copy()
    df = df.drop(columns=MODEL_DROP_COLS, errors="ignore")

    for col in MODEL_TEXT_COLS:
        if col not in df.columns:
            continue
        df[col] = (
            df[col]
            .astype("string")
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    for col in ["region", "locality"]:
        if col in df.columns:
            df[col] = df[col].replace(REGION_REPLACEMENTS)

    for col in MODEL_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = [col for col in ["price_vnd", "area_m2"] if col in df.columns]
    if required_cols:
        df = df.dropna(subset=required_cols)
    if "price_vnd" in df.columns:
        df = df[df["price_vnd"] > 0]
    if "area_m2" in df.columns:
        df = df[df["area_m2"] > 0]

    if {"price_vnd", "area_m2"}.issubset(df.columns):
        df["price_per_m2"] = df["price_vnd"] / df["area_m2"]
        lower = df["price_per_m2"].quantile(0.01)
        upper = df["price_per_m2"].quantile(0.99)
        df = df[df["price_per_m2"].between(lower, upper)]

    dedupe_cols = ["title", "street", "locality", "price_vnd", "area_m2"]
    dedupe_cols = [col for col in dedupe_cols if col in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols, keep="first")

    fill_values = {col: value for col, value in DISTANCE_FILL_VALUES.items() if col in df.columns}
    if fill_values:
        df = df.fillna(fill_values)

    for col in MEDIAN_IMPUTE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df = df.dropna(axis=1, how="all")

    return df.reset_index(drop=True)
