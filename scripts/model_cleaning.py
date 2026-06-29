from __future__ import annotations

import pandas as pd

MODEL_DROP_COLS = [
    "region",
    "street",
    "title",
    "link",
    "listing_id",
    "matched_address",
    "old_address",
    "direction",
    "owner_listing_bin",
    "legal_status",
    "listing_type",
    "post_day",
]

MODEL_TEXT_COLS = [
    #"listing_type",
    "property_type",
    "locality",
    #"legal_status",
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
    #"owner_listing_bin",
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
    "post_month",
    "post_day_of_month",
    "post_weekday",
    "post_is_weekend",
    "post_age_days",
    "property_type_area_m2",
    "property_type_road_width_m",
    "property_type_width_m",
    "property_type_length_m",
    "property_type_num_floors",
    "property_type_distance_to_center_km",
    "property_type_locality_population_density",
]

REGION_REPLACEMENTS = {
    "tp. hồ chí minh": "hồ chí minh",
    "tp hồ chí minh": "hồ chí minh",
    "thành phố hồ chí minh": "hồ chí minh",
    "hcm": "hồ chí minh",
    "tphcm": "hồ chí minh",
}

#DISTANCE_FILL_VALUES = {
#    "nearest_metro_km": 5,
#    "nearest_mall_km": 3,
#    "nearest_supermarket_km": 3,
#    "nearest_marketplace_km": 3,
#    "nearest_hospital_km": 5,
#    "nearest_bus_stop_km": 1,
#}

MEDIAN_IMPUTE_COLS = [
    "length_m",
    "width_m",
    "num_bedrooms",
    "num_floors",
    "road_width_m",
    "nearest_school_km",
    "nearest_metro_km",
    "nearest_mall_km",
    "nearest_supermarket_km",
    "nearest_marketplace_km",
    "nearest_hospital_km",
    "nearest_bus_stop_km",
]

PROPERTY_TYPE_ENCODING = {
    "nha_trong_hem": 0,
    "nha_mat_tien": 1,
}
PROPERTY_TYPE_LABELS = {
    0: "nha_trong_hem",
    1: "nha_mat_tien",
}
IQR_OUTLIER_COL = "price_per_m2"
IQR_MULTIPLIER = 1.5

PROPERTY_TYPE_INTERACTION_COLS = [
    "area_m2",
    "road_width_m",
    "width_m",
    "length_m",
    "num_floors",
    "distance_to_center_km",
    "locality_population_density",
]


def build_iqr_filter_report(df: pd.DataFrame) -> tuple[pd.Series, list[dict[str, float | int | str]]]:
    keep_mask = pd.Series(True, index=df.index)
    report: list[dict[str, float | int | str]] = []

    if "property_type" not in df.columns or IQR_OUTLIER_COL not in df.columns:
        return keep_mask, report

    for property_type, group in df.groupby("property_type", dropna=False):
        values = group[IQR_OUTLIER_COL].dropna()
        rows_before = len(group)
        if values.empty:
            report.append(
                {
                    "property_type": property_type,
                    "property_type_label": PROPERTY_TYPE_LABELS.get(property_type, str(property_type)),
                    "rows_before": rows_before,
                    "rows_removed": 0,
                    "rows_after": rows_before,
                    "lower_bound": float("nan"),
                    "upper_bound": float("nan"),
                }
            )
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        group_keep_mask = group[IQR_OUTLIER_COL].between(lower_bound, upper_bound)
        keep_mask.loc[group.index] = group_keep_mask
        rows_removed = int((~group_keep_mask).sum())

        report.append(
            {
                "property_type": property_type,
                "property_type_label": PROPERTY_TYPE_LABELS.get(property_type, str(property_type)),
                "rows_before": rows_before,
                "rows_removed": rows_removed,
                "rows_after": rows_before - rows_removed,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }
        )

    return keep_mask, report


def clean_for_modeling_with_report(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, float | int | str]]]:
    """Clean the engineered feature dataset for modeling.

    This is intentionally separate from pipeline/transformation/cleaning.py,
    which is team-owned ETL code. Keep notebook-driven modeling decisions here.
    """

    df = df.copy()

    if "post_day" in df.columns:
        post_day = pd.to_datetime(df["post_day"], errors="coerce")
        latest_post_day = post_day.max()
        df["post_month"] = post_day.dt.month
        df["post_day_of_month"] = post_day.dt.day
        df["post_weekday"] = post_day.dt.weekday
        df["post_is_weekend"] = post_day.dt.weekday.isin([5, 6]).astype("Int64")
        df["post_age_days"] = (latest_post_day - post_day).dt.days if pd.notna(latest_post_day) else pd.NA

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

    if "property_type" in df.columns:
        unknown_property_types = sorted(set(df["property_type"].dropna()) - set(PROPERTY_TYPE_ENCODING))
        if unknown_property_types:
            raise ValueError(f"Unknown property_type value(s): {unknown_property_types}")
        df["property_type"] = df["property_type"].map(PROPERTY_TYPE_ENCODING).astype("Int64")

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

    iqr_report: list[dict[str, float | int | str]] = []
    if {"property_type", IQR_OUTLIER_COL}.issubset(df.columns):
        iqr_keep_mask, iqr_report = build_iqr_filter_report(df)
        df = df[iqr_keep_mask]

    dedupe_cols = ["title", "street", "locality", "price_vnd", "area_m2"]
    dedupe_cols = [col for col in dedupe_cols if col in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols, keep="first")

    #fill_values = {col: value for col, value in DISTANCE_FILL_VALUES.items() if col in df.columns}
    #if fill_values:
    #    df = df.fillna(fill_values)

    for col in MEDIAN_IMPUTE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "property_type" in df.columns:
        for col in PROPERTY_TYPE_INTERACTION_COLS:
            if col in df.columns:
                df[f"property_type_{col}"] = df["property_type"].astype(float) * df[col]

    df = df.dropna(axis=1, how="all")

    return df.reset_index(drop=True), iqr_report


def clean_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    cleaned, _ = clean_for_modeling_with_report(df)
    return cleaned
