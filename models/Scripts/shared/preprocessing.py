import json
import numpy as np
import pandas as pd
import re


def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    df = df.copy()
    df = df.drop_duplicates()

    if 'price_vnd' in df.columns:
        df = df.dropna(subset=['price_vnd'])
        price_b = df['price_vnd'] / 1e9
        # ponytail: slight pruning as requested (2B to 50B) to drop extremes
        df = df[
            (price_b >= 2.0) & (price_b <= 50.0) &
            (df['area_m2'].isna() | df['area_m2'].between(15, 500) if 'area_m2' in df.columns else True)
        ]
        if 'area_m2' in df.columns:
            price_sqm = df['price_vnd'] / 1e6 / df['area_m2']
            df = df[price_sqm.isna() | ((price_sqm >= 30) & (price_sqm <= 800))]

    if "post_day" in df.columns:
        post_day_dt = pd.to_datetime(df["post_day"])
        df["post_day_year"] = post_day_dt.dt.year
        df["post_day_month"] = post_day_dt.dt.month
        df["post_day_day"] = post_day_dt.dt.day

    if "locality_square" in df.columns:
        def parse_locality_square(col: pd.Series) -> pd.Series:
            def parse_arr(x):
                if pd.isna(x):
                    return np.nan
                try:
                    return float(json.loads(x.replace("'", '"'))[0])
                except Exception:
                    return np.nan
            return col.apply(parse_arr)
        df["locality_square"] = parse_locality_square(df["locality_square"])

    if 'width_m' in df.columns and 'length_m' in df.columns:
        df['perimeter_m'] = (df['width_m'] + df['length_m']) * 2
        df['shape_ratio'] = (df['width_m'] + 0.1) / (df['length_m'] + 0.1)
    if 'area_m2' in df.columns and 'num_floors' in df.columns:
        df['area_x_floors'] = df['area_m2'] * df['num_floors']
    if 'area_m2' in df.columns and 'num_bedrooms' in df.columns:
        df['area_x_bedrooms'] = df['area_m2'] * df['num_bedrooms']
        df['area_per_bedroom'] = df['area_m2'] / (df['num_bedrooms'] + 1)
    if 'distance_to_center_km' in df.columns and 'area_m2' in df.columns:
        df['distance_vs_area'] = df['distance_to_center_km'] / (df['area_m2'] + 1)
    if 'area_m2' in df.columns:
        df['log_area'] = np.log1p(df['area_m2'])
    if 'distance_to_center_km' in df.columns:
        df['log_distance_to_center'] = np.log1p(df['distance_to_center_km'])
    if 'locality_population_density' in df.columns:
        df['log_population_density'] = np.log1p(df['locality_population_density'])

    df['location_score'] = (
        (10 / (df['distance_to_center_km'] + 1)) * 2.0 +
        (10 / (df['nearest_school_km'] + 1)) * 1.5 +
        (10 / (df['nearest_hospital_km'] + 1)) * 1.5 +
        (10 / (df['nearest_mall_km'] + 1)) * 1.0
    ) if 'distance_to_center_km' in df.columns else 0

    df['amenity_score'] = (
        df.get('school_count_3km', 0) * 1.0 +
        df.get('hospital_count_5km', 0) * 1.5 +
        df.get('supermarket_count_3km', 0) * 1.0 +
        df.get('mall_count_3km', 0) * 2.0 +
        df.get('metro_count_5km', 0) * 3.0
    )
    df['interaction_loc_amenity'] = df['location_score'] * df['amenity_score']

    amenities = ['school_count_3km', 'hospital_count_5km', 'marketplace_count_3km', 'supermarket_count_3km', 'mall_count_3km', 'bus_stop_count_1km', 'metro_count_5km']
    df['nearby_amenities'] = df[[c for c in amenities if c in df.columns]].sum(axis=1)

    text_cols = ['description', 'title']
    for col in text_cols:
        if col in df.columns:
            lower = df[col].astype(str).str.lower()
            df['is_hem_xe_hoi'] = lower.str.contains('hẻm xe hơi|hxh|ô tô|xe hơi|mặt ngõ').astype(int)
            df['is_mat_tien'] = lower.str.contains('mặt tiền|mặt phố').astype(int)
            df['is_no_hau'] = lower.str.contains('nở hậu').astype(int)
            df['has_noi_that'] = lower.str.contains('nội thất|full|đầy đủ').astype(int)
            df['is_gap'] = lower.str.contains('gấp|giảm giá|cần bán').astype(int)
            df['is_kinh_doanh'] = lower.str.contains('kinh doanh|cho thuê|thu nhập').astype(int)

    for col in ['nearest_metro_km', 'nearest_mall_km', 'nearest_supermarket_km']:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(999.0)

    for col in ['width_m', 'length_m']:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(df[col].median())

    drop_cols = ["id", "price_vnd", "url", "link", "title", "post_day", "description",
                 "street", "ward", "district", "locality", "region", "street_n", "locality_n",
                 "matched_address", "old_address", "lat", "lon",
                 "listing_id", "owner_listing_bin"]
    features_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    cat_cols = features_df.select_dtypes(include=['object', 'category']).columns
    features_df = pd.get_dummies(features_df, columns=cat_cols, dummy_na=True, drop_first=False)
    features_df = features_df.rename(columns=lambda x: re.sub(r'[\[\]{},"\' :]+', '_', str(x)))

    if features_df.columns.duplicated().any():
        cols = pd.Series(features_df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        features_df.columns = cols

    metadata = {
        "n_features": features_df.shape[1],
        "features": list(features_df.columns),
        "label_encoders": {},
    }

    return features_df, df["price_vnd"], metadata


def add_locality_features(X_train, X_test, df, train_idx, test_idx, y_train):
    """Add locality-based median price and price-per-sqm features."""
    if 'locality' not in df.columns:
        return X_train, X_test

    locality_train = df.loc[train_idx, 'locality']
    locality_test = df.loc[test_idx, 'locality']
    locality_price_map = df.loc[train_idx].groupby('locality')['price_vnd'].median()
    locality_sqm_map = df.loc[train_idx].groupby('locality').apply(lambda x: (x['price_vnd'] / (x['area_m2'] + 1)).median())
    global_median = float(y_train.median())
    global_sqm = float((df.loc[train_idx, 'price_vnd'] / (df.loc[train_idx, 'area_m2'] + 1)).median())

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train['locality_price_median'] = locality_train.map(locality_price_map).fillna(global_median).values
    X_test['locality_price_median'] = locality_test.map(locality_price_map).fillna(global_median).values
    X_train['price_per_sqm_market'] = locality_train.map(locality_sqm_map).fillna(global_sqm).values
    X_test['price_per_sqm_market'] = locality_test.map(locality_sqm_map).fillna(global_sqm).values

    return X_train, X_test
