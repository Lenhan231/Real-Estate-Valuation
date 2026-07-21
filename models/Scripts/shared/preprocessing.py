import json
import numpy as np
import pandas as pd
import re

def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    df = df.copy()
    df = df.drop_duplicates(subset=['listing_id'], keep='first')
    df = df.drop('created_at', axis=1, errors='ignore')

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

        # Drop post_day_year if it has no variance (all same year)
        if df["post_day_year"].nunique() == 1:
            df = df.drop("post_day_year", axis=1)

    if "locality_square" in df.columns:
        def parse_locality_square(col: pd.Series) -> pd.Series:
            def parse_val(x):
                if pd.isna(x) or x == '':
                    return np.nan
                try:
                    # Handle European format (comma as decimal separator)
                    return float(str(x).replace(',', '.'))
                except Exception:
                    return np.nan
            return col.apply(parse_val)
        df["locality_square"] = parse_locality_square(df["locality_square"])
   
    # Fill locality features with median
    # Note: locality_square may be redundant with locality_population_density
    # (pop_density = population / locality_square). Consider dropping after SHAP analysis.
    for col in ['locality_square', 'locality_population_density']:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill road_width_m with median
    if 'road_width_m' in df.columns and df['road_width_m'].isna().any():
        df['road_width_m_missing'] = df['road_width_m'].isna().astype(int)
        df['road_width_m'] = df['road_width_m'].fillna(df['road_width_m'].median())
        
    # Fill num_bedrooms with median group by property_type and area_segment
    # Use fixed bins to avoid CV leakage (qcut on entire dataset leaks test info)
    df["area_segment"] = pd.cut(
        df["area_m2"],
        bins=[0, 30, 60, 100, 150, 200, 300, 500],
        labels=['tiny', 'small', 'med_low', 'med', 'med_high', 'large', 'xlarge']
    )
    group_cols = ["property_type", "area_segment"]

    df["num_bedrooms"] = (
        df.groupby(group_cols)["num_bedrooms"]
        .transform(lambda x: x.fillna(x.median()))
    )

    # Fill num_floors with median group by property_type and area_segment
    df["num_floors"] = (
        df.groupby(group_cols)["num_floors"]
        .transform(lambda x: x.fillna(x.median()))
    )

    if 'width_m' in df.columns and 'length_m' in df.columns:
        # Only calculate if both width and length are not null
        valid_dims = df['width_m'].notna() & df['length_m'].notna()
        df.loc[valid_dims, 'perimeter_m'] = (df.loc[valid_dims, 'width_m'] + df.loc[valid_dims, 'length_m']) * 2
        df.loc[valid_dims, 'shape_ratio'] = (df.loc[valid_dims, 'width_m'] + 0.1) / (df.loc[valid_dims, 'length_m'] + 0.1)
        # If either width or length is null, fill both with square root of area_m2 (if area_m2 is available)
        missing_dims = df['width_m'].isna() | df['length_m'].isna()

        # Fill missing perimeter_m and shape_ratio with median
        if df['perimeter_m'].isna().any():
            df['perimeter_m_missing'] = df['perimeter_m'].isna().astype(int)
            df['perimeter_m'] = df['perimeter_m'].fillna(df['perimeter_m'].median())
        if df['shape_ratio'].isna().any():
            df['shape_ratio_missing'] = df['shape_ratio'].isna().astype(int)
            df['shape_ratio'] = df['shape_ratio'].fillna(df['shape_ratio'].median())
    
    for col in ["width_m", "length_m"]:
        if col in df.columns and df[col].isna().any():
            df[f"{col}_missing"] = df[col].isna().astype(int)
            # Fill by property_type + area_segment first (more granular)
            df[col] = (
                df.groupby(group_cols)[col]
                .transform(lambda x: x.fillna(x.median()))
            )
            # Then fill remaining with property_type only
            df[col] = (
                df.groupby("property_type")[col]
                .transform(lambda x: x.fillna(x.median()))
            )
            # Finally fill any remaining with global median
            df[col] = df[col].fillna(df[col].median())
            
    # # Fill missing values for all calculated features
    # amenities = ['school_count_3km', 'hospital_count_5km', 'marketplace_count_3km', 'supermarket_count_3km', 'mall_count_3km', 'bus_stop_count_1km', 'metro_count_5km']
    # df['nearby_amenities'] = df[[c for c in amenities if c in df.columns]].sum(axis=1)

    # # Fill missing values for nearest_hospital_km and nearest_marketplace_km with median and create missing indicator
    # for col in ['nearest_hospital_km', 'nearest_marketplace_km']:
    #     if col in df.columns:
    #         if df[col].isna().sum() > 0:
    #             df[f'{col}_missing'] = df[col].isna().astype(int)
    #             df[col] = df[col].fillna(df[col].median())
    
    # # Fill distance columns with max + buffer (far away marker)
    # for col in ['nearest_metro_km', 'nearest_mall_km', 'nearest_supermarket_km', 'nearest_bus_stop_km']:
    #     if col in df.columns:
    #         df[f'{col}_missing'] = df[col].isna().astype(int)
    #         df[col] = df[col].fillna(df[col].max() + 5)

    # # Calculate additional features
    # df['center_score'] = np.exp(-df['distance_to_center_km'] / 5)
    # df['school_score'] = np.exp(-df['nearest_school_km'] / 2)
    # df['hospital_score'] = np.exp(-df['nearest_hospital_km'] / 2)
    # df['mall_score'] = np.exp(-df['nearest_mall_km'] / 3)

    # df['location_score'] = (
    #     2.0 * df['center_score'] +
    #     1.5 * df['school_score'] +
    #     1.5 * df['hospital_score'] +
    #     1.0 * df['mall_score']
    # )

    # df['amenity_score'] = (
    #     np.log1p(df['school_count_3km']) +
    #     1.5*np.log1p(df['hospital_count_5km']) +
    #     np.log1p(df['supermarket_count_3km']) +
    #     2*np.log1p(df['mall_count_3km']) +
    #     3*np.log1p(df['metro_count_5km'])
    # )

    # df['interaction_loc_amenity'] = df['location_score'] * df['amenity_score']

    # =====================================================
    # Amenity Count Features
    # =====================================================

    amenities = [
        'school_count_3km',
        'hospital_count_5km',
        'marketplace_count_3km',
        'supermarket_count_3km',
        'mall_count_3km',
        'bus_stop_count_1km',
        'metro_count_5km'
    ]

    available_amenities = [c for c in amenities if c in df.columns]

    if available_amenities:
        df['nearby_amenities'] = df[available_amenities].sum(axis=1)
        df['nearby_amenities_log'] = np.log1p(df['nearby_amenities'])


    # =====================================================
    # Distance Features
    # =====================================================

    distance_cols = [
        'distance_to_center_km',
        'nearest_school_km',
        'nearest_hospital_km',
        'nearest_marketplace_km',
        'nearest_supermarket_km',
        'nearest_mall_km',
        'nearest_bus_stop_km',
        'nearest_metro_km'
    ]

    for col in distance_cols:

        if col not in df.columns:
            continue

        df[f'{col}_missing'] = df[col].isna().astype(int)

        if df[col].isna().any():

            # Nếu NaN = không tìm thấy POI
            fill_value = df[col].max() + 5

            # Nếu toàn bộ NaN
            if np.isnan(fill_value):
                fill_value = 50

            df[col] = df[col].fillna(fill_value)


    # # =====================================================
    # # Accessibility Scores
    # # =====================================================

    # if 'distance_to_center_km' in df.columns:
    #     df['center_score'] = np.exp(-df['distance_to_center_km'] / 5)

    # if 'nearest_school_km' in df.columns:
    #     df['school_score'] = np.exp(-df['nearest_school_km'] / 2)

    # if 'nearest_hospital_km' in df.columns:
    #     df['hospital_score'] = np.exp(-df['nearest_hospital_km'] / 2)

    # if 'nearest_mall_km' in df.columns:
    #     df['mall_score'] = np.exp(-df['nearest_mall_km'] / 3)

    # if 'nearest_supermarket_km' in df.columns:
    #     df['supermarket_score'] = np.exp(-df['nearest_supermarket_km'] / 2)

    # if 'nearest_metro_km' in df.columns:
    #     df['metro_score'] = np.exp(-df['nearest_metro_km'] / 4)

    # if 'nearest_bus_stop_km' in df.columns:
    #     df['bus_score'] = np.exp(-df['nearest_bus_stop_km'] / 1)


    # # =====================================================
    # # Location Score
    # # =====================================================

    # score_cols = {
    #     'center_score': 2.0,
    #     'school_score': 1.5,
    #     'hospital_score': 1.5,
    #     'mall_score': 1.0,
    #     'supermarket_score': 0.8,
    #     'metro_score': 2.5,
    #     'bus_score': 0.5,
    # }

    # df['location_score'] = 0

    # for col, weight in score_cols.items():
    #     if col in df.columns:
    #         df['location_score'] += weight * df[col]


    # # =====================================================
    # # Amenity Score
    # # =====================================================

    # df['amenity_score'] = 0

    # count_weights = {
    #     'school_count_3km': 1.0,
    #     'hospital_count_5km': 1.5,
    #     'marketplace_count_3km': 1.0,
    #     'supermarket_count_3km': 1.0,
    #     'mall_count_3km': 2.0,
    #     'metro_count_5km': 3.0,
    #     'bus_stop_count_1km': 0.5,
    # }

    # for col, weight in count_weights.items():

    #     if col in df.columns:

    #         df['amenity_score'] += (
    #             weight *
    #             np.log1p(df[col])
    #         )


    # # =====================================================
    # # Interaction Features
    # # =====================================================

    # df['interaction_loc_amenity'] = (
    #     df['location_score'] *
    #     np.log1p(df['amenity_score'])
    # )

    # if 'nearby_amenities' in df.columns:
    #     df['interaction_center_amenity'] = (
    #         df['center_score'] *
    #         np.log1p(df['nearby_amenities'])
    #     )


    # =====================================================
    # Urbanization Index
    # =====================================================

    urban = []

    if 'locality_population_density' in df.columns:
        urban.append(
            0.4 *
            np.log1p(df['locality_population_density'])
        )

    if 'location_score' in df.columns:
        urban.append(
            0.3 *
            df['location_score']
        )

    if 'nearby_amenities' in df.columns:
        urban.append(
            0.3 *
            np.log1p(df['nearby_amenities'])
        )

    if urban:
        df['urban_index'] = sum(urban)


    # =====================================================
    # Area-normalized Features
    # =====================================================

    if 'area_m2' in df.columns:

        if 'nearby_amenities' in df.columns:
            df['amenity_density'] = (
                df['nearby_amenities'] /
                (df['area_m2'] + 1)
            )

        if 'location_score' in df.columns:
            df['location_per_area'] = (
                df['location_score'] /
                np.sqrt(df['area_m2'] + 1)
            )
    
    # Fill missing values for ALL calculated features
    for col in ['area_x_floors', 'area_x_bedrooms', 'area_per_bedroom', 'distance_vs_area', 'location_score', 'interaction_loc_amenity']:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

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

    if 'area_m2' in df.columns and 'num_floors' in df.columns:
        df['area_x_floors'] = df['area_m2'] * df['num_floors']
    if 'area_m2' in df.columns and 'num_bedrooms' in df.columns:
        df['area_x_bedrooms'] = df['area_m2'] * df['num_bedrooms']
        mask = df["num_bedrooms"] > 0

        df.loc[mask, "area_per_bedroom"] = (
            df.loc[mask, "area_m2"] /
            df.loc[mask, "num_bedrooms"]
        )
    if 'distance_to_center_km' in df.columns and 'area_m2' in df.columns:
        df['distance_vs_area'] = df['distance_to_center_km'] / (df['area_m2'] + 1)
    if 'area_m2' in df.columns:
        df['log_area'] = np.log1p(df['area_m2'])
    if 'distance_to_center_km' in df.columns:
        df['log_distance_to_center'] = np.log1p(df['distance_to_center_km'])
    if 'locality_population_density' in df.columns:
        df['log_population_density'] = np.log1p(df['locality_population_density'])
        # Fill missing log values (from missing source values)
        if df['log_population_density'].isna().any():
            df['log_population_density'] = df['log_population_density'].fillna(df['log_population_density'].median())
    
    df["frontage_ratio"] = (
        df["width_m"] /
        (df["road_width_m"] + 0.1)
    )

    df["depth_ratio"] = (
        df["length_m"] /
        (df["width_m"] + 0.1)
    )

    if 'area_m2' in df.columns and 'road_width_m' in df.columns:
        df["road_area_ratio"] = (
            df["road_width_m"] /
            np.sqrt(df["area_m2"] + 1)
        )

    drop_cols = ["id", "price_vnd", "url", "link", "title", "post_day", "description",
                 "street", "ward", "district", "locality", "region", "street_n", "locality_n",
                 "matched_address", "old_address", "lat", "lon",
                 "listing_id", "owner_listing_bin", 'area_segment']

    features_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    cat_cols = features_df.select_dtypes(include=['object', 'category']).columns
    features_df = pd.get_dummies(features_df, columns=cat_cols, dummy_na=True, drop_first=False)
    features_df = features_df.rename(columns=lambda x: re.sub(r'[\[\]{},"\' :]+', '_', str(x)))
    # Convert boolean columns to int (0/1) for model compatibility
    bool_cols = features_df.select_dtypes(include=['bool']).columns
    features_df[bool_cols] = features_df[bool_cols].astype(int)

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
