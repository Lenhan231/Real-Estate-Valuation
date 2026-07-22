import numpy as np
import pandas as pd
import re


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE, ignoring zero values."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Comprehensive preprocessing pipeline with feature engineering.

    Steps:
    1. Outlier filtering (price, area, price per sqm)
    2. Temporal feature extraction
    3. Numeric imputation (hierarchical filling)
    4. Distance and amenity feature engineering
    5. Text-based feature extraction
    6. Ratio and interaction features
    7. One-hot encoding + boolean conversion
    """
    df = df.copy()
    df = df.drop_duplicates(subset=['listing_id'], keep='first')
    df = df.drop('created_at', axis=1, errors='ignore')

    # ===== OUTLIER FILTERING =====
    if 'price_vnd' in df.columns:
        df = df.dropna(subset=['price_vnd'])
        price_b = df['price_vnd'] / 1e9
        df = df[
            (price_b >= 2.0) & (price_b <= 50.0) &
            (df['area_m2'].isna() | df['area_m2'].between(15, 500) if 'area_m2' in df.columns else True)
        ]
        if 'area_m2' in df.columns:
            price_sqm = df['price_vnd'] / 1e6 / df['area_m2']
            df = df[price_sqm.isna() | ((price_sqm >= 30) & (price_sqm <= 800))]

    # ===== TEMPORAL FEATURES =====
    if "post_day" in df.columns:
        post_day_dt = pd.to_datetime(df["post_day"])
        df["post_day_year"] = post_day_dt.dt.year
        df["post_day_month"] = post_day_dt.dt.month
        df["post_day_day"] = post_day_dt.dt.day

        if df["post_day_year"].nunique() == 1:
            df = df.drop("post_day_year", axis=1)

    # ===== LOCALITY FEATURES =====
    if "locality_square" in df.columns:
        def parse_locality_square(col: pd.Series) -> pd.Series:
            def parse_val(x):
                if pd.isna(x) or x == '':
                    return np.nan
                try:
                    return float(str(x).replace(',', '.'))
                except Exception:
                    return np.nan
            return col.apply(parse_val)
        df["locality_square"] = parse_locality_square(df["locality_square"])

    for col in ['locality_square', 'locality_population_density']:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # ===== NUMERIC IMPUTATION WITH HIERARCHICAL FILLING =====
    # Road width
    if 'road_width_m' in df.columns and df['road_width_m'].isna().any():
        df['road_width_m_missing'] = df['road_width_m'].isna().astype(int)
        df['road_width_m'] = df['road_width_m'].fillna(df['road_width_m'].median())

    # Create area segments for grouped imputation (fixed bins to prevent CV leakage)
    df["area_segment"] = pd.cut(
        df["area_m2"],
        bins=[0, 30, 60, 100, 150, 200, 300, 500],
        labels=['tiny', 'small', 'med_low', 'med', 'med_high', 'large', 'xlarge']
    )
    group_cols = ["property_type", "area_segment"]

    # Bedrooms & floors: fill by property_type + area_segment (most specific)
    df["num_bedrooms"] = (
        df.groupby(group_cols)["num_bedrooms"]
        .transform(lambda x: x.fillna(x.median()))
    )
    df["num_floors"] = (
        df.groupby(group_cols)["num_floors"]
        .transform(lambda x: x.fillna(x.median()))
    )

    # ===== DIMENSION FEATURES (width, length, perimeter, ratio) =====
    if 'width_m' in df.columns and 'length_m' in df.columns:
        valid_dims = df['width_m'].notna() & df['length_m'].notna()
        df.loc[valid_dims, 'perimeter_m'] = (df.loc[valid_dims, 'width_m'] + df.loc[valid_dims, 'length_m']) * 2
        df.loc[valid_dims, 'shape_ratio'] = (df.loc[valid_dims, 'width_m'] + 0.1) / (df.loc[valid_dims, 'length_m'] + 0.1)

        if df['perimeter_m'].isna().any():
            df['perimeter_m_missing'] = df['perimeter_m'].isna().astype(int)
            df['perimeter_m'] = df['perimeter_m'].fillna(df['perimeter_m'].median())
        if df['shape_ratio'].isna().any():
            df['shape_ratio_missing'] = df['shape_ratio'].isna().astype(int)
            df['shape_ratio'] = df['shape_ratio'].fillna(df['shape_ratio'].median())

    # Width & length: 3-tier hierarchy (property_type + area_segment → property_type → global)
    for col in ["width_m", "length_m"]:
        if col in df.columns and df[col].isna().any():
            df[f"{col}_missing"] = df[col].isna().astype(int)
            df[col] = df.groupby(group_cols)[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df.groupby("property_type")[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

    # ===== AMENITY FEATURES =====
    amenities = [
        'school_count_3km', 'hospital_count_5km', 'marketplace_count_3km',
        'supermarket_count_3km', 'mall_count_3km', 'bus_stop_count_1km', 'metro_count_5km'
    ]
    available_amenities = [c for c in amenities if c in df.columns]

    if available_amenities:
        df['nearby_amenities'] = df[available_amenities].sum(axis=1)
        df['nearby_amenities_log'] = np.log1p(df['nearby_amenities'])

    # ===== DISTANCE FEATURES =====
    distance_cols = [
        'distance_to_center_km', 'nearest_school_km', 'nearest_hospital_km',
        'nearest_marketplace_km', 'nearest_supermarket_km', 'nearest_mall_km',
        'nearest_bus_stop_km', 'nearest_metro_km'
    ]

    for col in distance_cols:
        if col not in df.columns:
            continue
        df[f'{col}_missing'] = df[col].isna().astype(int)
        if df[col].isna().any():
            fill_value = df[col].max() + 5
            if np.isnan(fill_value):
                fill_value = 50
            df[col] = df[col].fillna(fill_value)


    # ===== AREA-NORMALIZED FEATURES =====
    if 'area_m2' in df.columns and 'nearby_amenities' in df.columns:
        df['amenity_density'] = df['nearby_amenities'] / (df['area_m2'] + 1)

    # ===== TEXT-BASED FEATURES =====
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

    # ===== INTERACTION & RATIO FEATURES =====
    if 'area_m2' in df.columns and 'num_floors' in df.columns:
        df['area_x_floors'] = df['area_m2'] * df['num_floors']

    if 'area_m2' in df.columns and 'num_bedrooms' in df.columns:
        df['area_x_bedrooms'] = df['area_m2'] * df['num_bedrooms']
        mask = df["num_bedrooms"] > 0
        df.loc[mask, "area_per_bedroom"] = df.loc[mask, "area_m2"] / df.loc[mask, "num_bedrooms"]

    if 'distance_to_center_km' in df.columns and 'area_m2' in df.columns:
        df['distance_vs_area'] = df['distance_to_center_km'] / (df['area_m2'] + 1)

    if 'area_m2' in df.columns:
        df['log_area'] = np.log1p(df['area_m2'])

    if 'distance_to_center_km' in df.columns:
        df['log_distance_to_center'] = np.log1p(df['distance_to_center_km'])

    if 'locality_population_density' in df.columns:
        df['log_population_density'] = np.log1p(df['locality_population_density'])
        if df['log_population_density'].isna().any():
            df['log_population_density'] = df['log_population_density'].fillna(df['log_population_density'].median())

    # Ratio features
    if 'width_m' in df.columns and 'road_width_m' in df.columns:
        df["frontage_ratio"] = df["width_m"] / (df["road_width_m"] + 0.1)

    if 'length_m' in df.columns and 'width_m' in df.columns:
        df["depth_ratio"] = df["length_m"] / (df["width_m"] + 0.1)

    if 'area_m2' in df.columns and 'road_width_m' in df.columns:
        df["road_area_ratio"] = df["road_width_m"] / np.sqrt(df["area_m2"] + 1)

    # ===== POLYNOMIAL FEATURES (v2.5+) =====
    # Capture non-linear relationships
    if 'area_m2' in df.columns:
        df['area_m2_squared'] = df['area_m2'] ** 2
        df['area_m2_sqrt'] = np.sqrt(df['area_m2'] + 0.1)

    if 'distance_to_center_km' in df.columns:
        df['distance_squared'] = df['distance_to_center_km'] ** 2

    if 'road_width_m' in df.columns:
        df['road_width_squared'] = df['road_width_m'] ** 2

    if 'num_bedrooms' in df.columns:
        df['bedrooms_squared'] = df['num_bedrooms'] ** 2

    if 'num_floors' in df.columns:
        df['floors_squared'] = df['num_floors'] ** 2

    # ===== ADDITIONAL INTERACTION FEATURES (v2.5+) =====
    # More cross-domain interactions
    if 'area_m2' in df.columns and 'distance_to_center_km' in df.columns:
        df['area_x_distance'] = df['area_m2'] * df['distance_to_center_km']
        df['area_per_distance'] = df['area_m2'] / (df['distance_to_center_km'] + 0.1)

    if 'num_bedrooms' in df.columns and 'distance_to_center_km' in df.columns:
        df['bedrooms_x_distance'] = df['num_bedrooms'] * df['distance_to_center_km']

    if 'num_floors' in df.columns and 'distance_to_center_km' in df.columns:
        df['floors_x_distance'] = df['num_floors'] * df['distance_to_center_km']

    if 'area_m2' in df.columns and 'road_width_m' in df.columns:
        df['area_x_road_width'] = df['area_m2'] * df['road_width_m']

    if 'width_m' in df.columns and 'length_m' in df.columns:
        df['width_x_length'] = df['width_m'] * df['length_m']  # Alternative to area

    # ===== LOCALITY-BASED INTERACTIONS (v2.5+) =====
    # Combine spatial + structural features
    if 'locality_population_density' in df.columns and 'area_m2' in df.columns:
        df['density_x_area'] = df['locality_population_density'] * df['area_m2']

    if 'locality_square' in df.columns and 'area_m2' in df.columns:
        df['locality_sq_x_area'] = df['locality_square'] * df['area_m2']


    # Fill any remaining NaN in calculated features
    calc_cols = [
        'area_x_floors', 'area_x_bedrooms', 'area_per_bedroom', 'distance_vs_area',
        'area_m2_squared', 'area_m2_sqrt', 'distance_squared', 'road_width_squared',
        'bedrooms_squared', 'floors_squared', 'area_x_distance', 'area_per_distance',
        'bedrooms_x_distance', 'floors_x_distance', 'area_x_road_width', 'width_x_length',
        'density_x_area', 'locality_sq_x_area'
    ]
    for col in calc_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # ===== FEATURE SELECTION & ENCODING =====
    drop_cols = [
        "id", "price_vnd", "url", "link", "title", "post_day", "description",
        "street", "ward", "district", "locality", "region", "street_n", "locality_n",
        "matched_address", "old_address", "lat", "lon",
        "listing_id", "owner_listing_bin", 'area_segment'
    ]

    features_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # One-hot encode categorical features
    cat_cols = features_df.select_dtypes(include=['object', 'category']).columns
    features_df = pd.get_dummies(features_df, columns=cat_cols, dummy_na=True, drop_first=False)
    features_df = features_df.rename(columns=lambda x: re.sub(r'[\[\]{},"\' :]+', '_', str(x)))

    # Convert boolean to int (0/1) for model compatibility
    bool_cols = features_df.select_dtypes(include=['bool']).columns
    features_df[bool_cols] = features_df[bool_cols].astype(int)

    # Drop low-impact features only (identified via feature importance analysis)
    # v2.6 (PRODUCTION): Keep all + polynomial features, drop only early low-impact
    # Rationale: Polynomial features (area², distance²) capture non-linearity → better MAPE
    # Result: 64 base + 14 polynomial/interaction = ~78 features
    # Performance: 13.10% MAPE, 0.9200 R² (BEST)
    # Note: v2.7 test removed 24 "low-impact" features → MAPE worsened to 13.20%
    #       Decision: Those features provide subtle signal in ensemble; keep all 78

    low_impact_features = [
        'kitchen_bin_False', 'dining_room_bin_True', 'perimeter_m_missing',
        'direction_dong', 'direction_tay_bac', 'nearest_mall_km_missing',
        'length_m_missing', 'kitchen_bin_True', 'direction_tay_nam',
        'post_day_year', 'direction_bac', 'direction_tay',
        'nearest_hospital_km_missing', 'terrace_bin_nan', 'kitchen_bin_nan',
        'dining_room_bin_nan', 'property_type_nan', 'legal_status_nan',
        'legal_status_giấy_phép_xd', 'distance_to_center_km_missing',
        'nearest_bus_stop_km_missing', 'nearest_school_km_missing',
        'listing_type_nan', 'listing_type_can_ban', 'direction_nan',
        'nearest_marketplace_km_missing', 'nearest_supermarket_km_missing',
        'car_parking_bin_nan'
    ]
    features_df = features_df.drop(columns=[c for c in low_impact_features if c in features_df.columns])

    # Handle duplicate column names
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

    # Price by locality
    locality_price_map = df.loc[train_idx].groupby('locality')['price_vnd'].median()

    # Price per sqm by locality (avoid groupby.apply warning with explicit calculation)
    train_data = df.loc[train_idx].copy()
    train_data['price_per_sqm'] = train_data['price_vnd'] / (train_data['area_m2'] + 1)
    locality_sqm_map = train_data.groupby('locality')['price_per_sqm'].median()

    global_median = float(y_train.median())
    global_sqm = float((df.loc[train_idx, 'price_vnd'] / (df.loc[train_idx, 'area_m2'] + 1)).median())

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train['locality_price_median'] = locality_train.map(locality_price_map).fillna(global_median).values
    X_test['locality_price_median'] = locality_test.map(locality_price_map).fillna(global_median).values
    X_train['price_per_sqm_market'] = locality_train.map(locality_sqm_map).fillna(global_sqm).values
    X_test['price_per_sqm_market'] = locality_test.map(locality_sqm_map).fillna(global_sqm).values

    return X_train, X_test
