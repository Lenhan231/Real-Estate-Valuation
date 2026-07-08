#!/usr/bin/env python3
"""Generate predictions on full dataset using hybrid model."""
import sys, pickle, joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

try:
    # Load model
    model_dir = Path('saved_models')
    with open(model_dir / 'hybrid_meta.pkl', 'rb') as f:
        metadata = pickle.load(f)

    meta_learner = joblib.load(model_dir / 'ensemble_meta_learner.joblib')
    base_models = joblib.load(model_dir / 'ensemble_base_models.joblib')
    segment_models = joblib.load(model_dir / 'segment_models.joblib')

    print("="*70)
    print("HYBRID MODEL INFERENCE")
    print("="*70)
    print(f"\n✓ Model: {metadata['model_type']}")
    print(f"  MAPE: {metadata['metrics_hybrid']['mape']:.2f}%")
    print(f"  MAE: {metadata['metrics_hybrid']['mae']:.4f}B")

    # Load data
    print("\n[1] Loading data...")
    df = pd.read_csv('data/raw_data.csv')
    df['price_billion_vnd'] = df['price_vnd'] / 1e9
    print(f"✓ {len(df)} records")

    # Prepare features
    print("\n[2] Engineering features...")

    NUMERIC = ['nearest_school_km', 'school_count_3km', 'nearest_hospital_km', 'hospital_count_5km',
               'nearest_marketplace_km', 'marketplace_count_3km', 'nearest_supermarket_km', 'supermarket_count_3km',
               'nearest_mall_km', 'mall_count_3km', 'nearest_bus_stop_km', 'bus_stop_count_1km',
               'nearest_metro_km', 'metro_count_5km', 'area_m2', 'distance_to_center_km',
               'num_floors', 'num_bedrooms', 'road_width_m', 'width_m', 'length_m',
               'locality_population_density', 'locality_square', 'lat', 'lon']
    BIN = ['dining_room_bin', 'kitchen_bin', 'terrace_bin', 'car_parking_bin', 'owner_listing_bin']
    CAT = ['property_type', 'legal_status', 'direction']

    for col in ['nearest_metro_km', 'nearest_mall_km', 'nearest_supermarket_km', 'width_m', 'length_m']:
        df[f'{col}_missing'] = df[col].isna().astype(int)

    title = df['title'].fillna('').str.lower()
    for flag, pattern in {
        'title_hem_xe_hoi': 'hẻm xe hơi|hxh', 'title_mat_tien': 'mặt tiền|mat tien',
        'title_biet_thu': 'biệt thự|villa', 'title_can_goc': 'căn góc|góc 2|2 mặt tiền',
        'title_thang_may': 'thang máy', 'title_ham': 'hầm',
        'title_nha_moi': 'nhà mới|mới xây|xây mới', 'title_ban_gap': 'bán gấp|ngộp|thanh lý',
        'title_kinh_doanh': 'kinh doanh|dòng tiền|cho thuê', 'title_compound': 'compound|khu biệt thự',
    }.items():
        df[flag] = title.str.contains(pattern, regex=True).astype(int)

    for col in [c for c in NUMERIC if c.startswith('nearest_')]:
        df[col] = df[col].fillna(df[col].max())

    df['width_m'] = df.apply(lambda r: r['area_m2']/r['length_m'] if pd.isna(r['width_m']) and r['length_m']>0 else r['width_m'], axis=1)
    df['length_m'] = df.apply(lambda r: r['area_m2']/r['width_m'] if pd.isna(r['length_m']) and r['width_m']>0 else r['length_m'], axis=1)

    for col in ['length_m', 'width_m', 'num_floors', 'num_bedrooms', 'road_width_m']:
        df[col] = df[col].fillna(df[col].median())

    df['locality_square'] = pd.to_numeric(df['locality_square'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    df = df.dropna(subset=['locality_population_density'])
    df['locality_square'] = df['locality_square'].fillna(df['locality_square'].median())

    for col in BIN:
        df[col] = df[col].fillna(False).astype(int)

    df = pd.get_dummies(df, columns=CAT, dummy_na=True, prefix=CAT)
    df['price_per_m2'] = df['price_billion_vnd'] / (df['area_m2'] + 1)
    df['amenity_score'] = np.log1p(df['supermarket_count_3km']) + np.log1p(df['school_count_3km'])

    X = df.reindex(columns=metadata['features'], fill_value=0)
    X['locality_price_median'] = df['locality'].map(metadata['locality_price_map']).fillna(metadata['locality_price_global'])

    print("✓ Features ready")

    # Predict
    print("\n[3] Making predictions...")

    meta_feat = np.zeros((len(X), len(base_models)))
    for idx, (name, model) in enumerate(base_models.items()):
        meta_feat[:, idx] = model.predict(X)
    y_ensemble = np.expm1(meta_learner.predict(meta_feat))

    y_pred = np.zeros(len(X))
    for seg_lo, seg_hi in metadata['segments']:
        mask = (df['price_billion_vnd'] > seg_lo) & (df['price_billion_vnd'] <= seg_hi)
        if mask.sum() == 0:
            continue
        key = f'{seg_lo}_{seg_hi}'
        if key in segment_models:
            y_log = segment_models[key].predict(X[mask])
            y_pred[mask] = np.expm1(y_log)
        else:
            y_pred[mask] = y_ensemble[mask]

    missing = y_pred == 0
    if missing.sum() > 0:
        y_pred[missing] = y_ensemble[missing]

    df['predicted_price_billion_vnd'] = y_pred

    print(f"✓ {len(df)} predictions made")

    # Evaluate
    if 'price_billion_vnd' in df.columns:
        mask = df['price_billion_vnd'] > 0
        mape = mean_absolute_percentage_error(df[mask]['price_billion_vnd'], df[mask]['predicted_price_billion_vnd']) * 100
        mae = mean_absolute_error(df[mask]['price_billion_vnd'], df[mask]['predicted_price_billion_vnd'])
        print(f"\n[4] Performance on Full Dataset:")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  MAE: {mae:.4f}B")

    # Save
    output = Path('data/predictions_latest.csv')
    df.to_csv(output, index=False)
    print(f"\n✓ Saved: {output}")

    print("\n" + "="*70)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
