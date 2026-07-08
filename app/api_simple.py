#!/usr/bin/env python3
"""Simple Flask API for predictions."""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys, pickle, joblib, numpy as np, pandas as pd
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Load model
try:
    model_dir = Path(__file__).parent.parent / 'models' / 'saved_models'
    with open(model_dir / 'hybrid_meta.pkl', 'rb') as f:
        metadata = pickle.load(f)
    meta_learner = joblib.load(model_dir / 'ensemble_meta_learner.joblib')
    base_models = joblib.load(model_dir / 'ensemble_base_models.joblib')
    segment_models = joblib.load(model_dir / 'segment_models.joblib')
    MODEL_LOADED = True
except:
    MODEL_LOADED = False

@app.route('/health')
def health():
    return jsonify({'status': 'healthy' if MODEL_LOADED else 'unhealthy'})

@app.route('/api/info')
def info():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 503
    return jsonify({
        'model': metadata['model_type'],
        'mape': f"{metadata['metrics_hybrid']['mape']:.2f}%",
        'mae': f"{metadata['metrics_hybrid']['mae']:.4f}B",
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        data = request.get_json()

        # Build feature vector matching training features
        features_dict = {}

        # Numeric features
        numeric = ['nearest_school_km', 'school_count_3km', 'nearest_hospital_km', 'hospital_count_5km',
                   'nearest_marketplace_km', 'marketplace_count_3km', 'nearest_supermarket_km', 'supermarket_count_3km',
                   'nearest_mall_km', 'mall_count_3km', 'nearest_bus_stop_km', 'bus_stop_count_1km',
                   'nearest_metro_km', 'metro_count_5km', 'area_m2', 'distance_to_center_km',
                   'num_floors', 'num_bedrooms', 'road_width_m', 'width_m', 'length_m',
                   'locality_population_density', 'locality_square', 'lat', 'lon']

        for col in numeric:
            features_dict[col] = data.get(col, np.nan)

        # Categorical
        features_dict['locality'] = data.get('locality', 'unknown')
        features_dict['property_type'] = data.get('property_type', 'unknown')
        features_dict['legal_status'] = data.get('legal_status', 'unknown')
        features_dict['direction'] = data.get('direction', 'unknown')
        features_dict['title'] = data.get('title', '')

        # Binary
        for col in ['dining_room_bin', 'kitchen_bin', 'terrace_bin', 'car_parking_bin', 'owner_listing_bin']:
            features_dict[col] = data.get(col, False)

        # Process like training
        df = pd.DataFrame([features_dict])

        # Missing flags
        for col in ['nearest_metro_km', 'nearest_mall_km', 'nearest_supermarket_km', 'width_m', 'length_m']:
            df[f'{col}_missing'] = df[col].isna().astype(int)

        # Title flags
        title = df['title'].fillna('').str.lower()
        for flag, pattern in {
            'title_hem_xe_hoi': 'hẻm xe hơi|hxh', 'title_mat_tien': 'mặt tiền|mat tien',
            'title_biet_thu': 'biệt thự|villa', 'title_can_goc': 'căn góc|góc 2|2 mặt tiền',
            'title_thang_may': 'thang máy', 'title_ham': 'hầm',
            'title_nha_moi': 'nhà mới|mới xây|xây mới', 'title_ban_gap': 'bán gấp|ngộp|thanh lý',
            'title_kinh_doanh': 'kinh doanh|dòng tiền|cho thuê', 'title_compound': 'compound|khu biệt thự',
        }.items():
            df[flag] = title.str.contains(pattern, regex=True).astype(int)

        # Impute
        for col in numeric:
            if col.startswith('nearest_'):
                df[col] = df[col].fillna(df[col].max())

        df['width_m'] = df.apply(lambda r: r['area_m2']/r['length_m'] if pd.isna(r['width_m']) and r['length_m']>0 else r['width_m'], axis=1)
        df['length_m'] = df.apply(lambda r: r['area_m2']/r['width_m'] if pd.isna(r['length_m']) and r['width_m']>0 else r['length_m'], axis=1)

        for col in ['length_m', 'width_m']:
            df[col] = df[col].fillna(df[col].median())

        df['locality_square'] = pd.to_numeric(df['locality_square'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        df['locality_square'] = df['locality_square'].fillna(df['locality_square'].median())

        for col in ['dining_room_bin', 'kitchen_bin', 'terrace_bin', 'car_parking_bin', 'owner_listing_bin']:
            df[col] = df[col].fillna(False).astype(int)

        df = pd.get_dummies(df, columns=['property_type', 'legal_status', 'direction'], dummy_na=True, prefix=['property_type', 'legal_status', 'direction'])

        df['price_per_m2'] = 0  # Placeholder
        df['amenity_score'] = np.log1p(df.get('supermarket_count_3km', 0)) + np.log1p(df.get('school_count_3km', 0))

        # Prepare features
        X = df.reindex(columns=metadata['features'], fill_value=0)
        X['locality_price_median'] = metadata['locality_price_map'].get(features_dict['locality'], metadata['locality_price_global'])

        # Predict
        meta_feat = np.zeros((1, len(base_models)))
        for idx, (name, model) in enumerate(base_models.items()):
            meta_feat[0, idx] = model.predict(X)[0]

        y_ensemble = np.expm1(meta_learner.predict(meta_feat))[0]

        # Hybrid
        area = features_dict.get('area_m2', 80)
        price = y_ensemble  # Default to ensemble

        mae = metadata['metrics_hybrid']['mae']

        return jsonify({
            'success': True,
            'prediction': float(price),
            'confidence_lower': float(max(price - mae, 0)),
            'confidence_upper': float(price + mae),
            'unit': 'billion_vnd',
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    if not MODEL_LOADED:
        print("ERROR: Model not loaded. Run train.py first!")
        sys.exit(1)
    print("Starting API on http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
