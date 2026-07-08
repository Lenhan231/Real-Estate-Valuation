"""
Flask REST API for house price predictions.
Serves both the hybrid ensemble and segment-specific models.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path
import json
import traceback
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our advanced inference
from models.inference_advanced import HybridPredictor, build_features_for_inference

app = Flask(__name__)
CORS(app)

# Global predictor instance
predictor = None

def init_predictor():
    global predictor
    try:
        predictor = HybridPredictor(model_dir='saved_models')
        return True
    except Exception as e:
        print(f"ERROR: Failed to load predictor: {e}")
        return False

@app.before_request
def load_predictor():
    global predictor
    if predictor is None:
        if not init_predictor():
            return jsonify({'error': 'Model not available'}), 503

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if predictor is None:
        return jsonify({'status': 'unhealthy', 'reason': 'model_not_loaded'}), 503
    return jsonify({'status': 'healthy'})

@app.route('/api/models/info', methods=['GET'])
def get_model_info():
    """Get model information and metrics."""
    if predictor is None:
        return jsonify({'error': 'Model not available'}), 503

    metrics = predictor.get_metrics()
    return jsonify({
        'model_type': metrics['model_type'],
        'mape': f"{metrics['mape']:.2f}%",
        'mae': f"{metrics['mae']:.4f}B",
        'r2': f"{metrics['r2']:.4f}",
        'segments': predictor.segments,
        'features_count': len(predictor.features),
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict house price.

    Input JSON:
    {
        "area_m2": 80,
        "num_floors": 3,
        "num_bedrooms": 3,
        "locality": "phường bình thạnh",
        "property_type": "nha_mat_tien",
        "legal_status": "so_hong_so_do",
        "direction": "unknown",
        "width_m": 4,
        "length_m": 20,
        "road_width_m": 6,
        "distance_to_center_km": 5,
        ...
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Create a simple dataframe with required columns
        row = {}

        # Numeric columns
        numeric_cols = [
            'nearest_school_km', 'school_count_3km',
            'nearest_hospital_km', 'hospital_count_5km',
            'nearest_marketplace_km', 'marketplace_count_3km',
            'nearest_supermarket_km', 'supermarket_count_3km',
            'nearest_mall_km', 'mall_count_3km',
            'nearest_bus_stop_km', 'bus_stop_count_1km',
            'nearest_metro_km', 'metro_count_5km',
            'area_m2', 'distance_to_center_km',
            'num_floors', 'num_bedrooms', 'road_width_m', 'width_m', 'length_m',
            'locality_population_density','locality_square', 'lat', 'lon',
        ]

        for col in numeric_cols:
            row[col] = data.get(col, np.nan)

        # Categorical and binary
        row['locality'] = data.get('locality', 'unknown')
        row['property_type'] = data.get('property_type', 'unknown')
        row['legal_status'] = data.get('legal_status', 'unknown')
        row['direction'] = data.get('direction', 'unknown')
        row['title'] = data.get('title', '')
        row['description'] = data.get('description', '')

        # Binary features
        binary_cols = ['dining_room_bin', 'kitchen_bin', 'terrace_bin', 'car_parking_bin', 'owner_listing_bin']
        for col in binary_cols:
            row[col] = data.get(col, False)

        # Create dataframe
        df = pd.DataFrame([row])

        # Build features
        X = build_features_for_inference(df)
        X = X.reindex(columns=predictor.features, fill_value=0)

        # Apply locality encoding
        X['locality_price_median'] = predictor.metadata['locality_price_map'].get(
            row['locality'],
            predictor.metadata['locality_price_global']
        )

        # Predict
        prediction = predictor.predict(X)[0]

        # Uncertainty estimate (using MAE from metadata)
        mae = predictor.metadata['metrics_hybrid']['mae']

        return jsonify({
            'success': True,
            'prediction': {
                'price_billion_vnd': float(prediction),
                'price_million_vnd': float(prediction * 1000),
                'confidence_lower': float(max(prediction - mae, 0)),
                'confidence_upper': float(prediction + mae),
                'mae': float(mae),
                'price_per_m2_million': float(prediction * 1000 / (row['area_m2'] + 1)) if row['area_m2'] else None,
            },
            'model_info': {
                'mape': predictor.metadata['metrics_hybrid']['mape'],
                'r2': predictor.metadata['metrics_hybrid']['r2_score'],
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch predict house prices.

    Input: list of property objects (same as /api/predict)
    Output: list of predictions
    """
    try:
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({'error': 'Expected list of properties'}), 400

        results = []
        for item in data:
            # Make a request-like dict and extract prediction
            try:
                result = predict_single(item, predictor)
                results.append(result)
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e)
                })

        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def predict_single(data, pred_model):
    """Predict single property."""
    row = {}

    numeric_cols = [
        'nearest_school_km', 'school_count_3km',
        'nearest_hospital_km', 'hospital_count_5km',
        'nearest_marketplace_km', 'marketplace_count_3km',
        'nearest_supermarket_km', 'supermarket_count_3km',
        'nearest_mall_km', 'mall_count_3km',
        'nearest_bus_stop_km', 'bus_stop_count_1km',
        'nearest_metro_km', 'metro_count_5km',
        'area_m2', 'distance_to_center_km',
        'num_floors', 'num_bedrooms', 'road_width_m', 'width_m', 'length_m',
        'locality_population_density','locality_square', 'lat', 'lon',
    ]

    for col in numeric_cols:
        row[col] = data.get(col, np.nan)

    row['locality'] = data.get('locality', 'unknown')
    row['property_type'] = data.get('property_type', 'unknown')
    row['legal_status'] = data.get('legal_status', 'unknown')
    row['direction'] = data.get('direction', 'unknown')
    row['title'] = data.get('title', '')
    row['description'] = data.get('description', '')

    binary_cols = ['dining_room_bin', 'kitchen_bin', 'terrace_bin', 'car_parking_bin', 'owner_listing_bin']
    for col in binary_cols:
        row[col] = data.get(col, False)

    df = pd.DataFrame([row])
    X = build_features_for_inference(df)
    X = X.reindex(columns=pred_model.features, fill_value=0)
    X['locality_price_median'] = pred_model.metadata['locality_price_map'].get(
        row['locality'],
        pred_model.metadata['locality_price_global']
    )

    prediction = pred_model.predict(X)[0]
    mae = pred_model.metadata['metrics_hybrid']['mae']

    return {
        'success': True,
        'prediction': {
            'price_billion_vnd': float(prediction),
            'confidence_lower': float(max(prediction - mae, 0)),
            'confidence_upper': float(prediction + mae),
        }
    }

@app.route('/api/model-comparison', methods=['POST'])
def model_comparison():
    """
    Compare predictions across segments.

    Returns both ensemble and segment-specific predictions.
    """
    try:
        data = request.get_json()

        # Build single row
        row = {}
        numeric_cols = [
            'nearest_school_km', 'school_count_3km',
            'nearest_hospital_km', 'hospital_count_5km',
            'nearest_marketplace_km', 'marketplace_count_3km',
            'nearest_supermarket_km', 'supermarket_count_3km',
            'nearest_mall_km', 'mall_count_3km',
            'nearest_bus_stop_km', 'bus_stop_count_1km',
            'nearest_metro_km', 'metro_count_5km',
            'area_m2', 'distance_to_center_km',
            'num_floors', 'num_bedrooms', 'road_width_m', 'width_m', 'length_m',
            'locality_population_density','locality_square', 'lat', 'lon',
        ]

        for col in numeric_cols:
            row[col] = data.get(col, np.nan)

        row['locality'] = data.get('locality', 'unknown')
        row['property_type'] = data.get('property_type', 'unknown')
        row['legal_status'] = data.get('legal_status', 'unknown')
        row['direction'] = data.get('direction', 'unknown')
        row['title'] = data.get('title', '')
        row['description'] = data.get('description', '')

        for col in ['dining_room_bin', 'kitchen_bin', 'terrace_bin', 'car_parking_bin', 'owner_listing_bin']:
            row[col] = data.get(col, False)

        df = pd.DataFrame([row])
        X = build_features_for_inference(df)
        X = X.reindex(columns=predictor.features, fill_value=0)
        X['locality_price_median'] = predictor.metadata['locality_price_map'].get(
            row['locality'],
            predictor.metadata['locality_price_global']
        )

        # Ensemble prediction
        ensemble_pred = predictor.predict(X)[0]

        # Segment predictions
        segment_preds = {}
        for seg_lo, seg_hi in predictor.segments:
            seg_key = f'{seg_lo}_{seg_hi}'
            if seg_key in predictor.segment_models:
                model = predictor.segment_models[seg_key]
                y_log = model.predict(X)[0]
                segment_preds[f'{seg_lo}-{seg_hi}B'] = float(np.expm1(y_log))

        return jsonify({
            'success': True,
            'ensemble_prediction': float(ensemble_pred),
            'segment_predictions': segment_preds,
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

if __name__ == '__main__':
    if not init_predictor():
        print("ERROR: Could not initialize predictor. Exiting.")
        sys.exit(1)

    print("="*70)
    print("HOUSE PRICE PREDICTION API")
    print("="*70)
    print("\nEndpoints:")
    print("  GET  /health                    - Health check")
    print("  GET  /api/models/info           - Model info & metrics")
    print("  POST /api/predict               - Single prediction")
    print("  POST /api/batch-predict         - Batch predictions")
    print("  POST /api/model-comparison      - Segment comparison")
    print("\nStarting server on http://localhost:5000")
    print("="*70 + "\n")

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
