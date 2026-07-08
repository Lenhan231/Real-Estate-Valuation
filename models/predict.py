"""Simple prediction script - loads trained model and makes predictions."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.supabase_handler import fetch_csv_from_supabase

NUMERIC_COLS = [
    'nearest_school_km', 'school_count_3km',
    'nearest_hospital_km', 'hospital_count_5km',
    'nearest_marketplace_km', 'marketplace_count_3km',
    'nearest_supermarket_km', 'supermarket_count_3km',
    'nearest_mall_km', 'mall_count_3km',
    'nearest_bus_stop_km', 'bus_stop_count_1km',
    'nearest_metro_km', 'metro_count_5km',
    'area_m2', 'distance_to_center_km',
    'num_floors', 'num_bedrooms', 'road_width_m', 'width_m', 'length_m',
    'locality_population_density'
]
BIN_COLS = [
    'dining_room_bin', 'kitchen_bin', 'terrace_bin',
    'car_parking_bin', 'owner_listing_bin'
]
CAT_COLS = ['property_type', 'legal_status', 'direction']
NEAREST_COLS = [c for c in NUMERIC_COLS if c.startswith('nearest_')]


def build_features(frame):
    """Fill missing + encode — same rules as training (train.ipynb)."""
    X = frame[NUMERIC_COLS].copy()

    # nearest_*: missing → column max (no amenity nearby)
    X[NEAREST_COLS] = X[NEAREST_COLS].fillna(X[NEAREST_COLS].max())

    # length/width: derive from area if one of the two is missing
    X['length_m'] = X['length_m'].fillna(
        (X['area_m2'] / X['width_m']).replace([np.inf, -np.inf], np.nan))
    X['width_m'] = X['width_m'].fillna(
        (X['area_m2'] / X['length_m']).replace([np.inf, -np.inf], np.nan))

    # Everything else (num_*, road_width_m, both length/width missing): median
    X = X.fillna(X.median())

    return pd.concat([
        X,
        frame[BIN_COLS].astype(int),
        pd.get_dummies(frame[CAT_COLS].fillna('unknown')).astype(int),
    ], axis=1)


def main():
    print("="*60)
    print("HOUSE PRICE PREDICTION - INFERENCE")
    print("="*60 + "\n")

    # Load model
    print("[1/3] Loading model...")
    model_dir = Path(__file__).parent / "saved_models"
    model_files = list(model_dir.glob("*.joblib"))

    if not model_files:
        print("❌ No trained models found!")
        print("Run the training notebook first: models/train.ipynb")
        return

    latest_model = sorted(model_files)[-1]
    model_name = latest_model.stem

    model = joblib.load(latest_model)

    meta_path = model_dir / f"{model_name}_meta.pkl"
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)

    features = metadata['features']
    print(f"✓ Loaded {model_name}")
    print(f"  R²: {metadata['metrics']['r2_score']:.4f}")
    print(f"  RMSE: {metadata['metrics']['rmse']:.4f}\n")

    # Load data
    print("[2/3] Loading data from Supabase...")
    df = fetch_csv_from_supabase("Raw_Features")
    df['price_billion_vnd'] = df['price_vnd'] / 1e9
    print(f"✓ Loaded {len(df)} records\n")

    # Prepare features (same rules as training)
    print("[3/3] Making predictions...")
    X = build_features(df).reindex(columns=features, fill_value=0)

    # Predict
    y_pred = model.predict(X)
    df['predicted_price_billion_vnd'] = y_pred

    print(f"✓ Made {len(y_pred)} predictions\n")

    # Calculate error if actual prices available
    if 'price_billion_vnd' in df.columns:
        df['error_billion_vnd'] = df['predicted_price_billion_vnd'] - df['price_billion_vnd']
        mae = np.abs(df['error_billion_vnd']).mean()
        rmse = np.sqrt((df['error_billion_vnd'] ** 2).mean())

        print("Prediction Performance:")
        print(f"  MAE: {mae:.4f} billion VND")
        print(f"  RMSE: {rmse:.4f} billion VND")
        print(f"  Mean Error: {df['error_billion_vnd'].mean():.4f} billion VND\n")

    # Sample predictions
    print("Sample Predictions (first 10):")
    print("-" * 80)
    sample_cols = ['area_m2', 'distance_to_center_km', 'predicted_price_billion_vnd']
    if 'price_billion_vnd' in df.columns:
        sample_cols.append('price_billion_vnd')
        sample_cols.append('error_billion_vnd')

    print(df[sample_cols].head(10).to_string(index=False))
    print("-" * 80 + "\n")

    # Save predictions
    print("Saving predictions...")
    pred_dir = Path(__file__).parent / "data"
    pred_dir.mkdir(parents=True, exist_ok=True)

    output_file = pred_dir / "predictions_latest.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  Records: {len(df)}")
    print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB\n")

    print("="*60)
    print("✅ Inference complete!")
    print("="*60)


if __name__ == "__main__":
    main()
