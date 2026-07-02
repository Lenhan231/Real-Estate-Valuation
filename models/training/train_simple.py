"""Simple training script - more reliable than notebook."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline.supabase_handler import fetch_csv_from_supabase


def main():
    print("="*60)
    print("HOUSE PRICE PREDICTION - MODEL TRAINING")
    print("="*60 + "\n")

    # Load data
    print("[1/6] Loading data from Supabase...")
    df = fetch_csv_from_supabase("Raw_Features")
    df['price_billion_vnd'] = df['price_vnd'] / 1e9
    print(f"✓ Loaded {len(df)} records\n")

    # Features
    FEATURE_COLS = [
        'nearest_school_km', 'school_count_3km',
        'nearest_hospital_km', 'hospital_count_5km',
        'nearest_marketplace_km', 'marketplace_count_3km',
        'nearest_supermarket_km', 'supermarket_count_3km',
        'nearest_mall_km', 'mall_count_3km',
        'nearest_bus_stop_km', 'bus_stop_count_1km',
        'nearest_metro_km', 'metro_count_5km',
        'area_m2', 'distance_to_center_km',
        'locality_population_density'
    ]
    TARGET_COL = 'price_billion_vnd'

    # Prepare data
    print("[2/6] Preparing data...")
    df_clean = df.dropna(subset=[TARGET_COL]).copy()
    X = df_clean[FEATURE_COLS].copy()
    y = df_clean[TARGET_COL].copy()

    # Fill missing
    X = X.fillna(X.median())

    # Remove outliers
    mask = y < 100
    X = X[mask]
    y = y[mask]
    print(f"✓ Prepared {len(X)} records\n")

    # Split
    print("[3/6] Splitting data (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}\n")

    # Train models
    print("[4/6] Training models...")

    rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    print(f"Random Forest - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}\n")

    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_r2 = r2_score(y_test, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    print(f"Gradient Boosting - R²: {gb_r2:.4f}, RMSE: {gb_rmse:.4f}\n")

    # Select best
    print("[5/6] Selecting best model...")
    best_name = 'random_forest' if rf_r2 > gb_r2 else 'gradient_boosting'
    best_model = rf if rf_r2 > gb_r2 else gb
    best_r2 = max(rf_r2, gb_r2)
    print(f"✓ Best: {best_name.upper()} (R² = {best_r2:.4f})\n")

    # Save
    print("[6/6] Saving model...")
    model_dir = Path(__file__).parent.parent / "saved_models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / f"{best_name}.joblib"
    joblib.dump(best_model, model_path)
    print(f"✓ Saved: {model_path}")

    metadata = {
        'model_type': best_name,
        'features': FEATURE_COLS,
        'metrics': {
            'r2_score': float(best_r2),
            'rmse': float(np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))),
            'mae': float(mean_absolute_error(y_test, best_model.predict(X_test)))
        },
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

    meta_path = model_dir / f"{best_name}_meta.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata: {meta_path}\n")

    print("="*60)
    print("✅ Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
