"""Train house price prediction model from Supabase data."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.supabase_handler import fetch_csv_from_supabase


FEATURE_COLS = [
    'nearest_school_km', 'school_count_3km',
    'nearest_hospital_km', 'hospital_count_5km',
    'nearest_marketplace_km', 'marketplace_count_3km',
    'nearest_supermarket_km', 'supermarket_count_3km',
    'nearest_mall_km', 'mall_count_3km',
    'nearest_bus_stop_km', 'bus_stop_count_1km',
    'nearest_metro_km', 'metro_count_5km',
    'area_m2', 'distance_to_center_km',
    'locality_square', 'locality_population_density'
]

TARGET_COL = 'price_billion_vnd'


def load_data():
    """Load training data from Supabase."""
    print("Loading data from Supabase...")
    df = fetch_csv_from_supabase("alonhadat_features")
    print(f"Loaded {len(df)} records\n")
    return df


def prepare_data(df):
    """Clean and prepare data for training."""
    print("Preparing data...")

    # Check if target column exists
    if TARGET_COL not in df.columns:
        print(f"⚠️  Target column '{TARGET_COL}' not found in data")
        print(f"Available columns: {df.columns.tolist()}")
        print("Using 'price' as target (if available)")
        if 'price' in df.columns:
            df = df.rename(columns={'price': TARGET_COL})
        else:
            raise ValueError("No price column found")

    # Check available features
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]

    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")

    print(f"✓ Using {len(available_features)} features")

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])
    print(f"✓ Rows with price: {len(df)}")

    # Select features and target
    X = df[available_features].copy()
    y = df[TARGET_COL].copy()

    # Handle missing values in features
    X = X.fillna(X.median())

    # Remove outliers (price > 100 billion is unrealistic for this market)
    mask = y < 100
    X = X[mask]
    y = y[mask]
    print(f"✓ After removing outliers: {len(X)} samples\n")

    return X, y, available_features


def train_model(X_train, y_train, X_test, y_test):
    """Train Random Forest and Gradient Boosting models."""
    print("Training models...\n")

    models = {}

    # Random Forest
    print("1. Random Forest Regressor")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    print(f"   R² Score: {rf_r2:.4f}")
    print(f"   RMSE: {rf_rmse:.4f} billion VND")
    print(f"   MAE: {rf_mae:.4f} billion VND\n")
    models['random_forest'] = (rf_model, rf_r2, rf_rmse, rf_mae)

    # Gradient Boosting
    print("2. Gradient Boosting Regressor")
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        random_state=42,
        verbose=1
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_r2 = r2_score(y_test, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    gb_mae = mean_absolute_error(y_test, gb_pred)
    print(f"   R² Score: {gb_r2:.4f}")
    print(f"   RMSE: {gb_rmse:.4f} billion VND")
    print(f"   MAE: {gb_mae:.4f} billion VND\n")
    models['gradient_boosting'] = (gb_model, gb_r2, gb_rmse, gb_mae)

    return models


def select_best_model(models):
    """Select model with highest R² score."""
    best_name = max(models, key=lambda k: models[k][1])
    best_model, best_r2, best_rmse, best_mae = models[best_name]

    print(f"✅ Best model: {best_name.upper()}")
    print(f"   R² Score: {best_r2:.4f}")
    print(f"   RMSE: {best_rmse:.4f} billion VND\n")

    return best_name, best_model, best_r2, best_rmse, best_mae


def save_model(model, model_name, features, metrics):
    """Save trained model and metadata."""
    model_dir = Path(__file__).parent.parent / "saved_models"
    model_dir.mkdir(exist_ok=True)

    # Save model
    model_path = model_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"✓ Saved model: {model_path}")

    # Save metadata
    metadata = {
        'model_type': model_name,
        'features': features,
        'metrics': metrics,
        'n_features': len(features)
    }

    meta_path = model_dir / f"{model_name}_meta.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata: {meta_path}\n")

    return model_path, meta_path


def main():
    print("="*60)
    print("HOUSE PRICE PREDICTION - MODEL TRAINING")
    print("="*60 + "\n")

    # Load data
    df = load_data()

    # Prepare data
    X, y, features = prepare_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")

    # Train models
    models = train_model(X_train, y_train, X_test, y_test)

    # Select best
    best_name, best_model, best_r2, best_rmse, best_mae = select_best_model(models)

    # Save model
    metrics = {
        'r2_score': float(best_r2),
        'rmse': float(best_rmse),
        'mae': float(best_mae),
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    model_path, meta_path = save_model(best_model, best_name, features, metrics)

    print(f"✅ Training complete!")
    print(f"Next step: python models/inference/predict.py")


if __name__ == "__main__":
    main()
