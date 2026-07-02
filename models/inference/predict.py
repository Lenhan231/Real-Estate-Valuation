"""Make house price predictions using trained model."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.supabase_handler import fetch_csv_from_supabase


def load_model_and_metadata():
    """Load best trained model and its metadata."""
    model_dir = Path(__file__).parent.parent / "saved_models"

    # Find latest model
    model_files = list(model_dir.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError("No trained models found. Run: python models/training/train.py")

    latest_model = sorted(model_files)[-1]
    model_name = latest_model.stem

    print(f"Loading model: {model_name}")
    model = joblib.load(latest_model)

    # Load metadata
    meta_path = model_dir / f"{model_name}_meta.pkl"
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)

    print(f"Model type: {metadata['model_type']}")
    print(f"Features: {len(metadata['features'])}")
    print(f"R² Score: {metadata['metrics']['r2_score']:.4f}\n")

    return model, metadata


def prepare_features(df, feature_cols):
    """Prepare features for prediction."""
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    return X


def predict_prices(model, X, df):
    """Predict house prices."""
    print("Making predictions...\n")

    y_pred = model.predict(X)

    # Add predictions to dataframe
    df_result = df.copy()
    df_result['predicted_price_billion_vnd'] = y_pred

    # Calculate error if actual prices exist
    if 'price' in df_result.columns or 'price_billion_vnd' in df_result.columns:
        price_col = 'price_billion_vnd' if 'price_billion_vnd' in df_result.columns else 'price'
        df_result['actual_price_billion_vnd'] = df_result[price_col]
        df_result['error_billion_vnd'] = df_result['predicted_price_billion_vnd'] - df_result['actual_price_billion_vnd']
        df_result['error_pct'] = (df_result['error_billion_vnd'] / df_result['actual_price_billion_vnd']) * 100

        # Statistics
        mae = np.abs(df_result['error_billion_vnd']).mean()
        print(f"Prediction Statistics:")
        print(f"  Mean Absolute Error: {mae:.4f} billion VND")
        print(f"  Median Error: {np.median(df_result['error_billion_vnd']):.4f} billion VND")
        print(f"  Std Dev: {df_result['error_billion_vnd'].std():.4f} billion VND\n")

    return df_result


def save_predictions(df_result, output_file):
    """Save predictions to CSV."""
    output_path = Path(__file__).parent.parent.parent / "data" / "predictions" / output_file
    output_path.parent.mkdir(exist_ok=True)

    df_result.to_csv(output_path, index=False)
    print(f"✓ Predictions saved: {output_path}")
    print(f"  Records: {len(df_result)}\n")

    return output_path


def display_sample_predictions(df_result, n=10):
    """Display sample predictions."""
    print(f"Sample Predictions (first {n} records):")
    print("="*80)

    cols_to_show = ['area_m2', 'predicted_price_billion_vnd']
    if 'actual_price_billion_vnd' in df_result.columns:
        cols_to_show.append('actual_price_billion_vnd')
        cols_to_show.append('error_pct')

    sample = df_result[cols_to_show].head(n)
    print(sample.to_string(index=False))
    print("="*80 + "\n")


def main():
    print("="*60)
    print("HOUSE PRICE PREDICTION - INFERENCE")
    print("="*60 + "\n")

    # Load model
    model, metadata = load_model_and_metadata()
    features = metadata['features']

    # Load data from Supabase
    print("Loading data from Supabase...")
    df = fetch_csv_from_supabase("alonhadat_features")
    print(f"Loaded {len(df)} records\n")

    # Prepare features
    X = prepare_features(df, features)

    # Make predictions
    df_result = predict_prices(model, X, df)

    # Display samples
    display_sample_predictions(df_result, n=10)

    # Save predictions
    output_file = "predictions_latest.csv"
    save_predictions(df_result, output_file)

    print(f"✅ Prediction complete!")


if __name__ == "__main__":
    main()
