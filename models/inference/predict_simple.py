"""Simple prediction script - loads trained model and makes predictions."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline.supabase_handler import fetch_csv_from_supabase


def main():
    print("="*60)
    print("HOUSE PRICE PREDICTION - INFERENCE")
    print("="*60 + "\n")

    # Load model
    print("[1/3] Loading model...")
    model_dir = Path(__file__).parent.parent / "saved_models"
    model_files = list(model_dir.glob("*.joblib"))

    if not model_files:
        print("❌ No trained models found!")
        print("Run: python models/training/train_simple.py")
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

    # Prepare features
    print("[3/3] Making predictions...")
    X = df[features].copy()
    X = X.fillna(X.median())

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
    pred_dir = Path(__file__).parent.parent.parent / "data" / "predictions"
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
