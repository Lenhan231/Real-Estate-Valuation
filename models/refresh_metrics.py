"""Recompute held-out test metrics for saved models and persist MAPE in metadata.

Run this after training to refresh metrics without reopening the notebook.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "saved_models"
READY_DATA = ROOT / "data" / "model_ready_data.csv"
TARGET_COL = "price_billion_vnd"
RANDOM_STATE = 42


def _inverse_transform(predictions: np.ndarray, metadata: dict) -> np.ndarray:
    if metadata.get("target_transform") == "log1p":
        return np.expm1(predictions)
    return predictions


def refresh_model_metrics(model_path: Path) -> dict:
    metadata_path = MODEL_DIR / f"{model_path.stem}_meta.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with open(metadata_path, "rb") as file:
        metadata = pickle.load(file)

    model = joblib.load(model_path)
    data = pd.read_csv(READY_DATA)
    features = metadata["features"]

    missing_features = [col for col in features if col not in data.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in {READY_DATA}: {missing_features[:10]}")

    if TARGET_COL not in data.columns:
        raise ValueError(f"Missing target column {TARGET_COL} in {READY_DATA}")

    data = data.dropna(subset=[TARGET_COL]).copy()
    train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=RANDOM_STATE)

    X_test = data.loc[test_idx, features]
    y_test = data.loc[test_idx, TARGET_COL].astype(float)
    y_pred = _inverse_transform(model.predict(X_test), metadata)

    mape = float(mean_absolute_percentage_error(y_test, y_pred) * 100)
    metrics = metadata.get("metrics", {}).copy()
    metrics["mape"] = mape
    metadata["metrics"] = metrics

    with open(metadata_path, "wb") as file:
        pickle.dump(metadata, file)

    return {
        "model": model_path.stem,
        "mape": mape,
        "test_size": len(test_idx),
    }


def main() -> None:
    if not READY_DATA.exists():
        raise FileNotFoundError(f"Missing training data file: {READY_DATA}")

    model_files = sorted(MODEL_DIR.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {MODEL_DIR}")

    rows = []
    for model_path in model_files:
        result = refresh_model_metrics(model_path)
        rows.append(result)
        print(f"{result['model']}: MAPE={result['mape']:.2f}% on {result['test_size']} test rows")

    summary = pd.DataFrame(rows)
    summary_path = ROOT / "data" / "model_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved metrics summary to {summary_path}")


if __name__ == "__main__":
    main()