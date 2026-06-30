"""
TabPFN Training Pipeline with W&B Tracking
==========================================
Trains a TabPFN regressor to predict house prices (price_vnd),
using the same preprocessing and metric suite as train_xgboost.py
for a direct apples-to-apples comparison.

Usage:
    python scripts/train_tabpfn.py --dataset optA --token YOUR_PRIORLABS_TOKEN
    python scripts/train_tabpfn.py --dataset optA_enriched --token YOUR_PRIORLABS_TOKEN

Alternatively, set TABPFN_TOKEN environment variable:
    $env:TABPFN_TOKEN = "YOUR_TOKEN"  # PowerShell
    export TABPFN_TOKEN="YOUR_TOKEN"  # bash

Get your token at: https://ux.priorlabs.ai/account
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import wandb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WANDB_PROJECT = "real-estate-valuation"

TARGET = "price_vnd"
RANDOM_STATE = 42
TEST_SIZE = 0.2

DROP_COLS = [
    "link", "title", "post_day", "street", "old_address",
    "locality", "region", "listing_id", "matched_address",
    "listing_type","area_m2"
]
LABEL_ENCODE_COLS = []


# ---------------------------------------------------------------------------
# Preprocessing (identical to train_xgboost.py for fair comparison)
# ---------------------------------------------------------------------------
def parse_locality_square(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(",", ".").astype(float)


def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def preprocess(df: pd.DataFrame) -> tuple:
    df = df.copy()

    # Split post_day to date/month/year
    if "post_day" in df.columns:
        post_day_dt = pd.to_datetime(df["post_day"])
        df["post_day_year"] = post_day_dt.dt.year
        df["post_day_month"] = post_day_dt.dt.month
        df["post_day_day"] = post_day_dt.dt.day

    if "locality_square" in df.columns:
        df["locality_square"] = parse_locality_square(df["locality_square"])

    y = df[TARGET].copy()
    df = df.drop(columns=[TARGET])

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    label_encoders = {}
    for col in LABEL_ENCODE_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

    meta = {
        "features": df.columns.tolist(),
        "n_features": len(df.columns),
        "label_encoders": label_encoders,
    }
    return df, y, meta

# ---------------------------------------------------------------------------
# Load XGBoost baseline metrics for comparison
# ---------------------------------------------------------------------------
def load_xgb_baseline(dataset_label: str) -> dict | None:
    """Try to load XGBoost metrics from W&B summary or a local cache."""
    # Hardcoded from our previous run (optA)
    baseline = {
        "optA": {"r2": 0.1473, "mape": 41.04, "mae": 30642333476.87, "rmse": 107882373891.96},
        "optB": {"r2": 0.1288, "mape": 40.91, "mae": 31337427720.05, "rmse": 109045594053.78},
    }
    base_key = dataset_label.replace("_enriched", "")
    return baseline.get(base_key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train TabPFN for house price prediction")
    parser.add_argument(
        "--dataset",
        default="cleaned",
        help="Which cleaned dataset to use. Used for naming output files.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Direct path to the csv file. If provided, overrides the default data dir path based on --dataset.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="PriorLabs API token for TabPFN license. Get it at https://ux.priorlabs.ai/account",
    )
    args = parser.parse_args()

    # Set TABPFN_TOKEN before any tabpfn imports
    import os
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    token = args.token or os.environ.get("TABPFN_TOKEN")
    if token:
        os.environ["TABPFN_TOKEN"] = token
    else:
        print(
            "\n[WARNING] No TABPFN_TOKEN set. TabPFN will attempt interactive browser auth.\n"
            "  To avoid this, pass --token YOUR_TOKEN or set the TABPFN_TOKEN env variable.\n"
            "  Get your token at: https://ux.priorlabs.ai/account\n"
        )
    dataset_label = args.dataset
    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        # Define candidate paths to search
        candidates = [
            Path(dataset_label),
            DATA_DIR / dataset_label,
            PROJECT_ROOT / "data" / "raw" / dataset_label,
        ]
        
        # If dataset_label ends with .csv, also try without the extension to find cleaned versions
        if dataset_label.endswith(".csv"):
            stem = Path(dataset_label).stem
            candidates.extend([
                DATA_DIR / f"alonhadat_features_cleaned_{stem}.csv",
                DATA_DIR / f"{stem}.csv",
                PROJECT_ROOT / "data" / "raw" / f"{stem}.csv",
            ])
        else:
            candidates.extend([
                DATA_DIR / f"alonhadat_features_cleaned_{dataset_label}.csv",
                DATA_DIR / f"{dataset_label}.csv",
                PROJECT_ROOT / "data" / "raw" / f"{dataset_label}.csv",
            ])
            
        csv_path = None
        for path in candidates:
            if path.is_file():
                csv_path = path
                dataset_label = path.stem
                break
                
        if csv_path is None:
            # Fallback to default expected path if none of the candidates exist
            stem = Path(dataset_label).stem if dataset_label.endswith(".csv") else dataset_label
            if stem == "cleaned":
                csv_path = DATA_DIR / "alonhadat_features_cleaned.csv"
            else:
                csv_path = DATA_DIR / f"alonhadat_features_cleaned_{stem}.csv"
                
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found. Tried checking the following paths:\n"
                    + "\n".join(f"  - {p}" for p in candidates)
                    + f"\nFallback path also failed: {csv_path}"
                )

    
    timestamp = datetime.now().isoformat()

    print("=" * 60)
    print(f"  TabPFN Training — Dataset: {dataset_label}")
    print(f"  Started: {timestamp}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Import TabPFN (deferred so import error is informative)
    # ------------------------------------------------------------------
    print("\n[1/6] Importing TabPFN...")
    try:
        # pyrefly: ignore [missing-import]
        from tabpfn import TabPFNRegressor
        print("  tabpfn imported successfully.")
    except ImportError:
        print("  tabpfn not installed. Run: pip install tabpfn")
        raise

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    print(f"\n[2/6] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")

    # Log direction distribution if enriched
    if "direction" in df.columns:
        dist = df["direction"].value_counts().to_dict()
        print(f"  Direction distribution: {dist}")

    # ------------------------------------------------------------------
    # 3. Preprocess
    # ------------------------------------------------------------------
    print("\n[3/6] Preprocessing...")
    X, y, meta = preprocess(df)
    feature_names = meta["features"]
    print(f"  Features ({meta['n_features']}): {feature_names}")

    # Log1p transform on target
    y_log = np.log1p(y)

    # ------------------------------------------------------------------
    # 4. Train / Test split (same seed as XGBoost for fair comparison)
    # ------------------------------------------------------------------
    print("\n[4/6] Splitting data (80/20)...")
    price_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test, y_log_train, y_log_test = train_test_split(
        X, y, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=price_bins
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # 5. Train TabPFN
    # ------------------------------------------------------------------
    print("\n[5/6] Training TabPFN...")
    model = TabPFNRegressor(random_state=RANDOM_STATE, ignore_pretraining_limits=True)

    import time
    t0 = time.time()
    model.fit(X_train, y_log_train)
    train_time = time.time() - t0
    print(f"  Training complete in {train_time:.2f}s")

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    print("\n[6/6] Evaluating...")
    y_log_pred = model.predict(X_test)
    y_pred = np.expm1(y_log_pred)
    y_pred = np.clip(y_pred, 0, None)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test.values, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_log_test, y_log_pred))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "rmse_log": rmse_log,
        "train_time_seconds": train_time,
    }

    print("\n" + "=" * 60)
    print(f"  Results — TabPFN on {dataset_label}")
    print("=" * 60)
    print(f"  RMSE:      {rmse / 1e9:.2f} Billion VND")
    print(f"  MAE:       {mae / 1e9:.2f} Billion VND")
    print(f"  R²:        {r2:.4f}")
    print(f"  MAPE:      {mape:.2f}%")
    print(f"  RMSE(log): {rmse_log:.4f}")
    print(f"  Train time:{train_time:.2f}s")
    print("=" * 60)
    # ------------------------------------------------------------------
    # 7. Log to W&B
    # ------------------------------------------------------------------
    print("\n  Logging to W&B...")
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"tabpfn-{dataset_label}",
        config={
            "model": "TabPFN",
            "dataset": dataset_label,
            "n_samples_train": X_train.shape[0],
            "n_samples_test": X_test.shape[0],
            "n_features": meta["n_features"],
            "features": meta["features"],
            "target_transform": "log1p",
            "random_state": RANDOM_STATE,
            "label_encoders": meta["label_encoders"],
        },
    )

    wandb.log(metrics)
    run.finish()

    print("Done!")


if __name__ == "__main__":
    main()
