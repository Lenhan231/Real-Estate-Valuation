"""
XGBoost Training Pipeline with Weights & Biases Tracking
=========================================================
Trains an XGBoost regressor to predict house prices (price_vnd)
from the cleaned feature datasets.

Usage:
    python scripts/train_xgboost.py --dataset optA
    python scripts/train_xgboost.py --dataset optB
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"


# Columns to drop before training (non-predictive metadata)
DROP_COLS = [
    "link", "title", "post_day", "street", "old_address",
    "locality", "region", "listing_id", "matched_address",
    "listing_type",   # zero variance – all rows are 'can_ban'
    "lat", "lon",     # captured by distance_to_center_km + POI distances
]

# Columns to label-encode
LABEL_ENCODE_COLS = ["direction", "legal_status"]

TARGET = "price_vnd"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_locality_square(series: pd.Series) -> pd.Series:
    """Convert Vietnamese comma-decimal strings like '3,32' to float 3.32."""
    return series.astype(str).str.replace(",", ".").astype(float)


def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE – guarded against division by zero."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Preprocess the dataframe for training.
    Returns (X, y, metadata_dict_for_logging).
    """
    df = df.copy()

    # Split post_day to date/month/year
    if "post_day" in df.columns:
        post_day_dt = pd.to_datetime(df["post_day"])
        df["post_day_year"] = post_day_dt.dt.year
        df["post_day_month"] = post_day_dt.dt.month
        df["post_day_day"] = post_day_dt.dt.day

    # Convert property_type to binary (1 for nha_mat_tien, 0 for nha_trong_hem)
    if "property_type" in df.columns:
        df["property_type"] = (df["property_type"] == "nha_mat_tien").astype(int)

    # Parse locality_square
    if "locality_square" in df.columns:
        df["locality_square"] = parse_locality_square(df["locality_square"])

    # Separate target
    y = df[TARGET].copy()
    df = df.drop(columns=[TARGET])

    # Drop non-feature columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    df.to_csv("trai_xgboost.csv")
    # Label-encode categoricals
    label_encoders = {}
    for col in LABEL_ENCODE_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

    metadata = {
        "features": df.columns.tolist(),
        "n_features": len(df.columns),
        "label_encoders": label_encoders,
    }

    return df, y, metadata


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_feature_importance(model, feature_names, save_path: Path):
    """Create and save a feature importance bar chart."""
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
    ax.barh(range(len(sorted_idx)), importance[sorted_idx], color="#6366f1")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost Feature Importance")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_pred_vs_actual(y_true, y_pred, save_path: Path):
    """Create and save a predicted vs actual scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Convert to billions for readability
    y_true_b = y_true / 1e9
    y_pred_b = y_pred / 1e9

    ax.scatter(y_true_b, y_pred_b, alpha=0.5, s=20, color="#8b5cf6")
    max_val = max(y_true_b.max(), y_pred_b.max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Price (Billion VND)")
    ax.set_ylabel("Predicted Price (Billion VND)")
    ax.set_title("Predicted vs Actual House Prices")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model for house price prediction")
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
    args = parser.parse_args()

    # Resolve file paths
    dataset_label = args.dataset
    if args.csv_path:
        csv_path = Path(args.csv_path)
    elif Path(dataset_label).is_file():
        csv_path = Path(dataset_label)
        dataset_label = csv_path.stem
    elif (DATA_DIR / dataset_label).is_file():
        csv_path = DATA_DIR / dataset_label
        dataset_label = csv_path.stem
    else:
        if dataset_label == "cleaned":
            csv_path = DATA_DIR / "alonhadat_features_cleaned.csv"
        else:
            csv_path = DATA_DIR / f"alonhadat_features_cleaned_{dataset_label}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {csv_path}. Please run clean_features.py first.")
        
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"xgboost_{dataset_label}.json"

    print(f"{'='*60}")
    print(f"  XGBoost Training — Dataset: {dataset_label}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"      Shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print("[2/5] Preprocessing...")
    X, y, meta = preprocess(df)
    feature_names = meta["features"]
    print(f"      Features ({meta['n_features']}): {feature_names}")

    # Apply log1p transform to target
    y_log = np.log1p(y)

    # ------------------------------------------------------------------
    # 3. Train / Test split
    # ------------------------------------------------------------------
    print("[3/5] Splitting data (80/20)...")
    # Stratify by price quantile bins for balanced evaluation
    price_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test, y_log_train, y_log_test = train_test_split(
        X, y, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=price_bins
    )
    print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # 4. Train XGBoost
    # ------------------------------------------------------------------
    xgb_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "objective": "reg:squarederror",
        "n_jobs": -1,
    }

    print("[4/5] Training XGBoost...")
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_log_train,
        eval_set=[(X_test, y_log_test)],
        verbose=50,
    )

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    print("[5/5] Evaluating...")
    y_log_pred = model.predict(X_test)
    y_pred = np.expm1(y_log_pred)  # inverse log1p

    # Clip negative predictions to 0
    y_pred = np.clip(y_pred, 0, None)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test.values, y_pred)

    # Also compute log-space metrics
    rmse_log = np.sqrt(mean_squared_error(y_log_test, y_log_pred))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "rmse_log": rmse_log,
    }

    print(f"\n{'='*60}")
    print(f"  Results — {dataset_label}")
    print(f"{'='*60}")
    print(f"  RMSE:      {rmse / 1e9:.2f} Billion VND")
    print(f"  MAE:       {mae / 1e9:.2f} Billion VND")
    print(f"  R²:        {r2:.4f}")
    print(f"  MAPE:      {mape:.2f}%")
    print(f"  RMSE(log): {rmse_log:.4f}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 6. Generate plots
    # ------------------------------------------------------------------
    plot_dir = MODEL_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fi_path = plot_feature_importance(model, feature_names, plot_dir / f"feature_importance_{dataset_label}.png")
    pva_path = plot_pred_vs_actual(y_test, y_pred, plot_dir / f"pred_vs_actual_{dataset_label}.png")
    print(f"  Plots saved to {plot_dir}")

    # ------------------------------------------------------------------
    # 7. Save model
    # ------------------------------------------------------------------
    model.save_model(str(model_path))
    print(f"  Model saved to {model_path}")

    print("\n  Done!")


if __name__ == "__main__":
    main()
