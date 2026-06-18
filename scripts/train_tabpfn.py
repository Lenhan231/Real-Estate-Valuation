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
import json
import logging
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
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WANDB_PROJECT = "real-estate-valuation"
TARGET = "price_vnd"
RANDOM_STATE = 42
TEST_SIZE = 0.2

DROP_COLS = [
    "link", "title", "post_day", "street", "old_address",
    "locality", "region", "listing_id", "matched_address",
    "listing_type",
    "lat", "lon",
]
LABEL_ENCODE_COLS = ["direction", "legal_status"]


# ---------------------------------------------------------------------------
# Logging setup (file + console)
# ---------------------------------------------------------------------------
def setup_logger(dataset_label: str) -> logging.Logger:
    log_file = REPORTS_DIR / f"train_tabpfn_{dataset_label}.log"
    logger = logging.getLogger(f"tabpfn_{dataset_label}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_file}")
    return logger, log_file


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

    # Convert property_type to binary (1 for nha_mat_tien, 0 for nha_trong_hem)
    if "property_type" in df.columns:
        df["property_type"] = (df["property_type"] == "nha_mat_tien").astype(int)

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


# # ---------------------------------------------------------------------------
# # Plots
# # ---------------------------------------------------------------------------
# def plot_pred_vs_actual(y_true, y_pred, save_path: Path, title: str):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     y_true_b = y_true / 1e9
#     y_pred_b = y_pred / 1e9
#     ax.scatter(y_true_b, y_pred_b, alpha=0.5, s=20, color="#10b981")
#     max_val = max(y_true_b.max(), y_pred_b.max())
#     ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Perfect prediction")
#     ax.set_xlabel("Actual Price (Billion VND)")
#     ax.set_ylabel("Predicted Price (Billion VND)")
#     ax.set_title(title)
#     ax.legend()
#     plt.tight_layout()
#     fig.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     return save_path


# def plot_feature_importance(model, X_test, y_test, feature_names, save_path: Path, title: str):
#     """Create and save a feature importance bar chart using permutation importance."""
#     from sklearn.inspection import permutation_importance
#     result = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=42, n_jobs=-1)
#     importance = result.importances_mean
#     sorted_idx = np.argsort(importance)

#     fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
#     ax.barh(range(len(sorted_idx)), importance[sorted_idx], color="#10b981")
#     ax.set_yticks(range(len(sorted_idx)))
#     ax.set_yticklabels([feature_names[i] for i in sorted_idx])
#     ax.set_xlabel("Permutation Feature Importance")
#     ax.set_title(title)
#     plt.tight_layout()
#     fig.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     return save_path


# def plot_model_comparison(xgb_metrics: dict, tabpfn_metrics: dict, save_path: Path):
#     """Side-by-side bar chart comparing XGBoost vs TabPFN metrics."""
#     labels = ["R²", "MAPE (%)", "MAE (10B VND)"]
#     xgb_vals = [
#         xgb_metrics.get("r2", 0),
#         xgb_metrics.get("mape", 0),
#         xgb_metrics.get("mae", 0) / 1e10,
#     ]
#     tab_vals = [
#         tabpfn_metrics.get("r2", 0),
#         tabpfn_metrics.get("mape", 0),
#         tabpfn_metrics.get("mae", 0) / 1e10,
#     ]

#     x = np.arange(len(labels))
#     width = 0.35

#     fig, ax = plt.subplots(figsize=(10, 6))
#     bars1 = ax.bar(x - width / 2, xgb_vals, width, label="XGBoost", color="#6366f1", alpha=0.85)
#     bars2 = ax.bar(x + width / 2, tab_vals, width, label="TabPFN", color="#10b981", alpha=0.85)

#     ax.set_title("Model Comparison: XGBoost vs TabPFN")
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.legend()
#     ax.bar_label(bars1, fmt="%.3f", padding=3)
#     ax.bar_label(bars2, fmt="%.3f", padding=3)
#     plt.tight_layout()
#     fig.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     return save_path


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
    elif Path(dataset_label).is_file():
        csv_path = Path(dataset_label)
        dataset_label = csv_path.stem
    else:
        if dataset_label == "cleaned":
            csv_path = DATA_DIR / "alonhadat_features_cleaned.csv"
        else:
            csv_path = DATA_DIR / f"alonhadat_features_cleaned_{dataset_label}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {csv_path}. Please run clean_features.py first.")

    logger, log_file = setup_logger(dataset_label)
    timestamp = datetime.now().isoformat()

    logger.info("=" * 60)
    logger.info(f"  TabPFN Training — Dataset: {dataset_label}")
    logger.info(f"  Started: {timestamp}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Import TabPFN (deferred so import error is informative)
    # ------------------------------------------------------------------
    logger.info("\n[1/6] Importing TabPFN...")
    try:
        from tabpfn import TabPFNRegressor
        logger.info("  tabpfn imported successfully.")
    except ImportError:
        logger.error("  tabpfn not installed. Run: pip install tabpfn")
        raise

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    logger.info(f"\n[2/6] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"  Shape: {df.shape}")

    # Log direction distribution if enriched
    if "direction" in df.columns:
        dist = df["direction"].value_counts().to_dict()
        logger.info(f"  Direction distribution: {dist}")

    # ------------------------------------------------------------------
    # 3. Preprocess
    # ------------------------------------------------------------------
    logger.info("\n[3/6] Preprocessing...")
    X, y, meta = preprocess(df)
    feature_names = meta["features"]
    logger.info(f"  Features ({meta['n_features']}): {feature_names}")

    # Log1p transform on target
    y_log = np.log1p(y)

    # ------------------------------------------------------------------
    # 4. Train / Test split (same seed as XGBoost for fair comparison)
    # ------------------------------------------------------------------
    logger.info("\n[4/6] Splitting data (80/20)...")
    price_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test, y_log_train, y_log_test = train_test_split(
        X, y, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=price_bins
    )
    logger.info(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # 5. Train TabPFN
    # ------------------------------------------------------------------
    logger.info("\n[5/6] Training TabPFN...")
    model = TabPFNRegressor(random_state=RANDOM_STATE, ignore_pretraining_limits=True)

    import time
    t0 = time.time()
    model.fit(X_train, y_log_train)
    train_time = time.time() - t0
    logger.info(f"  Training complete in {train_time:.2f}s")

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    logger.info("\n[6/6] Evaluating...")
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

    logger.info("\n" + "=" * 60)
    logger.info(f"  Results — TabPFN on {dataset_label}")
    logger.info("=" * 60)
    logger.info(f"  RMSE:      {rmse / 1e9:.2f} Billion VND")
    logger.info(f"  MAE:       {mae / 1e9:.2f} Billion VND")
    logger.info(f"  R²:        {r2:.4f}")
    logger.info(f"  MAPE:      {mape:.2f}%")
    logger.info(f"  RMSE(log): {rmse_log:.4f}")
    logger.info(f"  Train time:{train_time:.2f}s")
    logger.info("=" * 60)

    # # ------------------------------------------------------------------
    # # 7. Plots
    # # ------------------------------------------------------------------
    # plot_dir = MODEL_DIR / "plots"
    # plot_dir.mkdir(parents=True, exist_ok=True)

    # pva_path = plot_pred_vs_actual(
    #     y_test, y_pred,
    #     plot_dir / f"tabpfn_pred_vs_actual_{dataset_label}.png",
    #     title=f"TabPFN — Predicted vs Actual ({dataset_label})"
    # )

    # logger.info("  Calculating TabPFN feature importance (permutation)...")
    # fi_path = plot_feature_importance(
    #     model, X_test, y_log_test, feature_names,
    #     plot_dir / f"tabpfn_feature_importance_{dataset_label}.png",
    #     title=f"TabPFN Feature Importance ({dataset_label})"
    # )

    # # Comparison plot (if baseline XGBoost exists)
    # xgb_baseline = load_xgb_baseline(dataset_label)
    # comparison_path = None
    # if xgb_baseline:
    #     comparison_path = plot_model_comparison(
    #         xgb_baseline, metrics,
    #         plot_dir / f"comparison_xgb_vs_tabpfn_{dataset_label}.png"
    #     )
    #     logger.info("\n  XGBoost vs TabPFN Comparison:")
    #     logger.info(f"  {'Metric':<15} {'XGBoost':>12} {'TabPFN':>12} {'Delta':>12}")
    #     logger.info(f"  {'-'*51}")
    #     for k, label in [("r2", "R²"), ("mape", "MAPE (%)"), ("rmse_log", "RMSE(log)")]:
    #         xgb_val = xgb_baseline.get(k, 0)
    #         tab_val = metrics.get(k, 0)
    #         delta = tab_val - xgb_val
    #         sign = "+" if delta > 0 else ""
    #         logger.info(f"  {label:<15} {xgb_val:>12.4f} {tab_val:>12.4f} {sign}{delta:>11.4f}")

    # ------------------------------------------------------------------
    # 8. Log to W&B
    # ------------------------------------------------------------------
    logger.info("\n  Logging to W&B...")
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
    # wandb.log({
    #     "pred_vs_actual": wandb.Image(str(pva_path)),
    #     "feature_importance": wandb.Image(str(fi_path)),
    # })
    # if comparison_path:
    #     wandb.log({"model_comparison": wandb.Image(str(comparison_path))})

    # wandb.finish()

    # ------------------------------------------------------------------
    # 9. Save JSON report
    # ------------------------------------------------------------------
    report = {
        "timestamp": timestamp,
        "dataset": dataset_label,
        "model": "TabPFN",
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "n_features": meta["n_features"],
        "features": meta["features"],
        "metrics": metrics,
        # "xgboost_baseline": xgb_baseline,
        # "comparison": {
        #     k: {
        #         "xgboost": xgb_baseline.get(k) if xgb_baseline else None,
        #         "tabpfn": metrics.get(k),
        #         "delta": (metrics.get(k, 0) - xgb_baseline.get(k, 0)) if xgb_baseline else None,
        #     }
        #     for k in ["r2", "mape", "rmse_log", "mae", "rmse"]
        # } if xgb_baseline else {},
    }

    report_path = REPORTS_DIR / f"tabpfn_{dataset_label}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"\n  JSON report: {report_path}")
    logger.info(f"  Log file:    {log_file}")
    logger.info("\n  Done! Check W&B for plots and comparison.")


if __name__ == "__main__":
    main()
