"""
XGBoost Training Pipeline with Weights & Biases Tracking
=========================================================
Trains an XGBoost regressor to predict house prices 
from the cleaned feature datasets.
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import wandb
from shared import preprocess, mean_absolute_percentage_error, add_locality_features
from shared import plot_feature_importance, plot_pred_vs_actual
from shared import log_to_wandb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "models" / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"
PLOT_DIR = MODEL_DIR / "plots"

WANDB_PROJECT = "real-estate-valuation"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cleaned", help="Name of the dataset to use.")
    parser.add_argument("--data-source", type=str, choices=["supabase", "local"], default="local", help="Source of the data")
    args = parser.parse_args()

    dataset_label = args.dataset
    data_source = args.data_source
    csv_path = DATA_DIR / f"alonhadat_features_{dataset_label}.csv"
    
    print(f"============================================================")
    print(f"  XGBoost Training — Dataset: {dataset_label}")
    print(f"============================================================\n")

    print("[1/5] Loading data...")
    if data_source == "supabase":
        try:
            from pipeline.supabase_handler import fetch_csv_from_supabase
            df = fetch_csv_from_supabase("Raw_Features")
            if len(df) == 0:
                raise ValueError("No records fetched")
            print(f"  OK: {len(df)} records from Supabase")
        except Exception as e:
            print(f"  [Error] Supabase fetch failed: {e}")
            print(f"  Falling back to local data...")
            df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path)

    print("[2/5] Preprocessing...")
    X, y, meta = preprocess(df)
    feature_names = meta["features"]
    
    y_log = np.log1p(y)

    print("[3/5] Splitting data (80/20)...")
    train_idx, test_idx = train_test_split(X.index, test_size=0.2, random_state=42)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    y_log_train, y_log_test = y_log.loc[train_idx], y_log.loc[test_idx]

    X_train, X_test = add_locality_features(X_train, X_test, df, train_idx, test_idx, y_train)

    drop_text = [c for c in ['locality', 'description'] if c in X_train.columns]
    if drop_text:
        X_train = X_train.drop(columns=drop_text)
        X_test = X_test.drop(columns=drop_text)

    print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    # Update meta['features'] with the newly added locality columns
    meta["features"] = list(X_train.columns)
    feature_names = meta["features"]

    # ------------------------------------------------------------------
    # 4. Train XGBoost Model
    # ------------------------------------------------------------------
    print("[4/5] Training XGBoost Model...")
    import time
    import joblib

    xgb_params = {
        "n_estimators": 1000,
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    
    print(f"  Training single XGBoost model (Train: {len(X_train)}, Test: {len(X_test)})...")
    
    model = xgb.XGBRegressor(**xgb_params)
    eval_set = [(X_test, y_log_test)] if len(X_test) > 0 else None
    
    model.fit(X_train, y_log_train, eval_set=eval_set, verbose=False)
    
    if len(X_test) > 0:
        y_log_pred_b = model.predict(X_test)
        
        global_y_log_pred = y_log_pred_b
        global_y_log_test = y_log_test.values
        
        y_pred_b = np.expm1(y_log_pred_b)
        y_pred_b = np.clip(y_pred_b, 0, None)
        
        global_y_pred = y_pred_b
        global_y_test = np.expm1(y_log_test.values)
        
    model_path = MODEL_DIR / f"xgboost_{dataset_label}_{data_source}.pkl"
    joblib.dump(model, model_path)
            
    print(f"  Model trained in {time.time() - t0:.2f}s")

    
    BINS_VND = [0, 5e9, 20e9, float('inf')]
    print("[5/5] Evaluating Globally...")

    
    global_y_test = np.array(global_y_test)
    global_y_pred = np.array(global_y_pred)
    global_y_log_test = np.array(global_y_log_test)
    global_y_log_pred = np.array(global_y_log_pred)

    rmse = np.sqrt(mean_squared_error(global_y_test, global_y_pred))
    mae = mean_absolute_error(global_y_test, global_y_pred)
    r2 = r2_score(global_y_test, global_y_pred)
    mape = mean_absolute_percentage_error(global_y_test, global_y_pred)

    print("\n  Per-segment MAPE breakdown:")
    seg_test = pd.cut(global_y_test, bins=BINS_VND, labels=[0, 1, 2]).astype(int)
    for seg_id in [0, 1, 2]:
        mask = (seg_test == seg_id)
        if mask.sum() > 0:
            seg_mape = mean_absolute_percentage_error(global_y_test[mask], global_y_pred[mask])
            print(f"    Segment {seg_id} MAPE: {seg_mape:.2f}% ({mask.sum()} test samples)")

    rmse_log = np.sqrt(mean_squared_error(global_y_log_test, global_y_log_pred))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "rmse_log": rmse_log,
    }

    print(f"\n{'='*60}")
    print(f"  Results — {dataset_label} (6-Bucket Global, LGBM+CatBoost)")
    print(f"{'='*60}")
    print(f"  Global RMSE:      {rmse / 1e9:.2f} Billion VND")
    print(f"  Global MAE:       {mae / 1e9:.2f} Billion VND")
    print(f"  Global R²:        {r2:.4f}")
    print(f"  Global MAPE:      {mape:.2f}%")
    print(f"  RMSE(log):        {rmse_log:.4f}")
    print(f"{'='*60}\n")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if True:
        fi_path = plot_feature_importance(model, feature_names, PLOT_DIR / f"feature_importance_xgboost_{dataset_label}.png")

    pva_path = plot_pred_vs_actual(global_y_test, global_y_pred, PLOT_DIR / f"pred_vs_actual_{dataset_label}_global.png")
    print(f"  Plots saved to {PLOT_DIR}")

    print("  Logging to W&B...")
    log_to_wandb(
        project_name=WANDB_PROJECT,
        run_name=f"xgboost-{dataset_label}-{data_source}",
        config={
            "dataset": dataset_label,
            "data_source": data_source,
            "n_samples_train": len(global_y_log_pred),
            "n_samples_test": len(global_y_test),
            "n_features": meta["n_features"],
            "features": meta["features"],
            "target_transform": "log1p",
            "label_encoders": meta["label_encoders"],
            **xgb_params,
        },
        metrics=metrics,
        pred_vs_actual_path=pva_path,
        feature_importance_path=fi_path,
    )
    print("\n  Done! Check your W&B dashboard for results. Models saved to models/.")

if __name__ == "__main__":
    main()
