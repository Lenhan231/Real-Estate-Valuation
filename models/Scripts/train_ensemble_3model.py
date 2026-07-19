"""
Enhanced Ensemble Training Pipeline (LGBM + CatBoost + XGBoost)
================================================================
Trains a 3-model ensemble with early stopping for improved accuracy.
- LightGBM: Gradient boosting
- CatBoost: Categorical gradient boosting
- XGBoost: Extreme gradient boosting
Weighted averaging based on validation performance.
"""

import argparse
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
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping, log_evaluation
from xgboost import XGBRegressor
import joblib
import time

from shared import preprocess, add_locality_features, log_to_wandb, mean_absolute_percentage_error
from shared import plot_feature_importance, plot_pred_vs_actual

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "models" / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"
PLOT_DIR = MODEL_DIR / "plots"

WANDB_PROJECT = "real-estate-valuation"


def train_3model_ensemble(X_tr_b, y_tr_b, X_te_b, y_te_b, price_bin, ptype):
    """
    Train LightGBM, CatBoost, and XGBoost with early stopping.
    Returns dict of trained models and their validation errors.
    """
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        CatBoostRegressor = None

    models_dict = {}
    val_errors = {}

    lgb_params = {
        "n_estimators": 1000,
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    xgb_params = {
        "n_estimators": 1000,
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }

    cb_params = {
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.03,
        "loss_function": "RMSE",
        "verbose": 0,
        "random_seed": 42,
        "early_stopping_rounds": 50,
    }

    # Train LightGBM with early stopping
    print(f"    - Training LightGBM...")
    model_lgb = LGBMRegressor(**lgb_params)
    eval_set_lgb = [(X_te_b, y_te_b)] if len(X_te_b) > 0 else None
    callbacks_lgb = [lgb_early_stopping(50), log_evaluation(period=0)] if len(X_te_b) > 0 else []
    model_lgb.fit(
        X_tr_b, y_tr_b,
        eval_set=eval_set_lgb,
        callbacks=callbacks_lgb
    )
    models_dict["lgbm"] = model_lgb

    # Train XGBoost
    print(f"    - Training XGBoost...")
    model_xgb = XGBRegressor(**xgb_params)
    if len(X_te_b) > 0:
        model_xgb.fit(
            X_tr_b, y_tr_b,
            eval_set=[(X_te_b, y_te_b)],
            verbose=False
        )
    else:
        model_xgb.fit(X_tr_b, y_tr_b, verbose=False)
    models_dict["xgb"] = model_xgb

    # Train CatBoost with early stopping
    model_cb = None
    if CatBoostRegressor is not None:
        print(f"    - Training CatBoost...")
        model_cb = CatBoostRegressor(**cb_params)
        if len(X_te_b) > 0:
            model_cb.fit(X_tr_b, y_tr_b, eval_set=(X_te_b, y_te_b))
        else:
            model_cb.fit(X_tr_b, y_tr_b)
        models_dict["cb"] = model_cb

    # Calculate validation errors for weighting
    if len(X_te_b) > 0:
        y_pred_lgb = model_lgb.predict(X_te_b)
        y_pred_xgb = model_xgb.predict(X_te_b)

        val_errors["lgbm"] = np.sqrt(mean_squared_error(y_te_b, y_pred_lgb))
        val_errors["xgb"] = np.sqrt(mean_squared_error(y_te_b, y_pred_xgb))

        if model_cb is not None:
            y_pred_cb = model_cb.predict(X_te_b)
            val_errors["cb"] = np.sqrt(mean_squared_error(y_te_b, y_pred_cb))

    return models_dict, val_errors


def ensemble_predictions(models_dict, X_test, val_errors):
    """
    Generate weighted ensemble predictions based on validation RMSE.
    Better models get higher weights.
    """
    predictions = {}

    # Get predictions from each model
    for model_name, model in models_dict.items():
        predictions[model_name] = model.predict(X_test)

    # Calculate weights based on inverse MSE (better model = lower error = higher weight)
    if val_errors:
        total_error = sum(1.0 / (e + 1e-6) for e in val_errors.values())
        weights = {name: (1.0 / (val_errors.get(name, 1.0) + 1e-6)) / total_error
                   for name in predictions.keys()}
    else:
        weights = {name: 1.0 / len(predictions) for name in predictions.keys()}

    print(f"      Model weights: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")

    # Weighted average
    ensemble_pred = np.zeros_like(predictions[list(predictions.keys())[0]])
    for model_name, pred in predictions.items():
        ensemble_pred += pred * weights.get(model_name, 1.0 / len(predictions))

    return ensemble_pred, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cleaned", help="Name of the dataset to use.")
    parser.add_argument("--data-source", type=str, choices=["supabase", "local"], default="local", help="Source of the data")
    args = parser.parse_args()

    dataset_label = args.dataset
    data_source = args.data_source
    csv_path = DATA_DIR / f"alonhadat_features_{dataset_label}.csv"

    print(f"============================================================")
    print(f"  Ensemble Training V2 (LGBM+CatBoost+XGBoost)")
    print(f"  With Early Stopping & Weighted Averaging")
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

    processed_data_dir = PROJECT_ROOT / "data" / "processed"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    X_with_target = X.copy()
    X_with_target['price_vnd'] = y
    X_with_target.to_csv(processed_data_dir / "model_training_data.csv", index=False)
    print(f"  Saved training data to model_training_data.csv")

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

    meta["features"] = list(X_train.columns)
    feature_names = meta["features"]

    print("[4/5] Training 3-Model Ensemble (LGBM + CatBoost + XGBoost)...")

    global_y_pred = []
    global_y_test = []
    global_y_log_pred = []
    global_y_log_test = []

    BINS_VND = [0, 5e9, 20e9, float('inf')]
    BIN_LABELS = ['low', 'mid', 'high']

    train_prices = np.expm1(y_log_train)
    test_prices = np.expm1(y_log_test)
    train_bins = pd.cut(train_prices, bins=BINS_VND, labels=BIN_LABELS)
    test_bins = pd.cut(test_prices, bins=BINS_VND, labels=BIN_LABELS)

    models = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for price_bin in BIN_LABELS:
        for ptype in ['property_type_nha_mat_tien', 'property_type_nha_trong_hem']:
            if ptype not in X_train.columns:
                continue

            train_mask = (train_bins == price_bin) & (X_train[ptype] == 1)
            test_mask = (test_bins == price_bin) & (X_test[ptype] == 1)

            X_tr_b = X_train[train_mask]
            y_tr_b = y_log_train[train_mask]
            X_te_b = X_test[test_mask]
            y_te_b = y_log_test[test_mask]

            if len(X_tr_b) < 10:
                print(f"  Skipping bucket {price_bin} + {ptype} (Too few samples)")
                continue

            print(f"  Training bucket {price_bin} + {ptype} (Train: {len(X_tr_b)}, Test: {len(X_te_b)})...")

            models_dict, val_errors = train_3model_ensemble(X_tr_b, y_tr_b, X_te_b, y_te_b, price_bin, ptype)

            if len(X_te_b) > 0:
                y_log_pred_b, weights = ensemble_predictions(models_dict, X_te_b, val_errors)

                global_y_log_pred.extend(y_log_pred_b)
                global_y_log_test.extend(y_te_b.values)

                y_pred_b = np.expm1(y_log_pred_b)
                y_pred_b = np.clip(y_pred_b, 0, None)

                global_y_pred.extend(y_pred_b)
                global_y_test.extend(np.expm1(y_te_b.values))

            # Save all 3 models
            for model_name, model in models_dict.items():
                file_name = f"{model_name}_{price_bin}_{ptype.replace('property_type_', '')}.pkl"
                model_path = MODEL_DIR / file_name
                joblib.dump(model, model_path)

            models[f"{price_bin}_{ptype}"] = models_dict

    print(f"  All buckets trained in {time.time() - t0:.2f}s")

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
    print(f"  Results — {dataset_label} (3-Model Ensemble V2)")
    print(f"{'='*60}")
    print(f"  Global RMSE:      {rmse / 1e9:.2f} Billion VND")
    print(f"  Global MAE:       {mae / 1e9:.2f} Billion VND")
    print(f"  Global R²:        {r2:.4f}")
    print(f"  Global MAPE:      {mape:.2f}%")
    print(f"  RMSE(log):        {rmse_log:.4f}")
    print(f"{'='*60}\n")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if "mid_property_type_nha_trong_hem" in models:
        model_lgbm = models["mid_property_type_nha_trong_hem"].get("lgbm")
        if model_lgbm:
            fi_path = plot_feature_importance(model_lgbm, feature_names, PLOT_DIR / f"feature_importance_ensemble_v2_{dataset_label}.png")

    pva_path = plot_pred_vs_actual(global_y_test, global_y_pred, PLOT_DIR / f"pred_vs_actual_ensemble_v2_{dataset_label}.png")
    print(f"  Plots saved to {PLOT_DIR}")

    print("\n✅ Ensemble V2 training complete!")
    print(f"📊 Models saved: {MODEL_DIR}")


if __name__ == "__main__":
    main()
