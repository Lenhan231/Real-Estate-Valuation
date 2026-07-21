"""
STACKING ENSEMBLE: Base models (9) + Meta-learner (Ridge/Lasso/MLP)
=========================================================================
Architecture:
  Level 0 (Base): LightGBM, XGBoost, CatBoost × 3 price tiers (9 models)
  Level 1 (Meta): Ridge, Lasso, MLP trained on base predictions

Expected improvement: +0.5-1.5% MAPE (via ensemble diversity)
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping, log_evaluation
from xgboost import XGBRegressor
import joblib
import time

from shared import preprocess, add_locality_features, mean_absolute_percentage_error

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "models" / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"
STACKING_DIR = MODEL_DIR / "stacking"

def train_base_models(X_train, y_train, X_val, y_val, price_tier):
    """Train 3 base models (LightGBM, XGBoost, CatBoost) for one price tier."""
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        CatBoostRegressor = None

    models_dict = {}

    # Hyperparameters
    lgb_params = {
        "n_estimators": 1000, "max_depth": 8, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
        "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1, "verbose": -1,
    }

    xgb_params = {
        "n_estimators": 1500, "max_depth": 8, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": -1,
    }

    cb_params = {
        "iterations": 1500, "depth": 8, "learning_rate": 0.05, "loss_function": "RMSE",
        "verbose": 0, "random_seed": 42, "early_stopping_rounds": 50,
    }

    # LightGBM
    model_lgb = LGBMRegressor(**lgb_params)
    eval_set_lgb = [(X_val, y_val)] if len(X_val) > 0 else None
    callbacks_lgb = [lgb_early_stopping(50), log_evaluation(period=0)] if len(X_val) > 0 else []
    model_lgb.fit(X_train, y_train, eval_set=eval_set_lgb, callbacks=callbacks_lgb)
    models_dict["lgbm"] = model_lgb

    # XGBoost
    model_xgb = XGBRegressor(**xgb_params)
    if len(X_val) > 0:
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model_xgb.fit(X_train, y_train, verbose=False)
    models_dict["xgb"] = model_xgb

    # CatBoost
    if CatBoostRegressor is not None:
        model_cb = CatBoostRegressor(**cb_params)
        if len(X_val) > 0:
            model_cb.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            model_cb.fit(X_train, y_train)
        models_dict["cb"] = model_cb

    return models_dict


def get_base_predictions(models_dict, X):
    """Get predictions from all 3 base models."""
    predictions = {}
    for model_name, model in models_dict.items():
        predictions[model_name] = model.predict(X)
    return predictions


def train_meta_learners(X_meta_train, y_meta_train, X_meta_val, y_meta_val):
    """Train 3 meta-learners on base predictions."""
    meta_models = {}

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_meta_train, y_meta_train)
    ridge_pred = ridge.predict(X_meta_val)
    ridge_rmse = np.sqrt(mean_squared_error(y_meta_val, ridge_pred))
    meta_models["ridge"] = (ridge, ridge_rmse)

    # Lasso
    lasso = Lasso(alpha=0.001, max_iter=2000)
    lasso.fit(X_meta_train, y_meta_train)
    lasso_pred = lasso.predict(X_meta_val)
    lasso_rmse = np.sqrt(mean_squared_error(y_meta_val, lasso_pred))
    meta_models["lasso"] = (lasso, lasso_rmse)

    # Neural Network (MLP)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
                       learning_rate_init=0.001, early_stopping=True, validation_fraction=0.2)
    mlp.fit(X_meta_train, y_meta_train)
    mlp_pred = mlp.predict(X_meta_val)
    mlp_rmse = np.sqrt(mean_squared_error(y_meta_val, mlp_pred))
    meta_models["mlp"] = (mlp, mlp_rmse)

    return meta_models


def main():
    print("=" * 70)
    print("STACKING ENSEMBLE: Base (9) + Meta-learners (3)")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading data...")
    try:
        from pipeline.supabase_handler import fetch_csv_from_supabase
        df = fetch_csv_from_supabase("Raw_Features")
        if len(df) == 0:
            raise ValueError("No records fetched")
        print(f"  ✓ {len(df)} records from Supabase")
    except Exception as e:
        print(f"  [Warning] {e} - using local fallback")
        df = pd.read_csv(DATA_DIR / "alonhadat_features_cleaned.csv")

    # Preprocess
    print("\n[2/6] Preprocessing...")
    X, y, meta = preprocess(df)
    y_log = np.log1p(y)
    print(f"  ✓ Features: {X.shape[1]}")

    # Train/test split
    print("\n[3/6] Train/test split (80/20)...")
    train_idx, test_idx = train_test_split(X.index, test_size=0.2, random_state=42)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    y_log_train, y_log_test = y_log.loc[train_idx], y_log.loc[test_idx]

    X_train, X_test = add_locality_features(X_train, X_test, df, train_idx, test_idx, y_train)

    drop_text = [c for c in ['locality', 'description'] if c in X_train.columns]
    if drop_text:
        X_train = X_train.drop(columns=drop_text)
        X_test = X_test.drop(columns=drop_text)

    print(f"  ✓ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Price tier binning
    BINS_VND = [0, 5e9, 20e9, float('inf')]
    BIN_LABELS = ['low', 'mid', 'high']

    train_prices = np.expm1(y_log_train)
    test_prices = np.expm1(y_log_test)
    train_bins = pd.cut(train_prices, bins=BINS_VND, labels=BIN_LABELS)
    test_bins = pd.cut(test_prices, bins=BINS_VND, labels=BIN_LABELS)

    # Train base models for each tier
    print("\n[4/6] Training Base Models (9: LightGBM+XGBoost+CatBoost × 3-tier)...")

    STACKING_DIR.mkdir(parents=True, exist_ok=True)
    base_models = {}
    global_y_pred_base = []
    global_y_test_list = []

    # Meta-features for training meta-learners
    X_meta_train_list = []
    y_meta_train_list = []

    t0 = time.time()
    for price_bin in BIN_LABELS:
        train_mask = (train_bins == price_bin)
        test_mask = (test_bins == price_bin)

        X_tr_b = X_train[train_mask]
        y_tr_b = y_log_train[train_mask]
        X_te_b = X_test[test_mask]
        y_te_b = y_log_test[test_mask]

        if len(X_tr_b) < 10:
            print(f"  Skipping tier {price_bin}")
            continue

        print(f"  Training {price_bin} tier base models: {len(X_tr_b)} train, {len(X_te_b)} test")

        # Train base models
        models_dict = train_base_models(X_tr_b, y_tr_b, X_te_b, y_te_b, price_bin)
        base_models[f"{price_bin}"] = models_dict

        # Get base predictions on test set
        if len(X_te_b) > 0:
            base_preds = get_base_predictions(models_dict, X_te_b)
            base_pred_array = np.column_stack([base_preds[k] for k in sorted(base_preds.keys())])

            # Add to global for final evaluation
            global_y_pred_base.extend(np.expm1(np.clip(np.mean(base_pred_array, axis=1), 0, None)))
            global_y_test_list.extend(np.expm1(y_te_b.values))

            # Collect meta-features
            X_meta_train_list.append(base_pred_array)
            y_meta_train_list.append(y_te_b.values)

        # Save base models
        for model_name, model in models_dict.items():
            file_name = f"{model_name}_{price_bin}.pkl"
            model_path = MODEL_DIR / file_name
            joblib.dump(model, model_path)

    print(f"  ✓ Base training complete ({time.time() - t0:.1f}s)")

    # Train meta-learners
    print("\n[5/6] Training Meta-learners (Ridge + Lasso + MLP)...")

    X_meta_train = np.vstack(X_meta_train_list)
    y_meta_train = np.concatenate(y_meta_train_list)

    # Use 20% of meta-features for validation
    meta_train_idx, meta_val_idx = train_test_split(
        range(len(X_meta_train)), test_size=0.2, random_state=42
    )
    X_meta_train_split = X_meta_train[meta_train_idx]
    y_meta_train_split = y_meta_train[meta_train_idx]
    X_meta_val = X_meta_train[meta_val_idx]
    y_meta_val = y_meta_train[meta_val_idx]

    meta_models = train_meta_learners(X_meta_train_split, y_meta_train_split, X_meta_val, y_meta_val)

    print(f"  ✓ Meta-learners trained")
    print(f"    Ridge RMSE:   {meta_models['ridge'][1]:.6f}")
    print(f"    Lasso RMSE:   {meta_models['lasso'][1]:.6f}")
    print(f"    MLP RMSE:     {meta_models['mlp'][1]:.6f}")

    # Save meta-learners
    for model_name, (model, _) in meta_models.items():
        model_path = STACKING_DIR / f"meta_{model_name}.pkl"
        joblib.dump(model, model_path)

    # Evaluate
    print("\n[6/6] Evaluating Stacking Ensemble...")

    global_y_test = np.array(global_y_test_list)
    global_y_pred_base = np.array(global_y_pred_base)

    rmse_base = np.sqrt(mean_squared_error(global_y_test, global_y_pred_base))
    mae_base = mean_absolute_error(global_y_test, global_y_pred_base)
    r2_base = r2_score(global_y_test, global_y_pred_base)
    mape_base = mean_absolute_percentage_error(global_y_test, global_y_pred_base)

    print("\n" + "=" * 70)
    print("STACKING ENSEMBLE RESULTS")
    print("=" * 70)
    print(f"\n📊 Base Models (9-model average):")
    print(f"   MAPE:  {mape_base:.2f}%")
    print(f"   R²:    {r2_base:.4f}")
    print(f"   MAE:   {mae_base/1e9:.2f} Billion VND")
    print(f"   RMSE:  {rmse_base/1e9:.2f} Billion VND")

    print(f"\n📊 Meta-learners (Ridge / Lasso / MLP):")
    print(f"   Ridge RMSE: {meta_models['ridge'][1]:.6f}")
    print(f"   Lasso RMSE: {meta_models['lasso'][1]:.6f}")
    print(f"   MLP RMSE:   {meta_models['mlp'][1]:.6f}")

    print(f"\n💡 Improvement potential: Ridge/Lasso/MLP ensemble may improve base MAPE by 0.5-1.5%")
    print("=" * 70)

    print(f"\n✅ Stacking ensemble training complete!")
    print(f"📊 Base models saved to: {MODEL_DIR}")
    print(f"📊 Meta-learners saved to: {STACKING_DIR}")

if __name__ == "__main__":
    main()
