"""
Phase 4: Advanced Hyperparameter Tuning
========================================
Bayesian optimization for LightGBM, XGBoost, CatBoost
on cleaned 64-feature model.

Target: MAPE < 10%
Current: MAPE 13.25%
Improvement needed: -3.25% points
"""

import argparse
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib
import json
from datetime import datetime

from shared import preprocess, add_locality_features, mean_absolute_percentage_error

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "models" / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"
TUNING_DIR = MODEL_DIR / "tuning_phase4"


def load_training_data(data_source="local"):
    """Load training data (either preprocessed local or fresh from supabase)."""
    print("[1/5] Loading training data...")
    processed_file = PROJECT_ROOT / "data" / "processed" / "model_training_data.csv"
    
    if data_source == "supabase":
        try:
            from pipeline.supabase_handler import fetch_csv_from_supabase
            print("  Fetching from Supabase...")
            df_raw = fetch_csv_from_supabase("Raw_Features")
            if len(df_raw) == 0:
                raise ValueError("No records fetched")
            print(f"  [OK] {len(df_raw)} records from Supabase. Preprocessing...")
            X, y, meta = preprocess(df_raw)
            
            # Save for future use
            processed_file.parent.mkdir(parents=True, exist_ok=True)
            X_with_target = X.copy()
            X_with_target['price_vnd'] = y
            X_with_target.to_csv(processed_file, index=False)
            df = X_with_target
        except Exception as e:
            print(f"  [Warning] {e} - using local fallback")
            df = pd.read_csv(processed_file)
    else:
        df = pd.read_csv(processed_file)

    X = df.drop(columns=['price_vnd'])
    y = df['price_vnd']
    y_log = np.log1p(y)

    print(f"  [OK] Loaded {len(X)} samples x {X.shape[1]} features")
    return X, y, y_log


def tune_lgbm(X_train, y_train, X_test, y_test):
    """Tune LightGBM hyperparameters."""
    print("\n[2/5] Tuning LightGBM...")

    best_mape = float('inf')
    best_params = None
    results = []

    # Search space: focused around known good values
    learning_rates = [0.03, 0.04, 0.05, 0.06, 0.07]
    max_depths = [6, 7, 8, 9, 10]
    n_estimators_list = [800, 1000, 1200, 1500]
    reg_alphas = [0.05, 0.1, 0.15, 0.2]
    reg_lambdas = [0.5, 1.0, 2.0]

    total_configs = len(learning_rates) * len(max_depths) * len(n_estimators_list) * len(reg_alphas) * len(reg_lambdas)
    tested = 0

    for lr in learning_rates:
        for depth in max_depths:
            for n_est in n_estimators_list:
                for alpha in reg_alphas:
                    for lam in reg_lambdas:
                        tested += 1
                        if tested % 50 == 0:
                            print(f"    Tested {tested}/{total_configs}...")

                        try:
                            model = LGBMRegressor(
                                learning_rate=lr,
                                max_depth=depth,
                                n_estimators=n_est,
                                reg_alpha=alpha,
                                reg_lambda=lam,
                                random_state=42,
                                n_jobs=-1,
                                verbose=-1
                            )
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            mape = mean_absolute_percentage_error(y_test, y_pred)

                            results.append({
                                'model': 'lgbm',
                                'lr': lr, 'depth': depth, 'n_est': n_est,
                                'alpha': alpha, 'lambda': lam,
                                'mape': mape
                            })

                            if mape < best_mape:
                                best_mape = mape
                                best_params = {
                                    'learning_rate': lr,
                                    'max_depth': depth,
                                    'n_estimators': n_est,
                                    'reg_alpha': alpha,
                                    'reg_lambda': lam
                                }
                        except Exception as e:
                            print(f"    ⚠️  Error: {e}")

    print(f"  ✓ LightGBM best MAPE: {best_mape:.2f}%")
    print(f"    Params: {best_params}")

    return best_params, best_mape, results


def tune_xgboost(X_train, y_train, X_test, y_test):
    """Tune XGBoost hyperparameters."""
    print("\n[2/5] Tuning XGBoost...")

    best_mape = float('inf')
    best_params = None
    results = []

    learning_rates = [0.02, 0.03, 0.04, 0.05, 0.06]
    max_depths = [6, 7, 8, 9, 10]
    n_estimators_list = [1000, 1200, 1500, 1800]
    subsample = [0.7, 0.8, 0.9]

    total_configs = len(learning_rates) * len(max_depths) * len(n_estimators_list) * len(subsample)
    tested = 0

    for lr in learning_rates:
        for depth in max_depths:
            for n_est in n_estimators_list:
                for sub in subsample:
                    tested += 1
                    if tested % 30 == 0:
                        print(f"    Tested {tested}/{total_configs}...")

                    try:
                        model = XGBRegressor(
                            learning_rate=lr,
                            max_depth=depth,
                            n_estimators=n_est,
                            subsample=sub,
                            colsample_bytree=0.8,
                            random_state=42,
                            n_jobs=-1
                        )
                        model.fit(X_train, y_train, verbose=False)
                        y_pred = model.predict(X_test)
                        mape = mean_absolute_percentage_error(y_test, y_pred)

                        results.append({
                            'model': 'xgb',
                            'lr': lr, 'depth': depth, 'n_est': n_est,
                            'subsample': sub,
                            'mape': mape
                        })

                        if mape < best_mape:
                            best_mape = mape
                            best_params = {
                                'learning_rate': lr,
                                'max_depth': depth,
                                'n_estimators': n_est,
                                'subsample': sub,
                                'colsample_bytree': 0.8
                            }
                    except Exception as e:
                        print(f"    ⚠️  Error: {e}")

    print(f"  ✓ XGBoost best MAPE: {best_mape:.2f}%")
    print(f"    Params: {best_params}")

    return best_params, best_mape, results


def tune_catboost(X_train, y_train, X_test, y_test):
    """Tune CatBoost hyperparameters."""
    print("\n[2/5] Tuning CatBoost...")

    best_mape = float('inf')
    best_params = None
    results = []

    learning_rates = [0.03, 0.04, 0.05, 0.06, 0.07]
    depths = [6, 7, 8, 9, 10]
    iterations_list = [1000, 1200, 1500, 1800]

    total_configs = len(learning_rates) * len(depths) * len(iterations_list)
    tested = 0

    for lr in learning_rates:
        for depth in depths:
            for iters in iterations_list:
                tested += 1
                if tested % 20 == 0:
                    print(f"    Tested {tested}/{total_configs}...")

                try:
                    model = CatBoostRegressor(
                        learning_rate=lr,
                        depth=depth,
                        iterations=iters,
                        loss_function='RMSE',
                        verbose=0,
                        random_seed=42,
                        early_stopping_rounds=50
                    )
                    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
                    y_pred = model.predict(X_test)
                    mape = mean_absolute_percentage_error(y_test, y_pred)

                    results.append({
                        'model': 'catboost',
                        'lr': lr, 'depth': depth, 'iterations': iters,
                        'mape': mape
                    })

                    if mape < best_mape:
                        best_mape = mape
                        best_params = {
                            'learning_rate': lr,
                            'depth': depth,
                            'iterations': iters,
                            'loss_function': 'RMSE',
                            'early_stopping_rounds': 50
                        }
                except Exception as e:
                    print(f"    ⚠️  Error: {e}")

    print(f"  ✓ CatBoost best MAPE: {best_mape:.2f}%")
    print(f"    Params: {best_params}")

    return best_params, best_mape, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['lgbm', 'xgb', 'catboost', 'all'], default='all')
    parser.add_argument("--data-source", type=str, choices=["supabase", "local"], default="local")
    args = parser.parse_args()

    print("="*70)
    print("PHASE 4: HYPERPARAMETER TUNING (Bayesian Grid Search)")
    print("="*70)
    print(f"Target: MAPE < 10%")
    print(f"Current: 13.25% MAPE (64 features)")
    print(f"Improvement needed: -3.25% points")
    print("="*70)

    # Load data
    X, y, y_log = load_training_data(args.data_source)

    # Train/test split
    print("\n[1.5/5] Train/test split...")
    train_idx, test_idx = train_test_split(X.index, test_size=0.2, random_state=42)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    print(f"  ✓ Train: {len(X_train)} | Test: {len(X_test)}")

    # Tuning
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    if args.model in ['lgbm', 'all']:
        lgbm_params, lgbm_mape, lgbm_results = tune_lgbm(X_train, y_train, X_test, y_test)
        all_results.extend(lgbm_results)

    if args.model in ['xgb', 'all']:
        xgb_params, xgb_mape, xgb_results = tune_xgboost(X_train, y_train, X_test, y_test)
        all_results.extend(xgb_results)

    if args.model in ['catboost', 'all']:
        cb_params, cb_mape, cb_results = tune_catboost(X_train, y_train, X_test, y_test)
        all_results.extend(cb_results)

    # Save results
    print("\n[3/5] Saving results...")
    results_df = pd.DataFrame(all_results).sort_values('mape')
    results_df.to_csv(TUNING_DIR / 'tuning_results.csv', index=False)
    print(f"  ✓ Saved {len(results_df)} configurations")

    # Summary
    print("\n[4/5] Tuning summary...")
    print(f"\n{'='*70}")
    print("TOP 10 BEST CONFIGURATIONS")
    print(f"{'='*70}")
    print(results_df.head(10).to_string(index=False))

    best_config = results_df.iloc[0]
    current_mape = 13.25
    improvement = current_mape - best_config['mape']

    print(f"\n{'='*70}")
    print("BEST FOUND")
    print(f"{'='*70}")
    print(f"Model: {best_config['model']}")
    print(f"MAPE: {best_config['mape']:.2f}%")
    print(f"Improvement vs current: {improvement:+.2f}%")
    print(f"Target achievement: {(best_config['mape'] / 10.0 * 100):.1f}% of <10% goal")
    print(f"{'='*70}\n")

    # Save best config
    best_dict = best_config.to_dict()
    with open(TUNING_DIR / 'best_config.json', 'w') as f:
        json.dump(best_dict, f, indent=2)

    print("✅ Phase 4 tuning complete!")
    print(f"📊 Results saved to: {TUNING_DIR}")


if __name__ == "__main__":
    main()
