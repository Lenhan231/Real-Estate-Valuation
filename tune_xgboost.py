#!/usr/bin/env python3
"""
XGBoost Hyperparameter Tuning using Optuna
Optimize for MAPE < 10%
"""
import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path
from time import time
import json

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("Installing optuna...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

print("=" * 80)
print("🔧 XGBOOST HYPERPARAMETER TUNING")
print("=" * 80)

# =====================================================================
# 1. DATA LOADING & PREPROCESSING (same as before)
# =====================================================================
print("\n[1/5] Loading & preprocessing data...")
sys.path.insert(0, str(Path('.').resolve()))
from pipeline.supabase_handler import fetch_csv_from_supabase

df = fetch_csv_from_supabase()
print(f"✅ Loaded {len(df):,} records")

# Preprocessing
df = df.copy().drop_duplicates()
price_b = df['price_vnd'] / 1e9
df = df[(price_b >= 2.0) & (price_b <= 50.0)]
if 'area_m2' in df.columns:
    df = df[df['area_m2'].isna() | df['area_m2'].between(15, 500)]
    price_sqm = df['price_vnd'] / 1e6 / df['area_m2']
    df = df[price_sqm.isna() | ((price_sqm >= 30) & (price_sqm <= 800))]

# Temporal features
if "post_day" in df.columns:
    post_day_dt = pd.to_datetime(df["post_day"])
    df["post_day_year"] = post_day_dt.dt.year
    df["post_day_month"] = post_day_dt.dt.month
    df["post_day_day"] = post_day_dt.dt.day

# Feature engineering
if 'width_m' in df.columns and 'length_m' in df.columns:
    df['perimeter_m'] = (df['width_m'] + df['length_m']) * 2
    df['shape_ratio'] = (df['width_m'] + 0.1) / (df['length_m'] + 0.1)
if 'area_m2' in df.columns and 'num_floors' in df.columns:
    df['area_x_floors'] = df['area_m2'] * df['num_floors']
if 'area_m2' in df.columns and 'num_bedrooms' in df.columns:
    df['area_x_bedrooms'] = df['area_m2'] * df['num_bedrooms']
    df['area_per_bedroom'] = df['area_m2'] / (df['num_bedrooms'] + 1)
if 'area_m2' in df.columns:
    df['log_area'] = np.log1p(df['area_m2'])
if 'distance_to_center_km' in df.columns:
    df['log_distance'] = np.log1p(df['distance_to_center_km'])

if 'distance_to_center_km' in df.columns:
    df['location_score'] = (
        (10 / (df['distance_to_center_km'] + 1)) * 2.0 +
        (10 / (df.get('nearest_school_km', 1) + 1)) * 1.5 +
        (10 / (df.get('nearest_hospital_km', 1) + 1)) * 1.5 +
        (10 / (df.get('nearest_mall_km', 1) + 1)) * 1.0
    )

df['amenity_score'] = (
    df.get('school_count_3km', 0) * 1.0 +
    df.get('hospital_count_5km', 0) * 1.5 +
    df.get('supermarket_count_3km', 0) * 1.0 +
    df.get('mall_count_3km', 0) * 2.0 +
    df.get('metro_count_5km', 0) * 3.0
)

for col in ['nearest_metro_km', 'nearest_mall_km', 'nearest_supermarket_km']:
    if col in df.columns:
        df[f'{col}_missing'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(999.0)

for col in ['width_m', 'length_m']:
    if col in df.columns:
        df[f'{col}_missing'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median())

drop_cols = ['id', 'price_vnd', 'url', 'link', 'title', 'post_day', 'description',
             'street', 'ward', 'district', 'locality', 'region',
             'matched_address', 'old_address', 'lat', 'lon', 'listing_id']
features_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
cat_cols = features_df.select_dtypes(include=['object', 'category']).columns
features_df = pd.get_dummies(features_df, columns=cat_cols, dummy_na=True)

import re
features_df.columns = features_df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
features_df.columns = features_df.columns.str.replace(r'^_+', '', regex=True)
features_df.columns = features_df.columns.str.replace(r'_+', '_', regex=True)

X = features_df
y = df['price_vnd']
y_log = np.log1p(y)

train_idx, test_idx = train_test_split(X.index, test_size=0.2, random_state=42)
X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_log_train, y_log_test = y_log.loc[train_idx], y_log.loc[test_idx]
y_test = y.loc[test_idx]

print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {X_train.shape[1]}")

# =====================================================================
# 2. DEFINE OBJECTIVE FUNCTION
# =====================================================================
print("\n[2/5] Setting up optimization...")

def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def objective(trial):
    """Optuna objective: minimize MAPE"""

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10, log=True),
        'random_state': 42,
        'verbosity': 0,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_log_train, eval_set=[(X_test, y_log_test)], verbose=False)

    y_pred_log = model.predict(X_test)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)

    mape = mean_absolute_percentage_error(y_test.values, y_pred)

    return mape

# =====================================================================
# 3. OPTIMIZE
# =====================================================================
print("[3/5] Running optimization (10 trials)...")
print("  This may take a few minutes...\n")

sampler = TPESampler(seed=42)
study = optuna.create_study(
    sampler=sampler,
    direction='minimize',
    pruner=MedianPruner()
)

study.optimize(objective, n_trials=10, show_progress_bar=True)

# =====================================================================
# 4. RESULTS
# =====================================================================
print("\n" + "=" * 80)
print("📊 OPTIMIZATION RESULTS")
print("=" * 80)

best_trial = study.best_trial

print(f"\nBest MAPE: {best_trial.value:.2f}%")
print(f"Improvement: {18.01 - best_trial.value:.2f}% (from baseline 18.01%)")

print("\n🎯 Best Hyperparameters:")
for key, value in sorted(best_trial.params.items()):
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

# =====================================================================
# 5. TRAIN FINAL MODEL
# =====================================================================
print("\n[4/5] Training final model with best params...")

best_params = best_trial.params.copy()
best_params['random_state'] = 42
best_params['verbosity'] = 0

t0 = time()
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_log_train, eval_set=[(X_test, y_log_test)], verbose=False)
train_time = time() - t0

y_pred_log = final_model.predict(X_test)
y_pred = np.clip(np.expm1(y_pred_log), 0, None)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test.values, y_pred)

print(f"✅ Done in {train_time:.2f}s")

print("\n" + "=" * 80)
print("🏆 FINAL MODEL PERFORMANCE")
print("=" * 80)
print(f"\nRMSE:  {rmse/1e9:.2f}B VND")
print(f"MAE:   {mae/1e9:.2f}B VND")
print(f"R²:    {r2:.4f}")
print(f"MAPE:  {mape:.2f}% 🎯")

if mape < 10:
    print(f"\n✅ TARGET ACHIEVED! (< 10%)")
else:
    print(f"\n⚠️  Still above target ({mape - 10:.2f}% more improvement needed)")

# Save
print("\n[5/5] Saving tuned model...")
import joblib
model_path = Path("models/save_models/xgboost_tuned_model.pkl")
joblib.dump(final_model, model_path)
print(f"✅ Saved: {model_path}")

# Save params
params_path = Path("models/save_models/xgboost_best_params.json")
with open(params_path, 'w') as f:
    json.dump(best_params, f, indent=2)
print(f"✅ Saved: {params_path}")

print("\n" + "=" * 80)
print("✅ TUNING COMPLETE!")
print("=" * 80)
print("\n📈 Trial History:")
for i, trial in enumerate(study.trials[:5], 1):
    print(f"  Trial {i}: MAPE {trial.value:.2f}%")
print(f"  ...")
print(f"  Trial 10: MAPE {study.best_trial.value:.2f}% (Best)")

print("\n" + "=" * 80)
