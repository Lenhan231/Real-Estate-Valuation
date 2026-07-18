#!/usr/bin/env python3
"""
Standalone training script - LightGBM, XGBoost, CatBoost
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import warnings
from pathlib import Path
from time import time
import sys

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import xgboost as xgb
from catboost import CatBoostRegressor

print("=" * 80)
print("🤖 REAL ESTATE PRICE PREDICTION - MODEL TRAINING")
print("=" * 80)

# =====================================================================
# 1. DATA LOADING
# =====================================================================
print("\n[1/6] Loading data from Supabase...")
sys.path.insert(0, str(Path('.').resolve()))
from pipeline.supabase_handler import fetch_csv_from_supabase

df = fetch_csv_from_supabase()
print(f"✅ Loaded {len(df):,} records, {len(df.columns)} columns")

# =====================================================================
# 2. PREPROCESSING & FEATURE ENGINEERING
# =====================================================================
print("\n[2/6] Preprocessing & Feature Engineering...")

df = df.copy()
df = df.drop_duplicates()

# Filter outliers
price_b = df['price_vnd'] / 1e9
df = df[(price_b >= 2.0) & (price_b <= 50.0)]
if 'area_m2' in df.columns:
    df = df[df['area_m2'].isna() | df['area_m2'].between(15, 500)]
    price_sqm = df['price_vnd'] / 1e6 / df['area_m2']
    df = df[price_sqm.isna() | ((price_sqm >= 30) & (price_sqm <= 800))]

print(f"  After filtering: {len(df):,} records")

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

# Location & amenity scores
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

# Text features
for col in ['description', 'title']:
    if col in df.columns:
        lower = df[col].astype(str).str.lower()
        df['is_hem_xe_hoi'] = lower.str.contains('hẻm xe hơi|hxh|ô tô').astype(int)
        df['is_mat_tien'] = lower.str.contains('mặt tiền|mặt phố').astype(int)
        df['is_gap'] = lower.str.contains('gấp|giảm giá|cần bán').astype(int)

# Handle missing values
for col in ['nearest_metro_km', 'nearest_mall_km', 'nearest_supermarket_km']:
    if col in df.columns:
        df[f'{col}_missing'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(999.0)

for col in ['width_m', 'length_m']:
    if col in df.columns:
        df[f'{col}_missing'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median())

# Drop non-predictive columns
drop_cols = ['id', 'price_vnd', 'url', 'link', 'title', 'post_day', 'description',
             'street', 'ward', 'district', 'locality', 'region',
             'matched_address', 'old_address', 'lat', 'lon', 'listing_id']
features_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# One-hot encode
cat_cols = features_df.select_dtypes(include=['object', 'category']).columns
features_df = pd.get_dummies(features_df, columns=cat_cols, dummy_na=True)

# Fix column names - replace ALL special characters
import re
features_df.columns = features_df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
# Remove leading underscores and consecutive underscores
features_df.columns = features_df.columns.str.replace(r'^_+', '', regex=True)
features_df.columns = features_df.columns.str.replace(r'_+', '_', regex=True)
# Handle any remaining duplicates from renaming
if features_df.columns.duplicated().any():
    cols = pd.Series(features_df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    features_df.columns = cols

print(f"  Features engineered: {features_df.shape[1]} features")

# =====================================================================
# 3. TRAIN-TEST SPLIT
# =====================================================================
print("\n[3/6] Train-Test Split (80/20)...")

X = features_df
y = df['price_vnd']
y_log = np.log1p(y)

train_idx, test_idx = train_test_split(X.index, test_size=0.2, random_state=42)
X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]
y_log_train, y_log_test = y_log.loc[train_idx], y_log.loc[test_idx]

print(f"  Train: {len(X_train):,} samples")
print(f"  Test: {len(X_test):,} samples")

# =====================================================================
# 4. HELPER FUNCTIONS
# =====================================================================
def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# =====================================================================
# 5. MODEL TRAINING
# =====================================================================
print("\n[4/6] Training Models...")

models_trained = {}
predictions = {}

# LightGBM
print("\n  🌳 Training LightGBM...")
t0 = time()
lgb_model = LGBMRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_log_train)
lgbm_time = time() - t0
y_lgbm_log_pred = lgb_model.predict(X_test)
y_lgbm_pred = np.clip(np.expm1(y_lgbm_log_pred), 0, None)
models_trained['LightGBM'] = lgb_model
predictions['LightGBM'] = y_lgbm_pred
print(f"     ✅ Done in {lgbm_time:.2f}s")

# XGBoost
print("  🚀 Training XGBoost...")
t0 = time()
xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
)
xgb_model.fit(X_train, y_log_train, eval_set=[(X_test, y_log_test)], verbose=False)
xgb_time = time() - t0
y_xgb_log_pred = xgb_model.predict(X_test)
y_xgb_pred = np.clip(np.expm1(y_xgb_log_pred), 0, None)
models_trained['XGBoost'] = xgb_model
predictions['XGBoost'] = y_xgb_pred
print(f"     ✅ Done in {xgb_time:.2f}s")

# CatBoost
print("  🐱 Training CatBoost...")
t0 = time()
cat_model = CatBoostRegressor(
    iterations=500, depth=8, learning_rate=0.03,
    subsample=0.8, random_state=42, verbose=False
)
cat_model.fit(X_train, y_log_train)
cat_time = time() - t0
y_cat_log_pred = cat_model.predict(X_test)
y_cat_pred = np.clip(np.expm1(y_cat_log_pred), 0, None)
models_trained['CatBoost'] = cat_model
predictions['CatBoost'] = y_cat_pred
print(f"     ✅ Done in {cat_time:.2f}s")

# =====================================================================
# 6. MODEL EVALUATION
# =====================================================================
print("\n[5/6] Evaluating Models...")

results = {}
for name, y_pred in predictions.items():
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape}

print("\n" + "=" * 80)
print("📊 RESULTS")
print("=" * 80)

for name in ['LightGBM', 'XGBoost', 'CatBoost']:
    r = results[name]
    print(f"\n{name}:")
    print(f"  RMSE:  {r['RMSE']/1e9:.2f}B VND")
    print(f"  MAE:   {r['MAE']/1e9:.2f}B VND")
    print(f"  R²:    {r['R²']:.4f}")
    print(f"  MAPE:  {r['MAPE']:.2f}% 🎯")

best_model = min(results.items(), key=lambda x: x[1]['MAPE'])
best_name, best_results = best_model

print("\n" + "=" * 80)
print(f"🏆 BEST MODEL: {best_name}")
print(f"   MAPE: {best_results['MAPE']:.2f}%")
print(f"   Target (< 10%): {'✅ ACHIEVED!' if best_results['MAPE'] < 10 else '⚠️ Keep improving'}")
print("=" * 80)

# Save models
print("\n[6/6] Saving Models...")
model_dir = Path("models/save_models")
model_dir.mkdir(parents=True, exist_ok=True)

for name, model in models_trained.items():
    path = model_dir / f"{name.lower()}_model.pkl"
    joblib.dump(model, path)
    print(f"  ✅ {name} → {path}")

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv(model_dir / "model_results.csv")
print(f"  ✅ Results → {model_dir / 'model_results.csv'}")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)
print("\n📈 Next Steps:")
print("  1. Build BI Dashboard")
print("  2. Deploy Web Application")
print("  3. Write Research Paper")
print("\n" + "=" * 80)
