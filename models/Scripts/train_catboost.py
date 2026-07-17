"""
CatBoost Training Pipeline with Weights & Biases Tracking
=========================================================
Trains a CatBoost regressor to predict house prices 
from the cleaned feature datasets.
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import wandb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Models" / "data"
MODEL_DIR = PROJECT_ROOT / "Models"

WANDB_PROJECT = "real-estate-valuation"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def plot_feature_importance(model, feature_names, save_path):
    # Only for LGBM
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 8))
    plt.title("Top 20 Feature Importances (LGBM)")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path

def plot_pred_vs_actual(y_true, y_pred, save_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, edgecolors='none', s=15)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price (VND)')
    plt.ylabel('Predicted Price (VND)')
    plt.title('Predicted vs Actual Prices')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path

def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    df = df.copy()

    if 'price_vnd' in df.columns:
        df = df.dropna(subset=['price_vnd'])
        price_b = df['price_vnd'] / 1e9
        # ponytail: slight pruning as requested (2B to 50B) to drop extremes
        df = df[
            (price_b >= 2.0) & (price_b <= 50.0) &
            (df['area_m2'].isna() | df['area_m2'].between(15, 500) if 'area_m2' in df.columns else True)
        ]
        if 'area_m2' in df.columns:
            price_sqm = df['price_vnd'] / 1e6 / df['area_m2']
            df = df[price_sqm.isna() | ((price_sqm >= 30) & (price_sqm <= 800))]
            
    if "post_day" in df.columns:
        post_day_dt = pd.to_datetime(df["post_day"])
        df["post_day_year"] = post_day_dt.dt.year
        df["post_day_month"] = post_day_dt.dt.month
        df["post_day_day"] = post_day_dt.dt.day
        
    if "locality_square" in df.columns:
        def parse_locality_square(col: pd.Series) -> pd.Series:
            def parse_arr(x):
                if pd.isna(x): return np.nan
                try:
                    return float(json.loads(x.replace("'", '"'))[0])
                except:
                    return np.nan
            return col.apply(parse_arr)
        df["locality_square"] = parse_locality_square(df["locality_square"])

    if 'width_m' in df.columns and 'length_m' in df.columns:
        df['perimeter_m'] = (df['width_m'] + df['length_m']) * 2
        df['shape_ratio'] = (df['width_m'] + 0.1) / (df['length_m'] + 0.1)
    if 'area_m2' in df.columns and 'num_floors' in df.columns:
        df['area_x_floors'] = df['area_m2'] * df['num_floors']
    if 'area_m2' in df.columns and 'num_bedrooms' in df.columns:
        df['area_x_bedrooms'] = df['area_m2'] * df['num_bedrooms']
    if 'area_m2' in df.columns and 'num_bedrooms' in df.columns:
        df['area_per_bedroom'] = df['area_m2'] / (df['num_bedrooms'] + 1)
    if 'distance_to_center_km' in df.columns and 'area_m2' in df.columns:
        df['distance_vs_area'] = df['distance_to_center_km'] / (df['area_m2'] + 1)
    if 'area_m2' in df.columns:
        df['log_area'] = np.log1p(df['area_m2'])
    if 'distance_to_center_km' in df.columns:
        df['log_distance_to_center'] = np.log1p(df['distance_to_center_km'])
    if 'locality_population_density' in df.columns:
        df['log_population_density'] = np.log1p(df['locality_population_density'])
        
    df['location_score'] = (
        (10 / (df['distance_to_center_km'] + 1)) * 2.0 +
        (10 / (df['nearest_school_km'] + 1)) * 1.5 +
        (10 / (df['nearest_hospital_km'] + 1)) * 1.5 +
        (10 / (df['nearest_mall_km'] + 1)) * 1.0
    ) if 'distance_to_center_km' in df.columns else 0

    df['amenity_score'] = (
        df.get('school_count_3km', 0) * 1.0 +
        df.get('hospital_count_5km', 0) * 1.5 +
        df.get('supermarket_count_3km', 0) * 1.0 +
        df.get('mall_count_3km', 0) * 2.0 +
        df.get('metro_count_5km', 0) * 3.0
    )
    df['interaction_loc_amenity'] = df['location_score'] * df['amenity_score']
    
    amenities = ['school_count_3km', 'hospital_count_5km', 'marketplace_count_3km', 'supermarket_count_3km', 'mall_count_3km', 'bus_stop_count_1km', 'metro_count_5km']
    df['nearby_amenities'] = df[[c for c in amenities if c in df.columns]].sum(axis=1)

    text_cols = ['description', 'title']
    for col in text_cols:
        if col in df.columns:
            lower = df[col].astype(str).str.lower()
            df['is_hem_xe_hoi'] = lower.str.contains('hẻm xe hơi|hxh|ô tô|xe hơi|mặt ngõ').astype(int)
            df['is_mat_tien'] = lower.str.contains('mặt tiền|mặt phố').astype(int)
            df['is_no_hau'] = lower.str.contains('nở hậu').astype(int)
            df['has_noi_that'] = lower.str.contains('nội thất|full|đầy đủ').astype(int)
            df['is_gap'] = lower.str.contains('gấp|giảm giá|cần bán').astype(int)
            df['is_kinh_doanh'] = lower.str.contains('kinh doanh|cho thuê|thu nhập').astype(int)

    for col in ['nearest_metro_km', 'nearest_mall_km', 'nearest_supermarket_km']:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(999.0)

    for col in ['width_m', 'length_m']:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(df[col].median())

    drop_cols = ["id", "price_vnd", "url", "link", "title", "post_day", "description",
                 "street", "ward", "district", "locality", "region", "street_n", "locality_n",
                 "matched_address", "old_address", "lat", "lon",  # high-cardinality / leaky
                 "listing_id", "owner_listing_bin"]  # not available at inference time
    features_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    cat_cols = features_df.select_dtypes(include=['object', 'category']).columns
    features_df = pd.get_dummies(features_df, columns=cat_cols, dummy_na=True, drop_first=False)

    metadata = {
        "n_features": features_df.shape[1],
        "features": list(features_df.columns),
        "label_encoders": {},
    }
    
    import re
    # Only remove characters that break LightGBM (JSON chars and spaces)
    features_df = features_df.rename(columns=lambda x: re.sub(r'[\[\]{},"\' :]+', '_', str(x)))
    
    # Handle any remaining duplicates by appending a suffix
    if features_df.columns.duplicated().any():
        cols = pd.Series(features_df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        features_df.columns = cols
        
    metadata["features"] = list(features_df.columns)
    
    return features_df, df["price_vnd"], metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cleaned", help="Name of the dataset to use.")
    parser.add_argument("--data-source", type=str, choices=["supabase", "local"], default="local", help="Source of the data")
    args = parser.parse_args()

    dataset_label = args.dataset
    data_source = args.data_source
    csv_path = DATA_DIR / f"alonhadat_features_{dataset_label}.csv"
    
    print(f"============================================================")
    print(f"  Ensemble Training (LGBM+CatBoost) — Dataset: {dataset_label}")
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

    if 'locality' in df.columns:
        locality_train = df.loc[train_idx, 'locality']
        locality_test = df.loc[test_idx, 'locality']
        locality_price_map = df.loc[train_idx].groupby('locality')['price_vnd'].median()
        locality_sqm_map = df.loc[train_idx].groupby('locality').apply(lambda x: (x['price_vnd'] / (x['area_m2'] + 1)).median())
        global_median = float(y_train.median())
        global_sqm = float((df.loc[train_idx, 'price_vnd'] / (df.loc[train_idx, 'area_m2'] + 1)).median())
        
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train['locality_price_median'] = locality_train.map(locality_price_map).fillna(global_median).values
        X_test['locality_price_median'] = locality_test.map(locality_price_map).fillna(global_median).values
        X_train['price_per_sqm_market'] = locality_train.map(locality_sqm_map).fillna(global_sqm).values
        X_test['price_per_sqm_market'] = locality_test.map(locality_sqm_map).fillna(global_sqm).values

    drop_text = [c for c in ['locality', 'description'] if c in X_train.columns]
    if drop_text:
        X_train = X_train.drop(columns=drop_text)
        X_test = X_test.drop(columns=drop_text)

    print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    # Update meta['features'] with the newly added locality columns
    meta["features"] = list(X_train.columns)
    feature_names = meta["features"]

    # ------------------------------------------------------------------
    # 4. Train LightGBM & CatBoost (6-Bucket Ensemble Models)
    # ------------------------------------------------------------------
    print("[4/5] Training CatBoost Model...")
    from catboost import CatBoostRegressor
    import time
    import joblib

    cb_params = {
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.03,
        "loss_function": "RMSE",
        "verbose": 0,
        "random_seed": 42
    }
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    
    print(f"  Training single CatBoost model (Train: {len(X_train)}, Test: {len(X_test)})...")
    
    model = CatBoostRegressor(**cb_params)
    eval_set = (X_test, y_log_test) if len(X_test) > 0 else None
    
    model.fit(X_train, y_log_train, eval_set=eval_set)
    
    if len(X_test) > 0:
        y_log_pred_b = model.predict(X_test)
        
        global_y_log_pred = y_log_pred_b
        global_y_log_test = y_log_test.values
        
        y_pred_b = np.expm1(y_log_pred_b)
        y_pred_b = np.clip(y_pred_b, 0, None)
        
        global_y_pred = y_pred_b
        global_y_test = np.expm1(y_log_test.values)
        
    model_path = MODEL_DIR / f"catboost_{dataset_label}_{data_source}.pkl"
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

    plot_dir = MODEL_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if True:
        fi_path = plot_feature_importance(model, feature_names, plot_dir / f"feature_importance_catboost_{dataset_label}.png")
    
    pva_path = plot_pred_vs_actual(global_y_test, global_y_pred, plot_dir / f"pred_vs_actual_{dataset_label}_global.png")
    print(f"  Plots saved to {plot_dir}")

    print("  Logging to W&B...")
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"catboost-{dataset_label}-{data_source}",
        config={
            "dataset": dataset_label,
            "data_source": data_source,
            "n_samples_train": len(global_y_log_pred),
            "n_samples_test": len(global_y_test),
            "n_features": meta["n_features"],
            "features": meta["features"],
            "target_transform": "log1p",
            "label_encoders": meta["label_encoders"],
            **cb_params,
        },
    )

    wandb.log(metrics)
    wandb.log({
        "pred_vs_actual": wandb.Image(str(pva_path)),
    })
    if True:
        wandb.log({"feature_importance": wandb.Image(str(fi_path))})

    wandb.finish()
    print("\n  Done! Check your W&B dashboard for results. Models saved to models/.")

if __name__ == "__main__":
    main()
