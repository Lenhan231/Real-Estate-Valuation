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
    "region", "listing_id", "matched_address",
    "listing_type", "price_billion_vnd"
]
CAT_COLS = ["property_type", "legal_status", "direction"]



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

    # Stricter outlier bounds (from train.ipynb)
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])
        price_b = df[TARGET] / 1e9
        area = df.get('area_m2', pd.Series(dtype=float))
        df = df[
            (price_b >= 0.1) & (price_b <= 200) &
            (df['area_m2'].between(10, 2000) if 'area_m2' in df.columns else True) &
            (df['num_floors'].between(1, 20) if 'num_floors' in df.columns else True) &
            ((df['num_bedrooms'] <= 50) if 'num_bedrooms' in df.columns else True)
        ]
        # price/sqm sanity (5-3000 M/sqm)
        if 'area_m2' in df.columns:
            price_sqm = df[TARGET] / 1e6 / df['area_m2']
            df = df[(price_sqm >= 5) & (price_sqm <= 3000)]

    # Split post_day to date/month/year
    if "post_day" in df.columns:
        post_day_dt = pd.to_datetime(df["post_day"])
        df["post_day_year"] = post_day_dt.dt.year
        df["post_day_month"] = post_day_dt.dt.month
        df["post_day_day"] = post_day_dt.dt.day

    if "locality_square" in df.columns:
        df["locality_square"] = parse_locality_square(df["locality_square"])

    # --- Derived features (no leakage) ---
    if 'width_m' in df.columns and 'length_m' in df.columns:
        df['perimeter_m'] = (df['width_m'] + df['length_m']) * 2
        df['shape_ratio'] = (df['width_m'] + 0.1) / (df['length_m'] + 0.1)
    if 'area_m2' in df.columns and 'num_floors' in df.columns:
        df['area_x_floors'] = df['area_m2'] * df['num_floors']
    if 'area_m2' in df.columns and 'num_bedrooms' in df.columns:
        df['area_x_bedrooms'] = df['area_m2'] * df['num_bedrooms']
        df['area_per_bedroom'] = df['area_m2'] / (df['num_bedrooms'] + 0.1)
    if 'area_m2' in df.columns and 'distance_to_center_km' in df.columns:
        df['distance_vs_area'] = df['area_m2'] / (df['distance_to_center_km'] + 0.1)
    if 'area_m2' in df.columns:
        df['log_area'] = np.log1p(df['area_m2'])
    if 'distance_to_center_km' in df.columns:
        df['log_distance_to_center'] = np.log1p(df['distance_to_center_km'])
    if 'locality_population_density' in df.columns:
        df['log_population_density'] = np.log1p(df['locality_population_density'])

    # Location quality composite score
    loc_cols = ['nearest_school_km', 'nearest_hospital_km', 'nearest_supermarket_km', 'distance_to_center_km']
    if all(c in df.columns for c in loc_cols):
        df['location_score'] = (
            1 / (df['nearest_school_km'] + 0.5) +
            1 / (df['nearest_hospital_km'] + 0.5) +
            1 / (df['nearest_supermarket_km'] + 0.5) +
            0.5 / (df['distance_to_center_km'] + 0.5)
        )
    if 'supermarket_count_3km' in df.columns and 'school_count_3km' in df.columns:
        df['amenity_score'] = np.log1p(df['supermarket_count_3km']) + np.log1p(df['school_count_3km'])
    if 'location_score' in df.columns and 'amenity_score' in df.columns:
        df['interaction_loc_amenity'] = df['location_score'] * df['amenity_score']
    count_cols = ['school_count_3km', 'hospital_count_5km', 'supermarket_count_3km', 'mall_count_3km']
    if all(c in df.columns for c in count_cols):
        df['nearby_amenities'] = df[count_cols].sum(axis=1)

    # Missing value flags
    for col in ['nearest_metro_km', 'nearest_mall_km', 'nearest_supermarket_km', 'width_m', 'length_m']:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isna().astype(int)

    # Impute numeric NaNs with median
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # One-hot encode categoricals
    cat_present = [c for c in CAT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_present, dummy_na=True, prefix=cat_present)

    y = df[TARGET].copy()
    df = df.drop(columns=[TARGET])

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    meta = {
        "features": df.columns.tolist(),
        "n_features": len(df.columns),
        "label_encoders": {},
        "locality": None,  # populated after split
    }
    return df, y, meta, df.index


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
                
            if not csv_path.exists() and dataset_label != "supabase":
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
    print("\n[2/6] Loading data...")
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from pipeline.supabase_handler import fetch_csv_from_supabase
        df = fetch_csv_from_supabase()
        if df is not None and len(df) > 0:
            print(f"  OK: {len(df)} records from Supabase")
        else:
            raise Exception("No data returned")
    except Exception as e:
        print(f"  WARNING: Supabase failed: {e}")
        print("  Trying model_ready_data.csv...")
        try:
            df = pd.read_csv(PROJECT_ROOT / 'data/model_ready_data.csv')
            print(f"  OK: {len(df)} records from model_ready_data.csv (cached)")
        except FileNotFoundError:
            print("  Trying raw data...")
            df = pd.read_csv(PROJECT_ROOT / 'data/raw/alonhadat_listings.csv')
            print(f"  OK: {len(df)} records from raw")
            for col in ['title', 'description', 'locality', 'street', 'old_address']:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()

    if 'price_billion_vnd' not in df.columns:
        if 'price_vnd' in df.columns:
            df['price_billion_vnd'] = df['price_vnd'] / 1e9
    print(f"  Shape: {df.shape}")

    # Log direction distribution if enriched
    if "direction" in df.columns:
        dist = df["direction"].value_counts().to_dict()
        print(f"  Direction distribution: {dist}")

    # ------------------------------------------------------------------
    # 3. Preprocess
    # ------------------------------------------------------------------
    print("\n[3/6] Preprocessing...")
    X, y, meta, idx = preprocess(df)
    print(f"  After outlier filter: {len(X)} rows | Features: {meta['n_features']}")

    # Log1p transform on target
    y_log = np.log1p(y)

    # ------------------------------------------------------------------
    # 4. Train / Test split
    # ------------------------------------------------------------------
    print("\n[4/6] Splitting data (80/20)...")
    train_idx, test_idx = train_test_split(X.index, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    y_log_train, y_log_test = y_log.loc[train_idx], y_log.loc[test_idx]
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Leak-safe locality target encoding (from train set only)
    if 'locality' in df.columns:
        locality_train = df.loc[train_idx, 'locality']
        locality_test = df.loc[test_idx, 'locality']
        locality_price_map = df.loc[train_idx].groupby('locality')[TARGET].median()
        locality_sqm_map = (
            df.loc[train_idx]
            .groupby('locality')
            .apply(lambda x: (x[TARGET] / (x['area_m2'] + 1)).median())
        )
        global_median = float(y_train.median())
        global_sqm = float((df.loc[train_idx, TARGET] / (df.loc[train_idx, 'area_m2'] + 1)).median())
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train['locality_price_median'] = locality_train.map(locality_price_map).fillna(global_median).values
        X_test['locality_price_median'] = locality_test.map(locality_price_map).fillna(global_median).values
        X_train['price_per_sqm_market'] = locality_train.map(locality_sqm_map).fillna(global_sqm).values
        X_test['price_per_sqm_market'] = locality_test.map(locality_sqm_map).fillna(global_sqm).values
        print(f"  Added locality_price_median + price_per_sqm_market (leak-safe)")

    # Drop raw non-numeric columns that TabPFN can't use
    drop_text = [c for c in ['locality', 'description'] if c in X_train.columns]
    if drop_text:
        X_train = X_train.drop(columns=drop_text)
        X_test = X_test.drop(columns=drop_text)
    print(f"  Final features: {X_train.shape[1]}")



    # ------------------------------------------------------------------
    # 5. Train TabPFN — price segments: 0-5B, 5-20B, >20B
    # ------------------------------------------------------------------
    print("\n[5/6] Training TabPFN (segment models)...")
    from sklearn.ensemble import RandomForestClassifier

    BINS_VND = [0, 5e9, 20e9, float('inf')]
    seg_train = pd.cut(y_train, bins=BINS_VND, labels=[0, 1, 2]).astype(int)
    seg_test  = pd.cut(y_test,  bins=BINS_VND, labels=[0, 1, 2]).astype(int)

    # Route classifier
    rf_router = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
    rf_router.fit(X_train, seg_train)

    import time
    t0 = time.time()
    seg_models = {}
    for seg_id in [0, 1, 2]:
        mask = seg_train == seg_id
        n = mask.sum()
        print(f"  Segment {seg_id}: {n} train samples")
        if n < 30:
            continue
        MAX_SEG = 1000
        idx_seg = mask[mask].index[:MAX_SEG]
        m = TabPFNRegressor(random_state=RANDOM_STATE, ignore_pretraining_limits=True)
        m.fit(X_train.loc[idx_seg], y_log_train.loc[idx_seg])
        seg_models[seg_id] = m
    train_time = time.time() - t0
    print(f"  Training complete in {train_time:.2f}s")

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    print("\n[6/6] Evaluating...")
    seg_pred_test = rf_router.predict(X_test)
    y_log_pred = np.zeros(len(X_test))
    for seg_id, m in seg_models.items():
        mask = seg_pred_test == seg_id
        if mask.any():
            y_log_pred[mask] = m.predict(X_test[mask])

    y_pred = np.expm1(y_log_pred)
    y_pred = np.clip(y_pred, 0, None)

    # Per-segment MAPE breakdown
    for seg_id in [0, 1, 2]:
        mask = (seg_test == seg_id).values
        if mask.sum() > 0:
            seg_mape = mean_absolute_percentage_error(y_test.values[mask], y_pred[mask])
            print(f"  Segment {seg_id} MAPE: {seg_mape:.2f}% ({mask.sum()} samples)")

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
