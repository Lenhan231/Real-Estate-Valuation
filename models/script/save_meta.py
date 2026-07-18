import pickle
import pandas as pd
from pathlib import Path
import numpy as np

# Adjust imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from train_xgboost import preprocess

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Models" / "data"
MODEL_DIR = ROOT_DIR / "Models"
READY_DATA_DIR = ROOT_DIR / "Models" / "data"

def main():
    print("Loading data...")
    csv_path = DATA_DIR / "alonhadat_features_cleaned.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")
    
    print("Running preprocess...")
    X, y, meta = preprocess(df)
    
    print("Calculating locality encodings...")
    if 'locality' in df.columns:
        locality_price_map = df.groupby('locality')['price_vnd'].median()
        global_median = float(y.median())
        
        locality_sqm_map = df.groupby('locality').apply(lambda x: (x['price_vnd'] / (x['area_m2'] + 1)).median())
        global_sqm = float((df['price_vnd'] / (df['area_m2'] + 1)).median())
        
        meta['locality_price_map'] = locality_price_map.to_dict()
        meta['locality_price_global'] = global_median
        
        meta['locality_sqm_map'] = locality_sqm_map.to_dict()
        meta['locality_sqm_global'] = global_sqm
        
        meta['features'].extend(['locality_price_median', 'price_per_sqm_market'])
        
        X['locality_price_median'] = df['locality'].map(locality_price_map).fillna(global_median).values
        X['price_per_sqm_market'] = df['locality'].map(locality_sqm_map).fillna(global_sqm).values
        
    MODEL_DIR = ROOT_DIR / "Models"

    # Save meta
    meta_path = MODEL_DIR / "ensemble_meta.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved meta to {meta_path}")
    
    # Save model_ready_data.csv
    READY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ready_csv_path = READY_DATA_DIR / "model_ready_data.csv"
    X.to_csv(ready_csv_path, index=False)
    print(f"Saved model_ready_data to {ready_csv_path}")

if __name__ == "__main__":
    main()
