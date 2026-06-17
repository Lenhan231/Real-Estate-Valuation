"""
Main ETL pipeline with address-level caching and feature extraction.

Processing flow:
1. Load & clean raw data
2. Add coordinates (geocoding with caching)
3. Calculate distance to city center
4. For each batch:
   a. Get POI/metro features (check cache first, API fallback)
   b. Drop rows with missing features
   c. Save batch to output CSV
   d. Cache features to localities.csv for future reuse

Cache priority (highest to lowest):
1. Exact old_address match → instant (no API call)
2. Street + locality + region match → instant
3. Locality + region match → instant
4. API call → cache result for next run
5. Drop row if API fails or returns NaN

Output: data/processed/alonhadat_features.csv (incrementally saved)
Cache: data/localities.csv (auto-updated with features)
"""
import pandas as pd
import time
from pathlib import Path


from pipeline.transformation.cleaning import clean_data, final_clean
from pipeline.ingestion.load_density import (
    load_density,
    merge_density_with_alonhadat
)
from pipeline.ingestion.load_pois import (
    add_coordinates,
    distance_to_center
)
from pipeline.transformation.feature_pipeline import (
    get_additional_features
)

OUTPUT_FILE = Path(r"data\processed\alonhadat_features.csv")
BATCH_SIZE = 2  # Process 2 records at a time

FEATURE_COLS = [
    'nearest_school_km', 'school_count_3km',
    'nearest_hospital_km', 'hospital_count_5km',
    'nearest_marketplace_km', 'marketplace_count_3km',
    'nearest_supermarket_km', 'supermarket_count_3km',
    'nearest_mall_km', 'mall_count_3km',
    'nearest_bus_stop_km', 'bus_stop_count_1km',
    'nearest_metro_km', 'metro_count_5km'
]


def process_batch(batch_df):
    """
    Process batch: compute POI features from parquet files, fill NaN values, return
    """
    batch_df = batch_df.copy()

    # Add POI features from parquet files using BallTree
    batch_df = get_additional_features(batch_df)

    # Fill NaN count values with 0 (no POIs found within radius)
    for col in FEATURE_COLS:
        if 'count' in col:
            batch_df[col] = batch_df[col].fillna(0)

    # Keep rows that have at least some valid coordinates
    batch_df_before = len(batch_df)
    batch_df = batch_df.dropna(subset=['lat', 'lon'], how='any')
    rows_dropped = batch_df_before - len(batch_df)

    return batch_df, len(batch_df), rows_dropped


def main():
    t0 = time.time()

    print("=" * 60)
    print("HOUSE PRICE PREDICTION - ETL PIPELINE")
    print("=" * 60 + "\n")

    # Stage 1: Load & Clean
    INPUT_FILE = Path(r"data\raw\alonhadat_details.csv")
    print("[1/5] Loading raw data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"      Loaded {len(df)} records")

    print("[2/5] Cleaning data...")
    df = clean_data(df)
    print(f"      ✓ Cleaned")

    # Stage 2: Add base features
    print("[3/5] Adding base features...")
    density_df = load_density()
    df = merge_density_with_alonhadat(df, density_df)
    print(f"      ✓ Merged density")

    # Stage 3: Extract geospatial features in batches
    print(f"[4/5] Extracting geospatial features (batch_size={BATCH_SIZE})...")
    t1 = time.time()

    processed_batches = []
    total_rows_kept = 0
    total_rows_dropped = 0
    n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    # Track processed row indices for checkpoint
    processed_indices = []

    for i in range(n_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(df))

        batch = df.iloc[start_idx:end_idx].copy()
        batch_indices = list(range(start_idx, end_idx))

        batch = add_coordinates(batch)
        batch = distance_to_center(batch)

        # Process batch: add features from cache, drop incomplete rows, save
        batch, kept, dropped = process_batch(batch)
        total_rows_kept += kept
        total_rows_dropped += dropped

        if len(batch) > 0:
            processed_batches.append(batch)

            # Save batch immediately
            df_combined = pd.concat(processed_batches, ignore_index=True)
            df_combined.to_csv(OUTPUT_FILE, index=False)

            # Checkpoint: mark these rows as processed
            processed_indices.extend(batch_indices)

        # Progress
        elapsed_batch = time.time() - t1
        batch_pct = (end_idx / len(df)) * 100
        print(f"      [{i+1}/{n_batches}] {batch_pct:.1f}% | Kept: {kept}, Dropped: {dropped} | {elapsed_batch:.1f}s")

    # Remove processed rows from input file (checkpoint)
    if processed_indices:
        df_remaining = df.drop(processed_indices, errors='ignore')
        df_remaining.to_csv(INPUT_FILE, index=False)
        print(f"\n  ✓ Checkpoint: Removed {len(processed_indices)} processed rows from input")

    t2 = time.time()
    batch_time = t2 - t1

    # Stage 4: Final summary
    print(f"      ✓ Features extracted in {batch_time:.2f}s\n")

    print("[5/5] Finalizing...")
    if processed_batches:
        df_final = pd.concat(processed_batches, ignore_index=True)

        # Keep all columns
        print(f"      Output columns: {list(df_final.columns)}")
        df_final.to_csv(OUTPUT_FILE, index=False)
        final_count = len(df_final)
        print(f"      ✓ Saved {final_count} records to {OUTPUT_FILE}\n")
    else:
        print(f"      ⚠ No records with features saved\n")
        final_count = 0

    # Summary
    elapsed = time.time() - t0
    print("=" * 60)
    print(f"✅ Pipeline complete in {elapsed:.2f}s")
    if final_count > 0:
        features_count = len(df_final.columns)
        print(f"   Records:  {final_count} rows (dropped {total_rows_dropped})")
        print(f"   Features: {features_count} columns")
    print("=" * 60)


if __name__ == "__main__":
    main()