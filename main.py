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
import argparse

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
from pipeline.supabase_handler import push_csv_to_supabase
from scaper.Alonhadat.scheduling import crawl_list_pages
from scaper.Alonhadat.link_to_details import link_to_detail
from pathlib import Path

OUTPUT_FILE = Path("data") / "processed" / "alonhadat_features.csv"
DETAILS_FILE = Path("data") / "raw" / "alonhadat_details.csv"
LISTINGS_FILE = Path("data") / "raw" / "alonhadat_listings.csv"
CLEAN_FILE = Path("data") / "processed" / "alonhadat_cleaned.csv"

BATCH_SIZE = 2  # Process 2 records at a time (shows checkpoint clearly)


FEATURE_COLS = [
    'nearest_school_km', 'school_count_3km',
    'nearest_hospital_km', 'hospital_count_5km',
    'nearest_marketplace_km', 'marketplace_count_3km',
    'nearest_supermarket_km', 'supermarket_count_3km',
    'nearest_mall_km', 'mall_count_3km',
    'nearest_bus_stop_km', 'bus_stop_count_1km',
    'nearest_metro_km', 'metro_count_5km'
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--start-page",
    type=int,
    default=1,
)

parser.add_argument(
    "--end-page",
    type=int,
    default=50,
)

args = parser.parse_args()

def process_batch(batch_df):
    """
    Process batch: compute POI features from parquet files, fill NaN values, return
    """
    batch_df = batch_df.copy()

    # Drop rows with missing coordinates BEFORE computing features
    batch_df_before = len(batch_df)
    batch_df = batch_df.dropna(subset=['lat', 'lon'], how='any')
    rows_dropped = batch_df_before - len(batch_df)

    if len(batch_df) == 0:
        return batch_df, 0, rows_dropped

    # Add POI features from parquet files using BallTree
    batch_df = get_additional_features(batch_df)

    # Fill NaN count values with 0 (no POIs found within radius)
    for col in FEATURE_COLS:
        if 'count' in col:
            batch_df[col] = batch_df[col].fillna(0)

    return batch_df, len(batch_df), rows_dropped


def main():
    t0 = time.time()

    print("=" * 60)
    print("HOUSE PRICE PREDICTION - ETL PIPELINE")
    print("=" * 60 + "\n")

    # Stage 1: Load & Clean
    
    print("[1/5] Loading raw data...")
    crawl_list_pages(start_page=args.start_page, end_page=args.end_page)
    link_to_detail()
    df = pd.read_csv(DETAILS_FILE)
    print(f"      Loaded {len(df)} records")

    print("[2/5] Cleaning data...")
    clean_data(df)

    # Remove processed data from details.csv
    pd.DataFrame(columns=df.columns).to_csv(DETAILS_FILE, index=False)
    
    df = pd.read_csv(CLEAN_FILE)
    print(f"      ✓ Cleaned")

    # Stage 2: Add base features
    print("[3/5] Adding base features...")
    # remove old columns before merge
    df = df.drop(
        columns=["locality_square", "locality_population_density"],
        errors="ignore"
    )
    density_df = load_density()
    df = merge_density_with_alonhadat(df, density_df)
    print(f"      ✓ Merged density")

    # Stage 3: Extract geospatial features in batches
    print(f"[4/5] Extracting geospatial features (batch_size={BATCH_SIZE})...")
    t1 = time.time()

    # Load existing output if it exists (for appending)
    if OUTPUT_FILE.exists():
        df_output = pd.read_csv(OUTPUT_FILE)
        print(f"      Found {len(df_output)} existing records in output")
    else:
        df_output = None

    processed_batches = []
    total_rows_kept = 0
    total_rows_dropped = 0
    n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    all_processed_indices = []

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

        # Track processed indices
        all_processed_indices.extend(batch_indices)

        if len(batch) > 0:
            processed_batches.append(batch)

            # Append to existing output (don't overwrite)
            if df_output is not None:
                df_combined = pd.concat([df_output] + processed_batches, ignore_index=True)
            else:
                df_combined = pd.concat(processed_batches, ignore_index=True)
            df_combined.to_csv(OUTPUT_FILE, index=False)

        # Checkpoint: Write remaining rows to input file after each batch
        mask = ~df.index.isin(all_processed_indices)
        df_remaining = df[mask].reset_index(drop=True)
        df_remaining.to_csv(CLEAN_FILE, index=False)
        print(f"      [CHECKPOINT] Batch {i+1}: Removed {len(all_processed_indices)} rows, {len(df_remaining)} remain in input")

        # Progress
        elapsed_batch = time.time() - t1
        batch_pct = ((i + 1) / n_batches) * 100
        print(f"      [{i+1}/{n_batches}] {batch_pct:.1f}% | Kept: {kept}, Dropped: {dropped} | {elapsed_batch:.1f}s")

    t2 = time.time()
    batch_time = t2 - t1

    # Stage 4: Final summary
    print(f"      ✓ Features extracted in {batch_time:.2f}s\n")

    print("[5/5] Finalizing...")
    print("      Pushing data to Supabase...")
    push_csv_to_supabase(OUTPUT_FILE)

    t_total = time.time() - t0
    print(f"\n✅ Pipeline complete in {t_total:.2f}s")


if __name__ == "__main__":
    main()