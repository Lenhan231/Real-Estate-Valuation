"""
Main ETL pipeline with batch processing, caching, and checkpointing.

IMPORTANT: Before running this script for the first time, download POI data:
    python pipeline/ingestion/download_pois.py

Processing flow:
1. Load & clean all data
2. Add density & coordinates (uses geocode cache for speed)
3. Extract geospatial features in batches
4. Save incrementally to avoid data loss

Note: Geocoding cache is automatically saved to data/geocode_cache.csv
for faster processing on subsequent runs.
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
    distance_to_center,
    append_to_localities_csv
)
from pipeline.transformation.feature_pipeline import (
    get_additional_features
)

OUTPUT_FILE = Path(r"data\processed\alonhadat_features.csv")
BATCH_SIZE = 10  # Process 10 records at a time (faster processing)


def process_batch(batch_df, school_radius=3000, hospital_radius=5000,
                   marketplace_radius=3000, supermarket_radius=3000,
                  mall_radius=3000, bus_stop_radius=1000, metro_radius=5000):
    """Process single batch through feature pipeline and cache features"""
    batch_df = get_additional_features(
        batch_df,
        school_radius=school_radius,
        hospital_radius=hospital_radius,
        marketplace_radius=marketplace_radius,
        supermarket_radius=supermarket_radius,
        mall_radius=mall_radius,
        bus_stop_radius=bus_stop_radius,
        metro_radius=metro_radius
    )

    # Save computed features to localities.csv for future use
    feature_cols = [
        'nearest_school_km', 'school_count_3km',
        'nearest_hospital_km', 'hospital_count_5km',
        'nearest_marketplace_km', 'marketplace_count_3km',
        'nearest_supermarket_km', 'supermarket_count_3km',
        'nearest_mall_km', 'mall_count_3km',
        'nearest_bus_stop_km', 'bus_stop_count_1km',
        'nearest_metro_km', 'metro_count_5km'
    ]

    for _, row in batch_df.iterrows():
        street = str(row.get("street", "")).lower().strip() if "street" in batch_df.columns else ""
        locality = str(row.get("locality", "")).lower().strip()
        region = str(row.get("region", "")).lower().strip()
        old_address = str(row.get("old_address", "")).lower().strip() if "old_address" in batch_df.columns else ""
        lat = row.get("lat")
        lon = row.get("lon")

        features = {col: row[col] for col in feature_cols if col in batch_df.columns}
        append_to_localities_csv(street, locality, region, old_address, lat, lon, features=features)

    return batch_df


def main():
    t0 = time.time()

    print("=" * 60)
    print("HOUSE PRICE PREDICTION - ETL PIPELINE")
    print("=" * 60 + "\n")

    # Stage 1: Load & Clean
    print("[1/5] Loading raw data...")
    df = pd.read_csv(r"data\raw\alonhadat_details.csv")
    print(f"      Loaded {len(df)} records")

    print("[2/5] Cleaning data...")
    df = clean_data(df)
    print(f"      ✓ Cleaned")

    # Stage 2: Add base features
    print("[3/5] Adding base features...")
    density_df = load_density()
    df = merge_density_with_alonhadat(df, density_df)
    df = add_coordinates(df)
    df = distance_to_center(df)
    print(f"      ✓ Added density, coordinates, distance\n")

    # Stage 3: Extract geospatial features in batches
    print(f"[4/5] Extracting geospatial features (batch_size={BATCH_SIZE})...")
    t1 = time.time()

    processed_batches = []
    n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(n_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(df))
        batch = df.iloc[start_idx:end_idx].copy()

        # Process batch
        batch = process_batch(batch)
        processed_batches.append(batch)

        # Progress
        progress = f"      [{i+1}/{n_batches}] Processed rows {start_idx}-{end_idx}"
        elapsed_batch = time.time() - t1
        print(f"{progress} ({elapsed_batch:.1f}s)")

        # Checkpoint: save after every batch
        df_combined = pd.concat(processed_batches, ignore_index=True)
        df_combined.to_csv(OUTPUT_FILE, index=False)

    t2 = time.time()
    batch_time = t2 - t1
    print(f"      ✓ {len(df)} rows in {batch_time:.2f}s\n")

    # Stage 4: Final save
    print("[5/5] Finalizing...")
    df_final = pd.concat(processed_batches, ignore_index=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"      ✓ Saved {len(df_final)} records to {OUTPUT_FILE}\n")

    # Summary
    elapsed = time.time() - t0
    print("=" * 60)
    print(f"✅ Pipeline complete in {elapsed:.2f}s")
    print(f"   Features: {len(df_final.columns)} columns")
    print(f"   Records:  {len(df_final)} rows")
    print("=" * 60)


if __name__ == "__main__":
    main()