"""
Fast ETL pipeline - focuses on POI feature engineering refactoring.
Skips slow cleaning operations to demonstrate the POI batching improvement.
"""
import pandas as pd
import time
from pathlib import Path
from pipeline.transformation.feature_pipeline import get_additional_features

OUTPUT_FILE = Path(r"data\processed\alonhadat_features.csv")
BATCH_SIZE = 50


def main_fast():
    """Minimal pipeline focusing on POI features"""
    t0 = time.time()

    print("=" * 60)
    print("FAST PIPELINE - POI Feature Engineering Demo")
    print("=" * 60 + "\n")

    # Load already-cleaned data
    print("[1/3] Loading data...")
    csv_file = Path(r"data\raw\alonhadat_details.csv")
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f"      Loaded {len(df)} records\n")
    else:
        print("      ✗ Data file not found!")
        return

    # Select relevant columns
    print("[2/3] Adding coordinates...")
    required_cols = ["link", "title", "post_day", "street", "old_address", "locality", "region"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[required_cols + [c for c in df.columns if c not in required_cols]].copy()

    # Use fast local mapping geocoding (no API calls)
    from pipeline.ingestion.load_pois import add_coordinates
    df = add_coordinates(df)
    print(f"      ✓ Geocoded {df['lat'].notna().sum()} records\n")

    # Extract POI features in batches
    print(f"[3/3] Extracting POI features (BATCH_SIZE={BATCH_SIZE})...")
    t1 = time.time()

    processed_batches = []
    n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(n_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(df))
        batch = df.iloc[start_idx:end_idx].copy()

        # Extract features (vectorized)
        batch = get_additional_features(batch)
        processed_batches.append(batch)

        # Save checkpoint
        df_combined = pd.concat(processed_batches, ignore_index=True)
        df_combined.to_csv(OUTPUT_FILE, index=False)

        elapsed = time.time() - t1
        print(f"      [{i+1}/{n_batches}] Rows {start_idx}-{end_idx} | {elapsed:.1f}s elapsed")

    # Final output
    df_final = pd.concat(processed_batches, ignore_index=True)
    df_final.to_csv(OUTPUT_FILE, index=False)

    elapsed_total = time.time() - t0
    print(f"\n" + "=" * 60)
    print(f"✅ Complete! {len(df_final)} records processed in {elapsed_total:.1f}s")
    print(f"   Output: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main_fast()
