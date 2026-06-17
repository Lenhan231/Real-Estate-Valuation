"""
Remove all Hà Nội (Hanoi) records from alonhadat_details.csv
Focus on Hồ Chí Minh (HCM) data only
"""
import pandas as pd
from pathlib import Path

INPUT_FILE = Path(__file__).parent.parent / "data" / "raw" / "alonhadat_details.csv"

def remove_hanoi_records():
    """Remove all Hà Nội region records, keep only HCM"""
    if not INPUT_FILE.exists():
        print(f"File not found: {INPUT_FILE}")
        return

    # Read file
    df = pd.read_csv(INPUT_FILE)
    original_count = len(df)

    # Normalize region column (lowercase for case-insensitive matching)
    df['region_lower'] = df['region'].str.lower() if 'region' in df.columns else ""

    # Filter out Hanoi variants: Hà Nội, Ha Noi, hanoi, etc.
    hanoi_variants = ['hà nội', 'ha noi', 'hà nội', 'hanoi']
    mask_hanoi = df['region_lower'].isin(hanoi_variants)

    # Keep only non-Hanoi records
    df_hcm = df[~mask_hanoi].drop('region_lower', axis=1)
    removed_count = original_count - len(df_hcm)

    # Save back
    df_hcm.to_csv(INPUT_FILE, index=False)

    print(f"✓ Filtered Hà Nội records")
    print(f"  Original: {original_count} records")
    print(f"  Removed:  {removed_count} records")
    print(f"  Remaining: {len(df_hcm)} records (HCM only)")

if __name__ == "__main__":
    remove_hanoi_records()
