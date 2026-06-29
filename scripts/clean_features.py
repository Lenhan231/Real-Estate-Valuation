import pandas as pd
import numpy as np
from pathlib import Path


def iqr_filter(df: pd.DataFrame, cols: list[str], multiplier: float = 3.0) -> pd.DataFrame:
    """
    Remove rows where the given columns fall outside IQR fences.
    Applies fences computed from the input df's own distribution (per-segment safe).
    """
    mask = pd.Series(True, index=df.index)
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask &= df[col].between(lower, upper)
    return df[mask]


def impute_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing width_m / length_m from area_m2 where possible.
    Assumes a typical 1:3 width-to-length ratio when both are missing.
    """
    df = df.copy()
    w_null = df['width_m'].isnull()
    l_null = df['length_m'].isnull()

    # Both missing → assume 1:3 ratio
    both_null = w_null & l_null
    df.loc[both_null, 'width_m'] = np.sqrt(df.loc[both_null, 'area_m2'] / 3.0)
    df.loc[both_null, 'length_m'] = df.loc[both_null, 'area_m2'] / df.loc[both_null, 'width_m']

    # Only width missing
    w_only_null = df['width_m'].isnull()
    df.loc[w_only_null, 'width_m'] = df.loc[w_only_null, 'area_m2'] / df.loc[w_only_null, 'length_m']

    # Only length missing
    l_only_null = df['length_m'].isnull()
    df.loc[l_only_null, 'length_m'] = df.loc[l_only_null, 'area_m2'] / df.loc[l_only_null, 'width_m']

    return df


def main():
    # ---------------------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------------------
    input_file  = Path("data/raw/alonhadat_features.csv")
    output_dir  = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_full = output_dir / "alonhadat_features_cleaned.csv"

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")

    # ---------------------------------------------------------------------------
    # 1. Drop semantic duplicates
    # ---------------------------------------------------------------------------
    duplicate_subset = ['lat', 'lon', 'price_vnd', 'area_m2']
    initial_len = len(df)
    df = df.drop_duplicates(subset=duplicate_subset)
    print(f"[1] Dropped {initial_len - len(df)} semantic duplicates  ->  {len(df)} rows remain.")

    # ---------------------------------------------------------------------------
    # 2. Filter out unrealistically low prices (< 100M VND)
    # ---------------------------------------------------------------------------
    before = len(df)
    df = df[df['price_vnd'] >= 100_000_000]
    print(f"[2] Dropped {before - len(df)} rows with price < 100M VND  ->  {len(df)} rows remain.")

    # ---------------------------------------------------------------------------
    # 3. Impute width_m / length_m from area_m2
    # ---------------------------------------------------------------------------
    df = impute_dimensions(df)
    print("[3] Imputed missing width_m / length_m from area_m2.")

    # ---------------------------------------------------------------------------
    # 4. Save the full cleaned dataset (global IQR applied for the baseline model)
    # ---------------------------------------------------------------------------
    before = len(df)
    df_full = iqr_filter(df, ['price_vnd', 'area_m2'], multiplier=3.0)
    print(f"[4] Global IQR x3 filter: dropped {before - len(df_full)} outliers  ->  {len(df_full)} rows.")
    print(f"    Price: {df_full['price_vnd'].min()/1e9:.1f}B - {df_full['price_vnd'].max()/1e9:.1f}B VND")
    print(f"    Area : {df_full['area_m2'].min():.0f} - {df_full['area_m2'].max():.0f} m2")
    df_full.to_csv(output_full, index=False)
    print(f"    Saved -> {output_full}  ({df_full.shape})")

    # ---------------------------------------------------------------------------
    # 5. Split by property_type and apply per-split IQR filtering
    #    Per-split IQR is more statistically correct because each segment has a
    #    very different price distribution (frontage median ~3.4x alley median).
    # ---------------------------------------------------------------------------
    print("\n[5] Splitting by property_type with per-split IQR filtering...")

    # Tighter fence for nha_mat_tien (IQR x1.5) to prune ultra-luxury outliers
    # that span 0.3B-1100B VND and destroy model learning.
    # nha_trong_hem has a narrower distribution so IQR x3 is fine.
    iqr_multipliers = {
        "nha_mat_tien":  1.5,
        "nha_trong_hem": 3.0,
    }

    property_types = df['property_type'].unique()
    split_stats = []

    for pt in sorted(property_types):
        subset = df[df['property_type'] == pt].copy()
        before_split = len(subset)

        multiplier = iqr_multipliers.get(pt, 3.0)
        subset_filtered = iqr_filter(subset, ['price_vnd', 'area_m2'], multiplier=multiplier)
        dropped = before_split - len(subset_filtered)

        slug = pt.replace(" ", "_")
        out_path = output_dir / f"split_{slug}.csv"
        subset_filtered.to_csv(out_path, index=False)

        p_min = subset_filtered['price_vnd'].min() / 1e9
        p_max = subset_filtered['price_vnd'].max() / 1e9
        a_min = subset_filtered['area_m2'].min()
        a_max = subset_filtered['area_m2'].max()

        print(f"    [{pt}]  (IQR x{multiplier})")
        print(f"      Rows   : {before_split} -> {len(subset_filtered)} (dropped {dropped} outliers)")
        print(f"      Price  : {p_min:.1f}B - {p_max:.1f}B VND")
        print(f"      Area   : {a_min:.0f} - {a_max:.0f} m2")
        print(f"      Saved  -> {out_path}")

        split_stats.append({
            "property_type": pt,
            "multiplier": multiplier,
            "rows_before": before_split,
            "rows_after": len(subset_filtered),
            "dropped": dropped,
        })

    print("\n===== Summary =====")
    print(f"  Full cleaned dataset : {len(df_full):>5} rows  ->  {output_full.name}")
    for s in split_stats:
        slug = s['property_type'].replace(" ", "_")
        print(f"  split_{slug:<18}: {s['rows_after']:>5} rows")
    print("\nData cleaning completed successfully.")


if __name__ == "__main__":
    main()
