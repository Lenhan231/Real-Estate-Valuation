import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Define paths
    input_file = Path("data/processed/alonhadat_features.csv")
    output_file = Path("data/processed/alonhadat_features_cleaned.csv")
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")

    # 1. Drop semantic duplicates
    duplicate_subset = ['lat', 'lon', 'price_vnd', 'area_m2']
    initial_len = len(df)
    df = df.drop_duplicates(subset=duplicate_subset)
    print(f"Dropped {initial_len - len(df)} semantic duplicates.")

    # 1.5 Filter out unrealistically low prices (e.g., < 100 million VND)
    valid_price_mask = df['price_vnd'] >= 100_000_000
    invalid_price_count = (~valid_price_mask).sum()
    df = df[valid_price_mask]
    print(f"Dropped {invalid_price_count} properties with unrealistically low prices (< 100M VND).")

    # 2. Impute width_m and length_m using area_m2
    # This recovers lost information safely instead of leaving it missing
    print("Imputing width_m and length_m using area_m2...")
    w_null = df['width_m'].isnull()
    l_null = df['length_m'].isnull()
    
    # Both missing
    both_null = w_null & l_null
    df.loc[both_null, 'width_m'] = np.sqrt(df.loc[both_null, 'area_m2'] / 3.0)
    df.loc[both_null, 'length_m'] = df.loc[both_null, 'area_m2'] / df.loc[both_null, 'width_m']
    
    # Only width missing
    w_only_null = df['width_m'].isnull()
    df.loc[w_only_null, 'width_m'] = df.loc[w_only_null, 'area_m2'] / df.loc[w_only_null, 'length_m']
    
    # Only length missing
    l_only_null = df['length_m'].isnull()
    df.loc[l_only_null, 'length_m'] = df.loc[l_only_null, 'area_m2'] / df.loc[l_only_null, 'width_m']

    # 3. Leave other features (num_floors, POI distances, etc.) as NaN
    # XGBoost and TabPFN handle NaNs natively, which is mathematically optimal
    print("Leaving remaining missing values as NaN for native model handling.")

    # Save cleaned data
    df.to_csv(output_file, index=False)
    print(f"\nSaved cleaned dataset to {output_file}")
    print(f"Cleaned dataset shape: {df.shape}")
    print("\nData cleaning completed successfully.")

if __name__ == "__main__":
    main()
