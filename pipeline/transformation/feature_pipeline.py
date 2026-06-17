import pandas as pd
from pathlib import Path
from .metro_features import get_metro_features
from .poi_features import get_poi_features

LOCALITY_FILE = Path(__file__).parent.parent.parent / "data" / "localities.csv"

# Feature cache loaded from localities.csv
feature_cache = {}


def load_feature_cache():
    """Load cached POI features from localities.csv"""
    global feature_cache
    if LOCALITY_FILE.exists():
        try:
            df = pd.read_csv(LOCALITY_FILE)
            for _, row in df.iterrows():
                lat = row.get('lat')
                lon = row.get('lon')
                if pd.isna(lat) or pd.isna(lon):
                    continue

                # Use rounded coords as key
                key = (round(lat, 4), round(lon, 4))

                # Extract all feature columns (not address columns)
                features = {}
                address_cols = {'street', 'locality', 'region', 'old_address', 'lat', 'lon'}
                for col in df.columns:
                    if col not in address_cols and pd.notna(row[col]):
                        features[col] = row[col]

                if features:
                    feature_cache[key] = features

            print(f"  ✓ Loaded {len(feature_cache)} cached feature sets from localities.csv")
        except Exception as e:
            print(f"  Warning: Failed to load feature cache: {e}")


def save_features_to_localities(df):
    """Save computed features to localities.csv for persistent caching"""
    if not df.shape[0]:
        return

    try:
        LOCALITY_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Select ONLY address info + computed POI/metro features (nothing else)
        address_cols = ['street', 'locality', 'region', 'old_address', 'lat', 'lon']
        feature_keywords = ['nearest_', 'school_count', 'hospital_count', 'marketplace_count',
                           'supermarket_count', 'mall_count', 'bus_stop_count', 'metro_count']

        # Get only feature columns that exist
        feature_cols = [col for col in df.columns
                       if any(kw in col for kw in feature_keywords)]

        # Save only these columns
        save_cols = [col for col in address_cols if col in df.columns] + feature_cols
        save_df = df[save_cols].copy()

        if LOCALITY_FILE.exists():
            existing_df = pd.read_csv(LOCALITY_FILE)

            # Update existing rows with new features
            for idx, row in save_df.iterrows():
                lat = row.get('lat')
                lon = row.get('lon')
                if pd.isna(lat) or pd.isna(lon):
                    continue

                # Find matching row by rounded lat/lon
                mask = (existing_df['lat'].round(4) == round(lat, 4)) & (existing_df['lon'].round(4) == round(lon, 4))
                if mask.any():
                    match_idx = existing_df[mask].index[0]
                    # Update all columns (especially features)
                    for col in save_cols:
                        existing_df.loc[match_idx, col] = row[col]

            existing_df.to_csv(LOCALITY_FILE, index=False)
        else:
            save_df.to_csv(LOCALITY_FILE, index=False)
    except Exception as e:
        print(f"Warning: Failed to save features to localities.csv: {e}")


def get_additional_features(df, school_radius=3000, hospital_radius=5000, marketplace_radius=3000, supermarket_radius=3000, mall_radius=3000, bus_stop_radius=1000, metro_radius=5000) -> pd.DataFrame:
    # Education
    df[["nearest_school_km", f"school_count_{school_radius//1000}km"]] = df.apply(
        lambda row: pd.Series(
            get_poi_features(
                row["lat"],
                row["lon"],
                "amenity",
                "school",
                school_radius
            )
        ),
        axis=1
    )

    # Healthcare
    df[["nearest_hospital_km", f"hospital_count_{hospital_radius//1000}km"]] = df.apply(
        lambda row: pd.Series(
            get_poi_features(
                row["lat"],
                row["lon"],
                "amenity",
                "hospital",
                hospital_radius
            )
        ),
        axis=1
    )

    # Marketplace
    df[["nearest_marketplace_km", f"marketplace_count_{marketplace_radius//1000}km"]] = df.apply(
        lambda row: pd.Series(
            get_poi_features(
                row["lat"],
                row["lon"],
                "amenity",
                "marketplace",
                marketplace_radius
            )
        ),
        axis=1
    )

    # Supermarket
    df[["nearest_supermarket_km", f"supermarket_count_{supermarket_radius//1000}km"]] = df.apply(
        lambda row: pd.Series(
            get_poi_features(
                row["lat"],
                row["lon"],
                "shop",
                "supermarket",
                supermarket_radius
            )
        ),
        axis=1
    )

    # Mall
    df[["nearest_mall_km", f"mall_count_{mall_radius//1000}km"]] = df.apply(
        lambda row: pd.Series(
            get_poi_features(
                row["lat"],
                row["lon"],
                "shop",
                "mall",
                mall_radius
            )
        ),
        axis=1
    )

    # Bus stop
    df[["nearest_bus_stop_km", f"bus_stop_count_{bus_stop_radius//1000}km"]] = df.apply(
        lambda row: pd.Series(
            get_poi_features(
                row["lat"],
                row["lon"],
                "highway",
                "bus_stop",
                bus_stop_radius
            )
        ),
        axis=1
    )

    # Metro
    df[["nearest_metro_km", f"metro_count_{metro_radius//1000}km"]] = df.apply(
        lambda row: pd.Series(
            get_metro_features(
                row["lat"],
                row["lon"],
                metro_radius
            )
        ),
        axis=1
    )

    # Save computed features to localities.csv for persistent caching
    save_features_to_localities(df)

    return df
