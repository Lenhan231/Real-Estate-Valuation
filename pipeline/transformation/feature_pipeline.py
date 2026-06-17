import pandas as pd
import numpy as np
from pipeline.transformation.metro_features import _extractor as metro_extractor
from pipeline.transformation.poi_features import _extractor as poi_extractor
from pipeline.ingestion.load_pois import get_cached_features

def get_additional_features(df, school_radius=3000, hospital_radius=5000, marketplace_radius=3000, supermarket_radius=3000, mall_radius=3000, bus_stop_radius=1000, metro_radius=5000) -> pd.DataFrame:
    """Vectorized feature extraction using BallTree batch queries with caching support"""

    lats = df["lat"].values
    lons = df["lon"].values

    # Try to get features from cache for each row
    def get_row_features(row):
        street = str(row.get("street", "")).lower().strip() if "street" in row else ""
        locality = str(row.get("locality", "")).lower().strip()
        region = str(row.get("region", "")).lower().strip()
        old_address = str(row.get("old_address", "")).lower().strip() if "old_address" in row else ""

        # Try exact address match first
        if old_address:
            cached = get_cached_features(('old_address', old_address))
            if cached:
                return cached

        # Try street match
        if street:
            cached = get_cached_features((street, locality, region))
            if cached:
                return cached

        # Try locality match
        cached = get_cached_features((locality, region))
        if cached:
            return cached

        return None

    # Check if any rows have cached features
    cached_features_list = []
    compute_indices = []
    for idx, row in df.iterrows():
        cached = get_row_features(row)
        if cached:
            cached_features_list.append((idx, cached))
        else:
            compute_indices.append(idx)

    # Initialize feature columns
    feature_columns = {
        "nearest_school_km": np.nan,
        "school_count_3km": np.nan,
        "nearest_hospital_km": np.nan,
        "hospital_count_5km": np.nan,
        "nearest_marketplace_km": np.nan,
        "marketplace_count_3km": np.nan,
        "nearest_supermarket_km": np.nan,
        "supermarket_count_3km": np.nan,
        "nearest_mall_km": np.nan,
        "mall_count_3km": np.nan,
        "nearest_bus_stop_km": np.nan,
        "bus_stop_count_1km": np.nan,
        "nearest_metro_km": np.nan,
        "metro_count_5km": np.nan,
    }

    for col in feature_columns:
        df[col] = feature_columns[col]

    # Fill in cached features
    for idx, cached in cached_features_list:
        for col, val in cached.items():
            df.loc[idx, col] = val

    # Compute features only for rows not in cache
    if compute_indices:
        compute_lats = lats[compute_indices]
        compute_lons = lons[compute_indices]

        # Education
        df.loc[compute_indices, "nearest_school_km"] = poi_extractor.get_nearest_distances_batch(compute_lats, compute_lons, "schools")
        df.loc[compute_indices, "school_count_3km"] = poi_extractor.get_counts_batch(compute_lats, compute_lons, "schools", school_radius)

        # Healthcare
        df.loc[compute_indices, "nearest_hospital_km"] = poi_extractor.get_nearest_distances_batch(compute_lats, compute_lons, "hospitals")
        df.loc[compute_indices, "hospital_count_5km"] = poi_extractor.get_counts_batch(compute_lats, compute_lons, "hospitals", hospital_radius)

        # Marketplace
        df.loc[compute_indices, "nearest_marketplace_km"] = poi_extractor.get_nearest_distances_batch(compute_lats, compute_lons, "marketplaces")
        df.loc[compute_indices, "marketplace_count_3km"] = poi_extractor.get_counts_batch(compute_lats, compute_lons, "marketplaces", marketplace_radius)

        # Supermarket
        df.loc[compute_indices, "nearest_supermarket_km"] = poi_extractor.get_nearest_distances_batch(compute_lats, compute_lons, "supermarkets")
        df.loc[compute_indices, "supermarket_count_3km"] = poi_extractor.get_counts_batch(compute_lats, compute_lons, "supermarkets", supermarket_radius)

        # Mall
        df.loc[compute_indices, "nearest_mall_km"] = poi_extractor.get_nearest_distances_batch(compute_lats, compute_lons, "malls")
        df.loc[compute_indices, "mall_count_3km"] = poi_extractor.get_counts_batch(compute_lats, compute_lons, "malls", mall_radius)

        # Bus stop
        df.loc[compute_indices, "nearest_bus_stop_km"] = poi_extractor.get_nearest_distances_batch(compute_lats, compute_lons, "bus_stops")
        df.loc[compute_indices, "bus_stop_count_1km"] = poi_extractor.get_counts_batch(compute_lats, compute_lons, "bus_stops", bus_stop_radius)

        # Metro
        df.loc[compute_indices, "nearest_metro_km"] = metro_extractor.get_nearest_distances_batch(compute_lats, compute_lons)
        df.loc[compute_indices, "metro_count_5km"] = metro_extractor.get_counts_batch(compute_lats, compute_lons, metro_radius)

    return df