import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from pathlib import Path
from pipeline.ingestion.load_pois import get_cached_features

# Load POI data once on import
POI_DATA = {}
POI_TREES = {}

def _load_poi_trees():
    """Load all POI parquet files and build spatial indices"""
    global POI_DATA, POI_TREES

    data_dir = Path(__file__).parent.parent.parent / "data" / "pois"
    poi_types = ["schools", "hospitals", "marketplaces", "supermarkets", "malls", "bus_stops", "metro_stations"]

    for poi_type in poi_types:
        parquet_file = data_dir / f"{poi_type}.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            POI_DATA[poi_type] = df

            # Build BallTree for spatial queries
            if len(df) > 0:
                coords = np.radians(df[["lat", "lon"]].values)
                POI_TREES[poi_type] = BallTree(coords, metric="haversine")
            else:
                POI_TREES[poi_type] = None

# Load POI data on module import
_load_poi_trees()


def compute_poi_features(lat, lon):
    """Compute POI features for a single coordinate using local parquet files"""
    if pd.isna(lat) or pd.isna(lon):
        return None

    features = {}

    # Define radius for each POI type (in meters)
    poi_config = [
        ("schools", 3000),
        ("hospitals", 5000),
        ("marketplaces", 3000),
        ("supermarkets", 3000),
        ("malls", 3000),
        ("bus_stops", 1000),
        ("metro_stations", 5000),
    ]

    query_point = np.radians([[lat, lon]])

    for poi_type, radius_meters in poi_config:
        tree = POI_TREES.get(poi_type)

        if tree is None or poi_type not in POI_DATA:
            features[f"nearest_{poi_type.rstrip('s')}_km"] = np.nan
            features[f"{poi_type.rstrip('s')}_count_{radius_meters//1000}km"] = 0
            continue

        # Find nearest POI
        radius_km = radius_meters / 1000.0
        radius_rad = radius_km / 6371.0

        distances_rad, indices = tree.query(query_point, k=1, return_distance=True)
        nearest_dist_km = distances_rad[0][0] * 6371.0

        # Count within radius
        count_indices = tree.query_radius(query_point, r=radius_rad, return_distance=False)[0]
        count = len(count_indices)

        # Store features
        feature_name = poi_type.rstrip('s')
        features[f"nearest_{feature_name}_km"] = nearest_dist_km if nearest_dist_km > 0 else np.nan
        features[f"{feature_name}_count_{radius_meters//1000}km"] = count

    return features


def get_additional_features(df) -> pd.DataFrame:
    """Get POI features from cache (by lat/lon) or compute using local parquet files"""

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

    # Get unique coordinates to avoid duplicate computations
    unique_coords = {}
    for idx, row in df.iterrows():
        lat, lon = row.get("lat"), row.get("lon")
        if pd.notna(lat) and pd.notna(lon):
            # Round to 4 decimals for cache matching (~11m precision)
            key = (round(lat, 4), round(lon, 4))
            if key not in unique_coords:
                unique_coords[key] = idx

    # Fetch features for each unique coordinate
    cache_hits = 0
    computed = 0

    for (lat_key, lon_key), first_idx in unique_coords.items():
        # Try cache first (match by rounded lat/lon)
        cached = get_cached_features((lat_key, lon_key))

        if cached:
            # Use cached features
            cache_hits += 1
            for idx, row in df.iterrows():
                if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
                    if (round(row.get("lat"), 4), round(row.get("lon"), 4)) == (lat_key, lon_key):
                        for col, val in cached.items():
                            if col in df.columns:
                                df.loc[idx, col] = val
        else:
            # Compute features from parquet files
            computed += 1
            features = compute_poi_features(lat_key, lon_key)

            if features:
                # Apply computed features to all rows with same rounded lat/lon
                for idx, row in df.iterrows():
                    if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
                        if (round(row.get("lat"), 4), round(row.get("lon"), 4)) == (lat_key, lon_key):
                            for col, val in features.items():
                                if col in df.columns:
                                    df.loc[idx, col] = val

    if cache_hits > 0 or computed > 0:
        print(f"      POI Features: {cache_hits} cached by lat/lon, {computed} computed from parquet")

    return df