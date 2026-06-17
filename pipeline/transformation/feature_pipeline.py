import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from pathlib import Path

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
    """Compute POI features using BallTree on accurate parquet data"""
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
    """Compute POI features for all rows using parquet files with BallTree"""

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

    # Compute features for all rows
    computed = 0
    for idx, row in df.iterrows():
        lat, lon = row.get("lat"), row.get("lon")
        if pd.notna(lat) and pd.notna(lon):
            features = compute_poi_features(lat, lon)
            if features:
                for col, val in features.items():
                    if col in df.columns:
                        df.loc[idx, col] = val
                computed += 1

    print(f"      POI Features: {computed} rows computed from parquet files")
    return df