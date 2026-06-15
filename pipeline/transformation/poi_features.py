"""
POI feature extraction from pre-downloaded parquet files using BallTree.
No API calls during feature engineering - only uses local spatial indices.
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from pathlib import Path
from geopy.distance import geodesic

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pois"


class POIFeatureExtractor:
    def __init__(self):
        self.trees = {}
        self.pois = {}
        self._load_all_pois()

    def _load_all_pois(self):
        """Load all POI parquet files and build BallTree indices"""
        poi_types = [
            "schools",
            "hospitals",
            "marketplaces",
            "supermarkets",
            "malls",
            "bus_stops",
        ]

        for poi_type in poi_types:
            parquet_file = DATA_DIR / f"{poi_type}.parquet"
            if not parquet_file.exists():
                print(f"Warning: {parquet_file} not found. Run download_pois.py first.")
                self.pois[poi_type] = pd.DataFrame(columns=["lat", "lon"])
                self.trees[poi_type] = None
                continue

            df = pd.read_parquet(parquet_file)
            self.pois[poi_type] = df

            if len(df) > 0:
                coords = np.radians(df[["lat", "lon"]].values)
                self.trees[poi_type] = BallTree(coords, metric="haversine")
            else:
                self.trees[poi_type] = None

    def get_nearest_distance_and_count(self, lat, lon, poi_type, radius_meters=500):
        """
        Get distance to nearest POI and count within radius (vectorized).
        radius_meters: search radius in meters
        """
        if self.trees[poi_type] is None or len(self.pois[poi_type]) == 0:
            return None, 0

        # Convert to radians
        query_point = np.radians([[lat, lon]])
        radius_km = radius_meters / 1000.0
        radius_rad = radius_km / 6371.0  # Earth radius in km

        # Find nearest
        distances_rad, indices = self.trees[poi_type].query(
            query_point, k=1, return_distance=True
        )
        nearest_dist_km = distances_rad[0][0] * 6371.0

        # Count within radius
        count = self.trees[poi_type].query_radius(
            query_point, r=radius_rad, return_distance=False
        )[0]

        return nearest_dist_km if nearest_dist_km > 0 else None, len(count)

    def get_nearest_distances_batch(self, lats, lons, poi_type):
        """Vectorized: get nearest distance for all points at once"""
        if self.trees[poi_type] is None or len(self.pois[poi_type]) == 0:
            return np.full(len(lats), None)

        coords = np.radians(np.column_stack([lats, lons]))
        distances_rad, _ = self.trees[poi_type].query(coords, k=1, return_distance=True)
        return distances_rad[:, 0] * 6371.0

    def get_counts_batch(self, lats, lons, poi_type, radius_meters=500):
        """Vectorized: count POIs within radius for all points at once"""
        if self.trees[poi_type] is None or len(self.pois[poi_type]) == 0:
            return np.zeros(len(lats), dtype=int)

        coords = np.radians(np.column_stack([lats, lons]))
        radius_km = radius_meters / 1000.0
        radius_rad = radius_km / 6371.0

        counts = self.trees[poi_type].query_radius(
            coords, r=radius_rad, return_distance=False
        )
        return np.array([len(c) for c in counts])


# Global instance - loaded once when module imports
_extractor = POIFeatureExtractor()


def get_poi_features(lat, lon, poi_type_key, poi_type_value=None, around_meters=500):
    """
    Legacy API for compatibility.
    Maps OSM keys to our parquet file names.
    """
    poi_mapping = {
        ("amenity", "school"): "schools",
        ("amenity", "hospital"): "hospitals",
        ("amenity", "marketplace"): "marketplaces",
        ("shop", "supermarket"): "supermarkets",
        ("shop", "mall"): "malls",
        ("highway", "bus_stop"): "bus_stops",
    }

    poi_type = poi_mapping.get((poi_type_key, poi_type_value))
    if poi_type is None:
        return None, 0

    nearest_dist, count = _extractor.get_nearest_distance_and_count(
        lat, lon, poi_type, around_meters
    )
    return nearest_dist, count


if __name__ == "__main__":
    # Test with HCM city center
    lat, lon = 10.7769, 106.7009
    print(f"Testing at ({lat}, {lon}):\n")

    for poi_type in ["schools", "hospitals", "supermarkets"]:
        nearest, count = _extractor.get_nearest_distance_and_count(lat, lon, poi_type)
        print(f"{poi_type}: nearest={nearest:.2f}km, count_500m={count}")
