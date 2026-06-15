"""
Metro feature extraction from pre-downloaded parquet files using BallTree.
No API calls during feature engineering - only uses local spatial indices.
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pois"


class MetroFeatureExtractor:
    def __init__(self):
        self.tree = None
        self.metros = None
        self._load_metros()

    def _load_metros(self):
        """Load metro stations and build BallTree"""
        parquet_file = DATA_DIR / "metro_stations.parquet"
        if not parquet_file.exists():
            print(f"Warning: {parquet_file} not found. Run download_pois.py first.")
            self.metros = pd.DataFrame(columns=["lat", "lon"])
            self.tree = None
            return

        self.metros = pd.read_parquet(parquet_file)

        if len(self.metros) > 0:
            coords = np.radians(self.metros[["lat", "lon"]].values)
            self.tree = BallTree(coords, metric="haversine")
        else:
            self.tree = None

    def get_nearest_distance_and_count(self, lat, lon, radius_meters=5000):
        """Get distance to nearest metro and count within radius"""
        if self.tree is None or len(self.metros) == 0:
            return None, 0

        query_point = np.radians([[lat, lon]])
        radius_km = radius_meters / 1000.0
        radius_rad = radius_km / 6371.0

        # Nearest
        distances_rad, _ = self.tree.query(query_point, k=1, return_distance=True)
        nearest_dist_km = distances_rad[0][0] * 6371.0

        # Count within radius
        count = self.tree.query_radius(query_point, r=radius_rad, return_distance=False)[0]

        return nearest_dist_km if nearest_dist_km > 0 else None, len(count)

    def get_nearest_distances_batch(self, lats, lons):
        """Vectorized: nearest metro distance for all points"""
        if self.tree is None or len(self.metros) == 0:
            return np.full(len(lats), None)

        coords = np.radians(np.column_stack([lats, lons]))
        distances_rad, _ = self.tree.query(coords, k=1, return_distance=True)
        return distances_rad[:, 0] * 6371.0

    def get_counts_batch(self, lats, lons, radius_meters=5000):
        """Vectorized: metro count within radius for all points"""
        if self.tree is None or len(self.metros) == 0:
            return np.zeros(len(lats), dtype=int)

        coords = np.radians(np.column_stack([lats, lons]))
        radius_km = radius_meters / 1000.0
        radius_rad = radius_km / 6371.0

        counts = self.tree.query_radius(
            coords, r=radius_rad, return_distance=False
        )
        return np.array([len(c) for c in counts])


# Global instance
_extractor = MetroFeatureExtractor()


def get_metro_features(lat, lon, around_meters=5000):
    """Get metro features for a single point"""
    nearest_dist, count = _extractor.get_nearest_distance_and_count(lat, lon, around_meters)
    return nearest_dist, count


if __name__ == "__main__":
    lat, lon = 10.7769, 106.7009
    nearest, count = _extractor.get_nearest_distance_and_count(lat, lon)
    print(f"Metro: nearest={nearest:.2f}km, count_5km={count}")
