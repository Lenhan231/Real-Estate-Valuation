import pandas as pd
from pathlib import Path
from .overpass_client import OverpassAPIClient


POI_TYPES = [
    ('amenity', 'school', 'nearest_school_km', 'school_count_3km', 3000),
    ('amenity', 'hospital', 'nearest_hospital_km', 'hospital_count_5km', 5000),
    ('amenity', 'marketplace', 'nearest_marketplace_km', 'marketplace_count_3km', 3000),
    ('shop', 'supermarket', 'nearest_supermarket_km', 'supermarket_count_3km', 3000),
    ('shop', 'mall', 'nearest_mall_km', 'mall_count_3km', 3000),
    ('highway', 'bus_stop', 'nearest_bus_stop_km', 'bus_stop_count_1km', 1000),
]


class POIClient(OverpassAPIClient):
    """Client for querying POI (Points of Interest) from Overpass API."""

    def _parse_cache_row(self, row, lat, lon):
        """Load POI cache from localities.csv."""
        for key_type, key_value, nearest_col, count_col, radius in POI_TYPES:
            nearest_val = row.get(nearest_col)
            count_val = row.get(count_col)

            if pd.notna(nearest_val):
                cache_key = (round(lat, 5), round(lon, 5), key_type, key_value)
                self.cache[cache_key] = (nearest_val, int(count_val) if pd.notna(count_val) else 0)

                cache_key_radius = (round(lat, 5), round(lon, 5), key_type, key_value, radius)
                self.cache[cache_key_radius] = (nearest_val, int(count_val) if pd.notna(count_val) else 0)


_client = POIClient()


def get_nearest_poi(lat, lon, key, value, max_retries=5, base_sleep=2):
    cache_key = (round(lat, 5), round(lon, 5), key, value)
    if cache_key in _client.cache:
        return _client.cache[cache_key]

    query = f"""
    [out:json][timeout:25];
    (
      node["{key}"="{value}"](around:5000,{lat},{lon});
      way["{key}"="{value}"](around:5000,{lat},{lon});
      relation["{key}"="{value}"](around:5000,{lat},{lon});
    );
    out center;
    """

    data = _client._query_overpass_api(query, max_retries, base_sleep)
    if data is None:
        return (None, 0)

    nearest_dist, count = _client._calculate_metrics(lat, lon, data.get("elements", []))
    result = (nearest_dist, count)
    _client.cache[cache_key] = result
    return result


def count_poi_within_radius(lat, lon, key, value, around_meters, max_retries=5, base_sleep=2):
    cache_key = (round(lat, 5), round(lon, 5), key, value, around_meters)
    if cache_key in _client.cache:
        return _client.cache[cache_key]

    query = f"""
    [out:json][timeout:25];
    (
      node["{key}"="{value}"](around:{around_meters},{lat},{lon});
      way["{key}"="{value}"](around:{around_meters},{lat},{lon});
      relation["{key}"="{value}"](around:{around_meters},{lat},{lon});
    );
    out center;
    """

    data = _client._query_overpass_api(query, max_retries, base_sleep)
    if data is None:
        return (None, 0)

    nearest_dist, count = _client._calculate_metrics(lat, lon, data.get("elements", []))
    result = (nearest_dist, count)
    _client.cache[cache_key] = result
    return result


def get_poi_features(lat, lon, key, value, around_meters=500):
    nearest_dist, _ = get_nearest_poi(lat, lon, key, value)
    _, count = count_poi_within_radius(lat, lon, key, value, around_meters)
    return nearest_dist, count
