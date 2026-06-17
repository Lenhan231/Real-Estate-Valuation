import time
import requests
from geopy.distance import geodesic
import subprocess
import pandas as pd
from pathlib import Path

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
LOCALITY_FILE = Path(__file__).parent.parent.parent / "data" / "localities.csv"

cache = {}
session = requests.Session()
session.headers.update({"User-Agent": "DataProcessing/1.0"})


def _load_persistent_cache():
    """Load cached results from localities.csv to speed up subsequent runs"""
    global cache
    if not LOCALITY_FILE.exists():
        return

    try:
        df = pd.read_csv(LOCALITY_FILE)
        poi_types = [
            ('amenity', 'school', 'nearest_school_km', 'school_count_3km', 3000),
            ('amenity', 'hospital', 'nearest_hospital_km', 'hospital_count_5km', 5000),
            ('amenity', 'marketplace', 'nearest_marketplace_km', 'marketplace_count_3km', 3000),
            ('shop', 'supermarket', 'nearest_supermarket_km', 'supermarket_count_3km', 3000),
            ('shop', 'mall', 'nearest_mall_km', 'mall_count_3km', 3000),
            ('highway', 'bus_stop', 'nearest_bus_stop_km', 'bus_stop_count_1km', 1000),
        ]

        for _, row in df.iterrows():
            lat = row.get('lat')
            lon = row.get('lon')
            if pd.isna(lat) or pd.isna(lon):
                continue

            # Load cache for all POI types
            for key_type, key_value, nearest_col, count_col, radius in poi_types:
                nearest_val = row.get(nearest_col)
                count_val = row.get(count_col)

                # Cache nearest distance
                if pd.notna(nearest_val):
                    cache_key = (round(lat, 5), round(lon, 5), key_type, key_value)
                    cache[cache_key] = (nearest_val, int(count_val) if pd.notna(count_val) else 0)

                # Cache with radius
                if pd.notna(nearest_val):
                    cache_key_radius = (round(lat, 5), round(lon, 5), key_type, key_value, radius)
                    cache[cache_key_radius] = (nearest_val, int(count_val) if pd.notna(count_val) else 0)

        print(f"  ✓ Loaded {len(cache)} cached POI queries from localities.csv")
    except Exception as e:
        pass  # Silent fail


_load_persistent_cache()


def _extract_poi_coordinates(item):
    if item["type"] == "node":
        return item.get("lat"), item.get("lon")
    center = item.get("center", {})
    return center.get("lat"), center.get("lon")

def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()
    
def _query_overpass_api(query, max_retries=5, base_sleep=2):
    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(OVERPASS_URL, data=query, timeout=(10, 60))
            if response.status_code in (429, 504):
                print(run(["warp-cli", "disconnect"]))
                print(run(["warp-cli", "connect"]))
                time.sleep(base_sleep * (2 ** (attempt - 1)))
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            if attempt < max_retries:
                time.sleep(base_sleep * (2 ** (attempt - 1)))
    return None


def _calculate_poi_metrics(lat, lon, elements):
    distances = []
    for item in elements:
        poi_lat, poi_lon = _extract_poi_coordinates(item)
        if poi_lat is not None and poi_lon is not None:
            distances.append(geodesic((lat, lon), (poi_lat, poi_lon)).km)

    return (min(distances), len(distances)) if distances else (None, 0)


def get_nearest_poi(lat, lon, key, value, max_retries=5, base_sleep=2):
    cache_key = (round(lat, 5), round(lon, 5), key, value)
    if cache_key in cache:
        return cache[cache_key]

    query = f"""
    [out:json][timeout:25];
    (
      node["{key}"="{value}"](around:5000,{lat},{lon});
      way["{key}"="{value}"](around:5000,{lat},{lon});
      relation["{key}"="{value}"](around:5000,{lat},{lon});
    );
    out center;
    """

    data = _query_overpass_api(query, max_retries, base_sleep)
    if data is None:
        return (None, 0)

    nearest_dist, count = _calculate_poi_metrics(lat, lon, data.get("elements", []))
    result = (nearest_dist, count)
    cache[cache_key] = result
    return result


def count_poi_within_radius(lat, lon, key, value, around_meters, max_retries=5, base_sleep=2):
    cache_key = (round(lat, 5), round(lon, 5), key, value, around_meters)
    if cache_key in cache:
        return cache[cache_key]

    query = f"""
    [out:json][timeout:25];
    (
      node["{key}"="{value}"](around:{around_meters},{lat},{lon});
      way["{key}"="{value}"](around:{around_meters},{lat},{lon});
      relation["{key}"="{value}"](around:{around_meters},{lat},{lon});
    );
    out center;
    """

    data = _query_overpass_api(query, max_retries, base_sleep)
    if data is None:
        return (None, 0)

    nearest_dist, count = _calculate_poi_metrics(lat, lon, data.get("elements", []))
    result = (nearest_dist, count)
    cache[cache_key] = result
    return result


def get_poi_features(lat, lon, key, value, around_meters=500):
    nearest_dist, _ = get_nearest_poi(lat, lon, key, value)
    _, count = count_poi_within_radius(lat, lon, key, value, around_meters)
    return nearest_dist, count
