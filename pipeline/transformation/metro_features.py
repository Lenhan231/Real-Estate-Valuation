import time
import requests
from geopy.distance import geodesic
import subprocess
import pandas as pd
from pathlib import Path

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
LOCALITY_FILE = Path(__file__).parent.parent.parent / "data" / "localities.csv"

session = requests.Session()
session.headers.update({"User-Agent": "DataProcessing/1.0"})

cache = {}


def _load_persistent_cache():
    """Load cached metro results from localities.csv"""
    global cache
    if not LOCALITY_FILE.exists():
        return

    try:
        df = pd.read_csv(LOCALITY_FILE)
        for _, row in df.iterrows():
            lat = row.get('lat')
            lon = row.get('lon')
            if pd.isna(lat) or pd.isna(lon):
                continue

            nearest_val = row.get('nearest_metro_km')
            count_val = row.get('metro_count_5km')

            if pd.notna(nearest_val):
                # Cache nearest metro
                cache_key = (round(lat, 5), round(lon, 5))
                cache[cache_key] = (nearest_val, int(count_val) if pd.notna(count_val) else 0)

        print(f"  ✓ Loaded {len(cache)} cached metro queries from localities.csv")
    except Exception as e:
        pass  # Silent fail


_load_persistent_cache()


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()

def _extract_coordinates(item):
    if item["type"] == "node":
        return item.get("lat"), item.get("lon")
    center = item.get("center", {})
    return center.get("lat"), center.get("lon")


def _query_overpass_api(query, max_retries=10, base_sleep=5):
    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(OVERPASS_URL, data=query, timeout=(10, 60))
            if response.status_code == 504:
                print(run(["warp-cli", "disconnect"]))
                print(run(["warp-cli", "connect"]))
                time.sleep(base_sleep * attempt)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            if attempt < max_retries:
                time.sleep(base_sleep * attempt)
    return None


def _calculate_metro_distances(lat, lon, elements):
    distances = []
    for item in elements:
        poi_lat, poi_lon = _extract_coordinates(item)
        if poi_lat is not None and poi_lon is not None:
            distances.append(geodesic((lat, lon), (poi_lat, poi_lon)).km)
    return (min(distances), len(distances)) if distances else (None, 0)


def get_nearest_metro(lat, lon, max_retries=10, base_sleep=5):
    query = f"""
    [out:json][timeout:25];
    (
      node["railway"="station"]["station"="subway"](around:5000,{lat},{lon});
      way["railway"="station"]["station"="subway"](around:5000,{lat},{lon});
      relation["railway"="station"]["station"="subway"](around:5000,{lat},{lon});
    );
    out center;
    """

    data = _query_overpass_api(query, max_retries, base_sleep)
    if data is None:
        return (None, 0)

    return _calculate_metro_distances(lat, lon, data.get("elements", []))


def count_metro_within_radius(lat, lon, around_meters, max_retries=10, base_sleep=5):
    query = f"""
    [out:json][timeout:25];
    (
      node["railway"="station"]["station"="subway"](around:{around_meters},{lat},{lon});
      way["railway"="station"]["station"="subway"](around:{around_meters},{lat},{lon});
      relation["railway"="station"]["station"="subway"](around:{around_meters},{lat},{lon});
    );
    out center;
    """

    data = _query_overpass_api(query, max_retries, base_sleep)
    if data is None:
        return (None, 0)

    return _calculate_metro_distances(lat, lon, data.get("elements", []))


def get_metro_features(lat, lon, around_meters):
    # Check cache first
    cache_key = (round(lat, 5), round(lon, 5))
    if cache_key in cache:
        return cache[cache_key]

    # If not in cache, query API
    nearest_dist, _ = get_nearest_metro(lat, lon)
    _, count = count_metro_within_radius(lat, lon, around_meters)
    result = (nearest_dist, count)
    cache[cache_key] = result
    return result


if __name__ == "__main__":
    lat, lon = 10.7769, 106.7009  # HCM city center
    metro_features = get_metro_features(lat, lon, 5000)
    print(metro_features)
