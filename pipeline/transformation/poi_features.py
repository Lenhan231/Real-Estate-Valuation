import time
import requests
from geopy.distance import geodesic

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

cache = {}
session = requests.Session()
session.headers.update({"User-Agent": "DataProcessing/1.0"})


def _extract_poi_coordinates(item):
    if item["type"] == "node":
        return item.get("lat"), item.get("lon")
    center = item.get("center", {})
    return center.get("lat"), center.get("lon")


def _query_overpass_api(query, max_retries=5, base_sleep=2):
    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(OVERPASS_URL, data=query, timeout=(10, 60))
            if response.status_code in (429, 504):
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

# testing
if __name__ == "__main__":
    lat, lon = 10.7769, 106.7009  # HCM city center
    print(get_poi_features(lat, lon, "amenity", "school", 3000))