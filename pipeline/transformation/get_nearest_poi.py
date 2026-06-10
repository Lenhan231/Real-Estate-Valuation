import time
import requests
from geopy.distance import geodesic

cache = {}
session = requests.Session()
session.headers.update({"User-Agent": "DataProcessing/1.0"})

def get_nearest_poi(lat, lon, key, value, around_meters, max_retries=5, base_sleep=2):
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

    url = "https://overpass-api.de/api/interpreter"

    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(url, data=query, timeout=(10, 60))

            if response.status_code in (429, 504):
                wait = base_sleep * (2 ** (attempt - 1))
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            distances = []
            valid_count = 0

            for item in data.get("elements", []):
                if item["type"] == "node":
                    poi_lat = item.get("lat")
                    poi_lon = item.get("lon")
                else:
                    center = item.get("center", {})
                    poi_lat = center.get("lat")
                    poi_lon = center.get("lon")

                if poi_lat is not None and poi_lon is not None:
                    valid_count += 1
                    distances.append(geodesic((lat, lon), (poi_lat, poi_lon)).km)

            result = (min(distances), valid_count) if distances else (None, 0)
            cache[cache_key] = result
            return result

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RequestException):
            wait = base_sleep * (2 ** (attempt - 1))
            time.sleep(wait)

    return None, 0