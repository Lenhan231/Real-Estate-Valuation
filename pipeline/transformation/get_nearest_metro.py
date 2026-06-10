import time
import requests
from geopy.distance import geodesic

session = requests.Session()
session.headers.update({
    "User-Agent": "DataProcessing/1.0"
})

def get_nearest_metro(lat, lon, around_meters, max_retries=10, base_sleep=5):
    query = f"""
    [out:json][timeout:25];
    (
      node["railway"="station"]["station"="subway"](around:{around_meters},{lat},{lon});
      way["railway"="station"]["station"="subway"](around:{around_meters},{lat},{lon});
      relation["railway"="station"]["station"="subway"](around:{around_meters},{lat},{lon});
    );
    out center;
    """

    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(
                "https://overpass-api.de/api/interpreter",
                data=query,
                timeout=(10, 60)
            )

            if response.status_code == 504:
                sleep_time = base_sleep * attempt
                print(f"504 on attempt {attempt}, sleeping {sleep_time}s...")
                time.sleep(sleep_time)
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
                    d = geodesic((lat, lon), (poi_lat, poi_lon)).km
                    distances.append(d)

            if distances:
                return min(distances), valid_count

            return None, 0

        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            sleep_time = base_sleep * attempt
            print(f"Error on attempt {attempt}: {e}")
            print(f"Sleeping {sleep_time}s...")
            time.sleep(sleep_time)

    return None, 0