"""
Download POI data from Overpass API and save as parquet files.
Run this ONCE per month/quarter. Never call this during feature engineering.
"""
import requests
import pandas as pd
import time
from pathlib import Path

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pois"
DATA_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "DataProcessing/1.0"})

# Study area: HCM + 5km buffer (approximate bbox)
# Adjust bbox for your city
HCM_BBOX = "10.65,106.55,10.90,106.85"  # south,west,north,east
HN_BBOX = "20.95,105.75,21.15,105.95"


def query_overpass(query, max_retries=5, base_sleep=2):
    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(OVERPASS_URL, data=query, timeout=(10, 60))
            if response.status_code in (429, 504):
                wait = base_sleep * (2 ** (attempt - 1))
                print(f"Rate limit. Waiting {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait = base_sleep * (2 ** (attempt - 1))
                time.sleep(wait)
    return None


def extract_pois(data):
    """Convert Overpass JSON to DataFrame"""
    pois = []
    for item in data.get("elements", []):
        lat, lon = None, None

        if item.get("type") == "node":
            lat, lon = item.get("lat"), item.get("lon")
        elif "center" in item:
            lat, lon = item["center"].get("lat"), item["center"].get("lon")

        if lat is not None and lon is not None:
            tags = item.get("tags", {})
            pois.append({
                "lat": lat,
                "lon": lon,
                "name": tags.get("name", ""),
                "amenity": tags.get("amenity", ""),
            })

    return pd.DataFrame(pois)


def download_poi(poi_type, key, value, output_file, bbox=HCM_BBOX):
    """Download single POI type and save as parquet"""
    print(f"Downloading {poi_type}...")

    query = f"""
    [out:json][timeout:60];
    (
      node["{key}"="{value}"]({bbox});
      way["{key}"="{value}"]({bbox});
      relation["{key}"="{value}"]({bbox});
    );
    out center;
    """

    data = query_overpass(query)
    if data is None:
        print(f"  Failed to download {poi_type}")
        return

    df = extract_pois(data)
    df.to_parquet(output_file, index=False)
    print(f"  Saved {len(df)} {poi_type}s to {output_file.name}")
    time.sleep(2)


def download_all_pois(bbox=HCM_BBOX):
    """Download all POI types"""
    pois_config = [
        ("schools", "amenity", "school"),
        ("hospitals", "amenity", "hospital"),
        ("marketplaces", "amenity", "marketplace"),
        ("supermarkets", "shop", "supermarket"),
        ("malls", "shop", "mall"),
        ("bus_stops", "highway", "bus_stop"),
    ]

    for poi_type, key, value in pois_config:
        output_file = DATA_DIR / f"{poi_type}.parquet"
        download_poi(poi_type, key, value, output_file, bbox)


def download_metro(bbox=HCM_BBOX):
    """Download metro/subway stations"""
    print("Downloading metro stations...")

    query = f"""
    [out:json][timeout:60];
    (
      node["railway"="station"]["station"="subway"]({bbox});
      way["railway"="station"]["station"="subway"]({bbox});
      relation["railway"="station"]["station"="subway"]({bbox});
    );
    out center;
    """

    data = query_overpass(query)
    if data is None:
        print("  Failed to download metro stations")
        return

    df = extract_pois(data)
    output_file = DATA_DIR / "metro_stations.parquet"
    df.to_parquet(output_file, index=False)
    print(f"  Saved {len(df)} metro stations to {output_file.name}")


if __name__ == "__main__":
    print(f"Saving POI data to {DATA_DIR}\n")

    # Download for HCM
    print("=== Ho Chi Minh City ===")
    download_all_pois(bbox=HCM_BBOX)
    download_metro(bbox=HCM_BBOX)

    # Optionally download for HN
    # print("\n=== Ha Noi ===")
    # download_all_pois(bbox=HN_BBOX)
    # download_metro(bbox=HN_BBOX)

    print("\nDone!")
