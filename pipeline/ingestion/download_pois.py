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

# Geographic bounds (south,west,north,east) in decimal degrees
# Thành phố Hồ Chí Minh: 10°10' – 10°38' N, 106°22' – 106°54' E
HCM_BBOX = "10.167,106.367,10.633,106.900"
# Hà Nội: 20°34' – 21°18' N, 105°17' – 106°02' E
HN_BBOX = "20.567,105.283,21.300,106.033"


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
    import shutil

    print(f"Saving POI data to {DATA_DIR}\n")

    # Backup old data first
    backup_dir = DATA_DIR.parent / "pois_backup"
    if DATA_DIR.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(DATA_DIR, backup_dir)
        print(f"✓ Backed up old POI data to {backup_dir}\n")

    # Try downloading fresh data
    print("=== Ho Chi Minh City ===")
    download_all_pois(bbox=HCM_BBOX)
    download_metro(bbox=HCM_BBOX)

    # Optionally download for HN
    # print("\n=== Ha Noi ===")
    # download_all_pois(bbox=HN_BBOX)
    # download_metro(bbox=HN_BBOX)

    # Check if download was successful (at least 100 schools)
    schools_file = DATA_DIR / "schools.parquet"
    if schools_file.exists():
        df = pd.read_parquet(schools_file)
        if len(df) < 100:
            print(f"\n⚠️  WARNING: Only {len(df)} schools downloaded (expected 500+)")
            print("API might be rate-limited. Rolling back to old data...")
            shutil.rmtree(DATA_DIR)
            shutil.copytree(backup_dir, DATA_DIR)
            print("✓ Restored old POI data from backup")
        else:
            print(f"\n✓ Download successful: {len(df)} schools")

    print("\nDone!")
