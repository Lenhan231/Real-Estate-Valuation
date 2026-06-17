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

# Geographic bounds - split into smaller areas for better API results
# Overpass API limits results per query, so we download in halves then combine
# Thành phố Hồ Chí Minh: split North/South
HCM_AREAS = [
    "10.40,106.367,10.633,106.900",   # North half
    "10.167,106.367,10.40,106.900",   # South half
]
# Hà Nội: split North/South
HN_AREAS = [
    "20.93,105.283,21.300,106.033",   # North half
    "20.567,105.283,20.93,106.033",   # South half
]


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


def download_all_pois(bbox_list=HCM_AREAS):
    """Download all POI types from multiple area chunks and combine"""
    pois_config = [
        ("schools", "amenity", "school"),
        ("hospitals", "amenity", "hospital"),
        ("marketplaces", "amenity", "marketplace"),
        ("supermarkets", "shop", "supermarket"),
        ("malls", "shop", "mall"),
        ("bus_stops", "highway", "bus_stop"),
    ]

    for poi_type, key, value in pois_config:
        all_data = []
        for area_idx, bbox in enumerate(bbox_list, 1):
            print(f"  Area {area_idx}/{len(bbox_list)}: ", end="")
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
            if data:
                df = extract_pois(data)
                all_data.append(df)
                print(f"  {len(df)} {poi_type}s")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['lat', 'lon'])
            output_file = DATA_DIR / f"{poi_type}.parquet"
            combined_df.to_parquet(output_file, index=False)
            print(f"✓ Saved {len(combined_df)} total {poi_type}s to {output_file.name}\n")


def download_metro(bbox_list=HCM_AREAS):
    """Download metro/subway stations from multiple areas and combine"""
    print("Downloading metro stations...")

    all_data = []
    for area_idx, bbox in enumerate(bbox_list, 1):
        print(f"  Area {area_idx}/{len(bbox_list)}: ", end="")
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
        if data:
            df = extract_pois(data)
            all_data.append(df)
            print(f"  {len(df)} stations")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['lat', 'lon'])
        output_file = DATA_DIR / "metro_stations.parquet"
        combined_df.to_parquet(output_file, index=False)
        print(f"✓ Saved {len(combined_df)} total metro stations to {output_file.name}\n")


if __name__ == "__main__":
    print(f"Saving POI data to {DATA_DIR}\n")

    # Download for HCM (split into 2 areas for better API results)
    print("=== Ho Chi Minh City ===")
    download_all_pois(bbox_list=HCM_AREAS)
    download_metro(bbox_list=HCM_AREAS)

    # Optionally download for HN
    # print("\n=== Ha Noi ===")
    # download_all_pois(bbox_list=HN_AREAS)
    # download_metro(bbox_list=HN_AREAS)

    print("✅ All POI data downloaded successfully!")
