"""
Geocode addresses using fast local mapping + cache (avoid slow API calls).
Fallback to Nominatim for unknown locations.
"""
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import re
from pathlib import Path
import subprocess

LOCALITY_FILE = Path(__file__).parent.parent.parent / "data" / "localities.csv"

# City centers - used for distance_to_center calculation only
HCM_CENTER = (10.7769, 106.7009)
HN_CENTER = (21.0285, 105.8542)

geocode_cache = {}


def load_cache_from_csv():
    """Load geocoding cache from localities.csv"""
    global geocode_cache
    if LOCALITY_FILE.exists():
        try:
            df = pd.read_csv(LOCALITY_FILE)
            # Index by various combinations for cache lookup
            for _, row in df.iterrows():
                old_address = str(row.get('old_address', '')).lower().strip()
                street = str(row.get('street', '')).lower().strip()
                locality = str(row.get('locality', '')).lower().strip()
                region = str(row.get('region', '')).lower().strip()
                lat = row['lat']
                lon = row['lon']

                if pd.isna(lat) or pd.isna(lon):
                    continue

                # Apply same cleaning as geocode_with_fallback
                old_address = re.sub(r"\s*\(cũ\)", "", old_address)
                old_address = re.sub(r"\s+", " ", old_address).strip()

                # Index by meaningful address combinations only
                if old_address:
                    geocode_cache[(old_address,)] = (lat, lon)
                if street and locality and region:
                    geocode_cache[(street, locality, region)] = (lat, lon)
                if locality and region:
                    geocode_cache[(locality, region)] = (lat, lon)
                if locality:
                    geocode_cache[(locality,)] = (lat, lon)

            print(f"  ✓ Loaded {len(geocode_cache)} geocoding cache keys from {len(df)} records")
        except Exception as e:
            print(f"  Warning: Failed to load localities.csv: {e}")
    else:
        print(f"  Note: {LOCALITY_FILE} not found (will be created on first geocoding)")


def save_coordinate_to_localities(lat, lon, street="", locality="", region="", old_address=""):
    """Append a single coordinate to localities.csv immediately"""
    if pd.isna(lat) or pd.isna(lon):
        return

    try:
        LOCALITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        new_row = pd.DataFrame([{
            'street': street,
            'locality': locality,
            'region': region,
            'old_address': old_address,
            'lat': lat,
            'lon': lon
        }])

        if LOCALITY_FILE.exists():
            df = pd.read_csv(LOCALITY_FILE)
            # Avoid duplicates: check if this lat/lon already exists
            if not ((df['lat'].round(4) == round(lat, 4)) & (df['lon'].round(4) == round(lon, 4))).any():
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(LOCALITY_FILE, index=False)
        else:
            new_row.to_csv(LOCALITY_FILE, index=False)
    except Exception as e:
        pass  # Silent fail



def append_to_localities_csv(lat, lon, features=None):
    """Append POI features to localities.csv by lat/lon coordinate"""
    if pd.isna(lat) or pd.isna(lon):
        return

    try:
        LOCALITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        new_row_data = {
            'lat': lat,
            'lon': lon
        }

        # Add features if provided
        if features:
            new_row_data.update(features)

        new_row = pd.DataFrame([new_row_data])

        if LOCALITY_FILE.exists():
            df = pd.read_csv(LOCALITY_FILE)
            # Ensure all columns are present
            for col in new_row_data.keys():
                if col not in df.columns:
                    df[col] = None
            # Avoid duplicates: check if this lat/lon already exists
            if not ((df['lat'].round(4) == round(lat, 4)) & (df['lon'].round(4) == round(lon, 4))).any():
                df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row

        df.to_csv(LOCALITY_FILE, index=False)
    except Exception as e:
        pass  # Silent fail - don't break pipeline if CSV update fails


# Load data on import
load_cache_from_csv()


def geocode_with_fallback(row):
    """
    Geocode with priority: cache > Nominatim API > region center.
    Check cache FIRST with proper keys, then call API.
    """
    old_address = str(row.get("old_address", "")).lower().strip()
    street = str(row.get("street", "")).lower().strip()
    locality = str(row.get("locality", "")).lower().strip()
    region = str(row.get("region", "")).lower().strip()

    # Clean address
    old_address = re.sub(r"\s*\(cũ\)", "", old_address)
    old_address = re.sub(r"\s+", " ", old_address).strip()

    # Build candidates: (cache_key, api_query, label)
    # Only match meaningful address combinations, not individual components
    candidates = []
    if old_address:
        candidates.append(((old_address,), f"{old_address}, Vietnam", f"Address: {old_address}"))
    if street and locality and region:
        candidates.append(((street, locality, region), f"{street}, {locality}, {region}, Vietnam", f"Street: {street}, {locality}, {region}"))
    if locality and region:
        candidates.append(((locality, region), f"{locality}, {region}, Vietnam", f"Locality: {locality}, {region}"))

    # Debug: show what we're looking for
    for i, (cache_key, api_query, label) in enumerate(candidates):
        # CHECK CACHE FIRST
        if cache_key in geocode_cache:
            lat, lon = geocode_cache[cache_key]
            print(f"    Cache HIT {i+1}: {repr(cache_key)}")
            return pd.Series([lat, lon, f"Cache: {label}"])
        else:
            print(f"    Cache MISS {i+1}: {repr(cache_key)} (cache has {len(geocode_cache)} keys)")

    # If not in cache, try API calls in order
    for cache_key, api_query, label in candidates:
        attempt = 0
        while True:
            try:
                time.sleep(3)  # 3s per request to avoid rate limit
                geolocator = Nominatim(user_agent="housing_project", timeout=10)
                location = geolocator.geocode(api_query, timeout=10)
                if location:
                    lat = location.latitude
                    lon = location.longitude
                    geocode_cache[cache_key] = (lat, lon)
                    save_coordinate_to_localities(lat, lon, street, locality, region, old_address)
                    return pd.Series([lat, lon, f"API: {label}"])
                else:
                    break  # No result found, try next candidate
            except Exception as e:
                if "429" in str(e):
                    # Rate limited: keep retrying with exponential backoff
                    print(run(["warp-cli", "disconnect"]))
                    print(run(["warp-cli", "connect"]))
                    attempt += 1
                    wait = attempt
                    print(f"    Rate limited (attempt {attempt}), waiting {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    # Other error: try next candidate
                    break

    # No fallback - return NaN if can't geocode, will be dropped later
    return pd.Series([None, None, None])

def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()

def add_coordinates(df):
    """Add coordinates using cache + Nominatim API (saves to localities.csv)"""
    if "lat" in df.columns and "lon" in df.columns:
        if df[["lat", "lon"]].notna().all(axis=1).all():
            print("  ✓ Coordinates already present")
            return df

    print(f"  Adding coordinates (cache has {len(geocode_cache)} keys)...")
    df[["lat", "lon", "matched_address"]] = df.apply(
        geocode_with_fallback,
        axis=1
    )
    return df


def distance_to_center(df):
    """Calculate distance to nearest city center"""
    df["distance_to_center_km"] = df.apply(
        lambda r: geodesic(
            (r["lat"], r["lon"]),
            HCM_CENTER if str(r["region"]).strip().lower() in ["hồ chí minh", "ho chi minh"]
            else HN_CENTER
        ).km
        if pd.notnull(r["lat"]) and pd.notnull(r["lon"])
        else None,
        axis=1
    )
    return df


