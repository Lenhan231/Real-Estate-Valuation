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

HCM_CENTER = (10.7769, 106.7009)
HN_CENTER = (21.0285, 105.8542)
CACHE_FILE = Path(__file__).parent.parent.parent / "data" / "geocode_cache.csv"
LOCALITY_FILE = Path(__file__).parent.parent.parent / "data" / "localities.csv"

geocode_cache = {}
locality_coords = {}


def load_locality_coords():
    """Load locality, street, and address coordinates from CSV file"""
    global locality_coords
    if LOCALITY_FILE.exists():
        try:
            df = pd.read_csv(LOCALITY_FILE)
            for _, row in df.iterrows():
                # Check for full address match (highest priority)
                if 'old_address' in df.columns and pd.notna(row.get('old_address')):
                    old_address = str(row['old_address']).lower().strip()
                    key = ('old_address', old_address)
                    lat = row['lat']
                    lon = row['lon']
                    locality_coords[key] = (lat, lon)
                # Street-level mapping: (street, locality, region) -> (lat, lon)
                elif 'street' in df.columns and pd.notna(row.get('street')):
                    street = str(row['street']).lower().strip()
                    locality = str(row['locality']).lower().strip()
                    region = str(row.get('region', '')).lower().strip()
                    key = (street, locality, region)
                    lat = row['lat']
                    lon = row['lon']
                    locality_coords[key] = (lat, lon)
                else:
                    # Locality/region-level mapping: (locality, region) -> (lat, lon)
                    locality = str(row['locality']).lower().strip()
                    region = str(row.get('region', '')).lower().strip()
                    key = (locality, region)
                    lat = row['lat']
                    lon = row['lon']
                    locality_coords[key] = (lat, lon)
            print(f"  ✓ Loaded {len(locality_coords)} coordinates from CSV")
        except Exception as e:
            print(f"  Warning: Failed to load localities.csv: {e}")
    else:
        print(f"  Note: {LOCALITY_FILE} not found. Using Nominatim for all locations.")


def load_cache_from_csv():
    """Load geocoding cache from CSV file"""
    global geocode_cache
    if CACHE_FILE.exists():
        try:
            df = pd.read_csv(CACHE_FILE)
            for _, row in df.iterrows():
                key = (row['locality'], row['region'])
                geocode_cache[key] = (row['lat'], row['lon'])
            print(f"  ✓ Loaded {len(geocode_cache)} cached coordinates")
        except Exception as e:
            print(f"  Warning: Failed to load cache: {e}")


def save_cache_to_csv():
    """Save geocoding cache to CSV file"""
    if not geocode_cache:
        return

    cache_data = []
    for (locality, region), (lat, lon) in geocode_cache.items():
        cache_data.append({'locality': locality, 'region': region, 'lat': lat, 'lon': lon})

    df = pd.DataFrame(cache_data)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)
    print(f"  ✓ Saved {len(geocode_cache)} coordinates to cache")


def append_to_localities_csv(street, locality, region, old_address, lat, lon):
    """Append successfully geocoded address to localities.csv"""
    try:
        LOCALITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        new_row = pd.DataFrame([{
            'street': street if street else '',
            'locality': locality,
            'region': region,
            'old_address': old_address if old_address else '',
            'lat': lat,
            'lon': lon
        }])

        if LOCALITY_FILE.exists():
            df = pd.read_csv(LOCALITY_FILE)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row

        df.to_csv(LOCALITY_FILE, index=False)
    except Exception as e:
        pass  # Silent fail - don't break geocoding if CSV update fails


# Load data on import
load_locality_coords()
load_cache_from_csv()


def get_locality_key(row):
    """Create cache key from locality + region"""
    locality = str(row.get("locality", "")).lower().strip()
    region = str(row.get("region", "")).lower().strip()
    return (locality, region)


def geocode_with_fallback(row):
    """
    Geocode with priority: exact address > street > locality > cache > Nominatim > region center.
    Automatically saves successful Nominatim results to localities.csv.
    """
    old_address = str(row.get("old_address", "")).lower().strip()
    street = str(row.get("street", "")).lower().strip()
    locality = str(row.get("locality", "")).lower().strip()
    region = str(row.get("region", "")).lower().strip()

    # Clean address
    old_address = re.sub(r"\s*\(cũ\)", "", old_address)
    old_address = re.sub(r"\s+", " ", old_address).strip()

    # Try exact old_address match from localities.csv (highest priority)
    if old_address:
        addr_key = ('old_address', old_address)
        if addr_key in locality_coords:
            lat, lon = locality_coords[addr_key]
            return pd.Series([lat, lon, f"Address: {old_address}"])

    # Try street-level CSV
    if street:
        street_key = (street, locality, region)
        if street_key in locality_coords:
            lat, lon = locality_coords[street_key]
            return pd.Series([lat, lon, f"Street: {street}"])

    # Try locality/region CSV
    locality_key = (locality, region)
    if locality_key in locality_coords:
        lat, lon = locality_coords[locality_key]
        return pd.Series([lat, lon, f"Locality: {locality}"])

    candidates = [
        (f"{locality},{region}", f"{locality}, {region}, Vietnam"),
        ((locality, region), f"{locality}, Vietnam"),
        ((region,), f"{region}, Vietnam"),
        (old_address, f"{old_address}, Vietnam")
    ]

    for cache_key, addr_str in candidates:
        # Check cache
        if cache_key in geocode_cache:
            lat, lon = geocode_cache[cache_key]
            return pd.Series([lat, lon, addr_str])

        # Try Nominatim
        try:
            geolocator = Nominatim(user_agent="housing_project")
            time.sleep(0.3)  # Rate limiting
            location = geolocator.geocode(addr_str, timeout=10)
            if location:
                lat = location.latitude
                lon = location.longitude
                result = (lat, lon)
                geocode_cache[cache_key] = result
                # Append successful geocoding to localities.csv for future use
                append_to_localities_csv(street, locality, region, old_address, lat, lon)
                return pd.Series([lat, lon, addr_str])
        except Exception:
            pass

    # Fallback to region center if all else fails
    if "hồ chí minh" in region.lower() or "ho chi minh" in region.lower():
        return pd.Series([HCM_CENTER[0], HCM_CENTER[1], f"Region: {region}"])
    elif "hà nội" in region.lower() or "ha noi" in region.lower():
        return pd.Series([HN_CENTER[0], HN_CENTER[1], f"Region: {region}"])

    return pd.Series([None, None, None])


def add_coordinates(df):
    """Add coordinates using fast local mapping + cache"""
    if "lat" in df.columns and "lon" in df.columns:
        if df[["lat", "lon"]].notna().all().any():
            print("  ✓ Coordinates already present")
            return df

    print("  Adding coordinates (using local mapping + cache)...")
    df[["lat", "lon", "matched_address"]] = df.apply(
        geocode_with_fallback,
        axis=1
    )
    save_cache_to_csv()
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


