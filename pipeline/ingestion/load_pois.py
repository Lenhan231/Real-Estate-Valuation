"""
Geocode addresses using fast local mapping + cache (avoid slow API calls).
Fallback to Nominatim for unknown locations.
"""
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import re

HCM_CENTER = (10.7769, 106.7009)
HN_CENTER = (21.0285, 105.8542)

# Fast local mapping of common Vietnamese districts/wards to approximate coordinates
# Format: "locality" or "locality,region" -> (lat, lon)
LOCALITY_COORDS = {
    "phường 1": (10.7769, 106.7009),
    "phường 2": (10.7769, 106.7109),
    "phường 3": (10.7669, 106.7009),
    "phường 4": (10.7669, 106.7109),
    "phường 5": (10.7569, 106.7009),
    "phường 6": (10.7569, 106.7109),
    "phường 7": (10.7469, 106.7009),
    "phường 8": (10.7469, 106.7109),
    "phường 9": (10.7369, 106.7009),
    "phường 10": (10.7369, 106.7109),
    "phường 11": (10.7269, 106.7009),
    "phường 12": (10.7269, 106.7109),
    "phường 13": (10.7169, 106.7009),
    "phường 14": (10.8045, 106.6985),
    "phường 15": (10.7969, 106.7109),
    "phường 16": (10.8169, 106.7009),
    "phường bình thạnh": (10.8045, 106.6985),
    "phường sài gòn": (10.7803, 106.7055),
    "quận 1": (10.7769, 106.7009),
    "quận 2": (10.7969, 106.7509),
    "quận 3": (10.7669, 106.7009),
    "quận 4": (10.7369, 106.7009),
    "quận 5": (10.7569, 106.6809),
    "quận 6": (10.7369, 106.6609),
    "quận 7": (10.7169, 106.7209),
    "quận 8": (10.7169, 106.6909),
    "quận 9": (10.7769, 106.7909),
    "quận 10": (10.7469, 106.6609),
    "quận 11": (10.7469, 106.6309),
    "quận 12": (10.8269, 106.7309),
    "quận bình tân": (10.7969, 106.6209),
    "quận bình thạnh": (10.8045, 106.6985),
    "quận gò vấp": (10.8369, 106.6909),
    "quận phú nhuận": (10.7869, 106.7509),
    "quận tân bình": (10.8169, 106.6609),
    "quận tân phú": (10.8069, 106.6009),
    "quận thủ đức": (10.8569, 106.7709),
}

geocode_cache = {}


def get_locality_key(row):
    """Create cache key from locality + region"""
    locality = str(row.get("locality", "")).lower().strip()
    region = str(row.get("region", "")).lower().strip()
    return (locality, region)


def geocode_with_fast_fallback(row):
    """
    Fast geocoding: local mapping → cache → API fallback
    Order: 1. Local map, 2. Cache, 3. Nominatim (slow)
    """
    locality = str(row.get("locality", "")).lower().strip()
    region = str(row.get("region", "")).lower().strip()
    cache_key = (locality, region)

    # 1. Check local mapping
    if locality in LOCALITY_COORDS:
        lat, lon = LOCALITY_COORDS[locality]
        return pd.Series([lat, lon, f"Local: {locality}"])

    # 2. Check cache
    if cache_key in geocode_cache:
        lat, lon, addr = geocode_cache[cache_key]
        return pd.Series([lat, lon, addr])

    # 3. Try Nominatim (only for truly unknown locations)
    try:
        geolocator = Nominatim(user_agent="housing_project")
        time.sleep(0.2)  # Minimal delay
        addr_str = f"{locality}, {region}, Vietnam"
        location = geolocator.geocode(addr_str, timeout=5)
        if location:
            result = (location.latitude, location.longitude, addr_str)
            geocode_cache[cache_key] = result
            return pd.Series([location.latitude, location.longitude, addr_str])
    except Exception:
        pass

    # 4. Fallback: use region center
    if "hồ chí minh" in region.lower() or "ho chi minh" in region.lower():
        return pd.Series([HCM_CENTER[0], HCM_CENTER[1], f"Region center: {region}"])
    elif "hà nội" in region.lower() or "ha noi" in region.lower():
        return pd.Series([HN_CENTER[0], HN_CENTER[1], f"Region center: {region}"])

    return pd.Series([None, None, None])


def add_coordinates(df):
    """Add coordinates using fast local mapping + cache"""
    if "lat" in df.columns and "lon" in df.columns:
        if df[["lat", "lon"]].notna().all().any():
            print("  ✓ Coordinates already present")
            return df

    print("  Adding coordinates (using local mapping + cache)...")
    df[["lat", "lon", "matched_address"]] = df.apply(
        geocode_with_fast_fallback,
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


