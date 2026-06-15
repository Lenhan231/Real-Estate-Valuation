"""
Geocode addresses and calculate distances to city centers.
Uses cached geocoding with exponential backoff to avoid API rate limits.
"""
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import re

HCM_CENTER = (10.7769, 106.7009)
HN_CENTER = (21.0285, 105.8542)

geocode_cache = {}
geolocator = Nominatim(user_agent="housing_project")


def geocode_with_cache(address, backoff=1.5, max_attempts=2):
    """Cached geocoding with exponential backoff"""
    if address in geocode_cache:
        return geocode_cache[address]

    for attempt in range(max_attempts):
        try:
            time.sleep(0.5)  # Reduced from 1s
            location = geolocator.geocode(address, timeout=10)
            if location:
                result = (location.latitude, location.longitude)
                geocode_cache[address] = result
                return result
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(backoff ** attempt)

    return None, None


def geocode_with_fallback(row):
    """Try multiple address formats with caching"""
    old_address = str(row["old_address"])
    old_address = re.sub(r"\s*\(cũ\)", "", old_address)
    old_address = re.sub(r"\s+", " ", old_address).strip()

    candidates = [
        f"{old_address}, Vietnam",
        f"{row['street']}, {row['locality']}, {row['region']}, Vietnam",
        f"{row['street']}, {row['region']}, Vietnam",
        f"{row['locality']}, {row['region']}, Vietnam",
        f"{row['region']}, Vietnam"
    ]

    for addr in candidates:
        lat, lon = geocode_with_cache(addr, max_attempts=1)
        if lat is not None:
            return pd.Series([lat, lon, addr])

    return pd.Series([None, None, None])


def add_coordinates(df):
    """Geocode addresses in dataframe. Skip if already has valid coordinates."""
    if "lat" in df.columns and "lon" in df.columns:
        if df[["lat", "lon"]].notna().any().all():
            print("  ✓ Lat/lon already present, skipping geocoding")
            return df

    print("  Geocoding addresses (this may take a while)...")
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


