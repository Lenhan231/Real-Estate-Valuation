import pandas as pd
from pathlib import Path
from .metro_features import get_metro_features
from .poi_features import get_poi_features
from pipeline.cache_handler import get_cached_features, cache_address_features

LOCALITY_FILE = Path(__file__).parent.parent.parent / "data" /"cache"/ "localities.csv"


def get_cached_or_compute_features(street: str, locality: str, region: str,
                                   old_address: str, lat: float, lon: float,
                                   school_radius=3000, hospital_radius=5000,
                                   marketplace_radius=3000, supermarket_radius=3000,
                                   mall_radius=3000, bus_stop_radius=1000,
                                   metro_radius=5000) -> dict:
    """
    Get features from Supabase cache, or compute if not cached.

    Returns:
        Dict with all POI features
    """
    # 1. Try Supabase cache first
    cached = get_cached_features(street, locality, region)
    if cached:
        return {
            'nearest_school_km': cached.get('nearest_school_km'),
            'school_count_3km': cached.get('school_count_3km'),
            'nearest_hospital_km': cached.get('nearest_hospital_km'),
            'hospital_count_5km': cached.get('hospital_count_5km'),
            'nearest_marketplace_km': cached.get('nearest_marketplace_km'),
            'marketplace_count_3km': cached.get('marketplace_count_3km'),
            'nearest_supermarket_km': cached.get('nearest_supermarket_km'),
            'supermarket_count_3km': cached.get('supermarket_count_3km'),
            'nearest_mall_km': cached.get('nearest_mall_km'),
            'mall_count_3km': cached.get('mall_count_3km'),
            'nearest_bus_stop_km': cached.get('nearest_bus_stop_km'),
            'bus_stop_count_1km': cached.get('bus_stop_count_1km'),
            'nearest_metro_km': cached.get('nearest_metro_km'),
            'metro_count_5km': cached.get('metro_count_5km'),
        }

    # 2. If not cached, compute features
    features = {}

    # Education
    nearest_school, school_count = get_poi_features(lat, lon, "amenity", "school", school_radius)
    features['nearest_school_km'] = nearest_school
    features['school_count_3km'] = school_count

    # Hospital
    nearest_hospital, hospital_count = get_poi_features(lat, lon, "amenity", "hospital", hospital_radius)
    features['nearest_hospital_km'] = nearest_hospital
    features['hospital_count_5km'] = hospital_count

    # Marketplace
    nearest_marketplace, marketplace_count = get_poi_features(lat, lon, "shop", "supermarket", marketplace_radius)
    features['nearest_marketplace_km'] = nearest_marketplace
    features['marketplace_count_3km'] = marketplace_count

    # Supermarket
    nearest_supermarket, supermarket_count = get_poi_features(lat, lon, "shop", "supermarket", supermarket_radius)
    features['nearest_supermarket_km'] = nearest_supermarket
    features['supermarket_count_3km'] = supermarket_count

    # Mall
    nearest_mall, mall_count = get_poi_features(lat, lon, "building", "mall", mall_radius)
    features['nearest_mall_km'] = nearest_mall
    features['mall_count_3km'] = mall_count

    # Bus stops
    nearest_bus, bus_count = get_poi_features(lat, lon, "highway", "bus_stop", bus_stop_radius)
    features['nearest_bus_stop_km'] = nearest_bus
    features['bus_stop_count_1km'] = bus_count

    # Metro
    nearest_metro, metro_count = get_metro_features(lat, lon, metro_radius)
    features['nearest_metro_km'] = nearest_metro
    features['metro_count_5km'] = metro_count

    # 3. Store in Supabase cache
    cache_address_features(street, locality, region, old_address, lat, lon, features)

    return features


def get_additional_features(df, school_radius=3000, hospital_radius=5000,
                           marketplace_radius=3000, supermarket_radius=3000,
                           mall_radius=3000, bus_stop_radius=1000,
                           metro_radius=5000) -> pd.DataFrame:
    """
    Add POI features using Supabase cache (check cache first, compute if needed).

    Flow:
    1. Check Supabase cache for (street, locality, region)
    2. If found: use cached features (fast!)
    3. If not found: compute features + store in Supabase
    """
    print("   Adding POI features with Supabase caching...")

    # Extract features using cache
    features_list = []
    for idx, row in df.iterrows():
        street = row.get('street', '')
        locality = row.get('locality', '')
        region = row.get('region', '')
        old_address = row.get('old_address', '')
        lat = row.get('lat')
        lon = row.get('lon')

        if pd.isna(lat) or pd.isna(lon):
            features_list.append({
                'nearest_school_km': None,
                'school_count_3km': None,
                'nearest_hospital_km': None,
                'hospital_count_5km': None,
                'nearest_marketplace_km': None,
                'marketplace_count_3km': None,
                'nearest_supermarket_km': None,
                'supermarket_count_3km': None,
                'nearest_mall_km': None,
                'mall_count_3km': None,
                'nearest_bus_stop_km': None,
                'bus_stop_count_1km': None,
                'nearest_metro_km': None,
                'metro_count_5km': None,
            })
            continue

        # Get features (from cache or compute)
        features = get_cached_or_compute_features(
            street, locality, region, old_address, lat, lon,
            school_radius, hospital_radius, marketplace_radius,
            supermarket_radius, mall_radius, bus_stop_radius, metro_radius
        )
        features_list.append(features)

        # Progress
        if (idx + 1) % 10 == 0:
            print(f"      [{idx+1}/{len(df)}] processed")

    # Convert features list to DataFrame and merge
    features_df = pd.DataFrame(features_list)
    df = pd.concat([df, features_df], axis=1)

    print("   ✓ Features added with caching\n")
    return df
