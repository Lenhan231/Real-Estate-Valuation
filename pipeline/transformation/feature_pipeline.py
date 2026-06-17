import pandas as pd
import numpy as np
from pipeline.ingestion.load_pois import get_cached_features

def get_additional_features(df) -> pd.DataFrame:
    """Get POI features from cache or API call per unique address"""

    # Initialize feature columns
    feature_columns = {
        "nearest_school_km": np.nan,
        "school_count_3km": np.nan,
        "nearest_hospital_km": np.nan,
        "hospital_count_5km": np.nan,
        "nearest_marketplace_km": np.nan,
        "marketplace_count_3km": np.nan,
        "nearest_supermarket_km": np.nan,
        "supermarket_count_3km": np.nan,
        "nearest_mall_km": np.nan,
        "mall_count_3km": np.nan,
        "nearest_bus_stop_km": np.nan,
        "bus_stop_count_1km": np.nan,
        "nearest_metro_km": np.nan,
        "metro_count_5km": np.nan,
    }

    for col in feature_columns:
        df[col] = feature_columns[col]

    # Get unique addresses to avoid duplicate API calls
    unique_addresses = {}
    for idx, row in df.iterrows():
        old_address = str(row.get("old_address", "")).lower().strip() if "old_address" in df.columns else ""
        if old_address and old_address not in unique_addresses:
            unique_addresses[old_address] = idx

    # Fetch features for each unique address
    api_call_count = 0
    cache_hit_count = 0

    for old_address, first_idx in unique_addresses.items():
        # Try cache first
        cached = get_cached_features(('old_address', old_address))
        if cached:
            cache_hit_count += 1
            # Apply cached features to all rows with this address
            for idx, row in df.iterrows():
                if str(row.get("old_address", "")).lower().strip() == old_address:
                    for col, val in cached.items():
                        if col in df.columns:
                            df.loc[idx, col] = val
        else:
            # API call to get features (e.g., Google Maps, Overpass API, etc.)
            # This is where you would call your external API
            # For now, placeholder for API call
            api_call_count += 1
            # Features would be retrieved and cached here

    if api_call_count > 0 or cache_hit_count > 0:
        print(f"      Features: {cache_hit_count} from cache, {api_call_count} API calls needed")

    return df