import pandas as pd
from pathlib import Path
from .overpass_client import OverpassAPIClient


class MetroClient(OverpassAPIClient):
    """Client for querying metro stations from Overpass API."""

    def _parse_cache_row(self, row, lat, lon):
        """Load metro cache from localities.csv."""
        nearest_val = row.get('nearest_metro_km')
        count_val = row.get('metro_count_5km')

        if pd.notna(nearest_val):
            cache_key = (round(lat, 5), round(lon, 5))
            self.cache[cache_key] = (nearest_val, int(count_val) if pd.notna(count_val) else 0)


_client = MetroClient()


def get_nearest_metro(lat, lon, max_retries=10, base_sleep=5):
    query = f"""
    [out:json][timeout:25];
    (
      node["railway"="station"]["station"="subway"](around:5000,{lat},{lon});
      way["railway"="station"]["station"="subway"](around:5000,{lat},{lon});
      relation["railway"="station"]["station"="subway"](around:5000,{lat},{lon});
    );
    out center;
    """

    data = _client._query_overpass_api(query, max_retries, base_sleep)
    if data is None:
        return (None, 0)

    return _client._calculate_metrics(lat, lon, data.get("elements", []))


def count_metro_within_radius(lat, lon, around_meters, max_retries=10, base_sleep=5):
    query = f"""
    [out:json][timeout:25];
    (
      node["railway"="station"]["station"="subway"](around:{around_meters},{lat},{lon});
      way["railway"="station"]["station"="subway"](around:{around_meters},{lat},{lon});
      relation["railway"="station"]["station"="subway"](around:{around_meters},{lat},{lon});
    );
    out center;
    """

    data = _client._query_overpass_api(query, max_retries, base_sleep)
    if data is None:
        return (None, 0)

    return _client._calculate_metrics(lat, lon, data.get("elements", []))


def get_metro_features(lat, lon, around_meters):
    cache_key = (round(lat, 5), round(lon, 5))
    if cache_key in _client.cache:
        return _client.cache[cache_key]

    nearest_dist, _ = get_nearest_metro(lat, lon)
    _, count = count_metro_within_radius(lat, lon, around_meters)
    result = (nearest_dist, count)
    _client.cache[cache_key] = result
    return result


if __name__ == "__main__":
    lat, lon = 10.7769, 106.7009  # HCM city center
    metro_features = get_metro_features(lat, lon, 5000)
    print(metro_features)
