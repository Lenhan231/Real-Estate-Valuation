import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse connections
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=2)
session.mount("http://", adapter)
session.mount("https://", adapter)

def get_aqi(lat, lon, timeout=(3, 10)):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["us_aqi", "pm2_5"]
    }

    try:
        res = session.get(url, params=params, timeout=timeout)
        res.raise_for_status()
        data = res.json()
        current = data.get("current", {})
        return {
            "lat": lat,
            "lon": lon,
            "aqi": current.get("us_aqi"),
            "pm25": current.get("pm2_5"),
        }
    except Exception:
        return {
            "lat": lat,
            "lon": lon,
            "aqi": None,
            "pm25": None,
        }