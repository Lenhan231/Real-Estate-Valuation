import time
import requests
from geopy.distance import geodesic
import subprocess
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


class OverpassAPIClient(ABC):
    """Base client for querying Overpass API with caching."""

    def __init__(self, locality_file: Path = None):
        self.locality_file = locality_file or (Path(__file__).parent.parent.parent / "data" / "localities.csv")
        self.cache = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "DataProcessing/1.0"})
        self._load_persistent_cache()

    def _load_persistent_cache(self):
        """Load cached results from localities.csv to speed up subsequent runs."""
        if not self.locality_file.exists():
            return

        try:
            df = pd.read_csv(self.locality_file)
            for _, row in df.iterrows():
                lat = row.get('lat')
                lon = row.get('lon')
                if pd.isna(lat) or pd.isna(lon):
                    continue
                self._parse_cache_row(row, lat, lon)

            print(f"  ✓ Loaded {len(self.cache)} cached queries from localities.csv")
        except Exception:
            pass  # Silent fail

    @abstractmethod
    def _parse_cache_row(self, row, lat, lon):
        """Parse a row from localities.csv and populate cache. Subclasses implement query-specific logic."""
        pass

    def _run_cmd(self, cmd: list[str]) -> str:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def _extract_coordinates(self, item):
        """Extract coordinates from an OSM element."""
        if item["type"] == "node":
            return item.get("lat"), item.get("lon")
        center = item.get("center", {})
        return center.get("lat"), center.get("lon")

    def _query_overpass_api(self, query, max_retries=5, base_sleep=2):
        """Query Overpass API with retry logic for rate limits."""
        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.post(OVERPASS_URL, data=query, timeout=(10, 60))
                if response.status_code in (429, 504):
                    self._handle_rate_limit()
                    time.sleep(base_sleep * (2 ** (attempt - 1)))
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException:
                if attempt < max_retries:
                    time.sleep(base_sleep * (2 ** (attempt - 1)))
        return None

    def _handle_rate_limit(self):
        """Handle rate limiting by reconnecting VPN/proxy if available."""
        try:
            self._run_cmd(["warp-cli", "disconnect"])
            self._run_cmd(["warp-cli", "connect"])
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # Silently skip if warp-cli not available

    def _calculate_metrics(self, lat, lon, elements):
        """Calculate nearest distance and count from a list of OSM elements."""
        distances = []
        for item in elements:
            item_lat, item_lon = self._extract_coordinates(item)
            if item_lat is not None and item_lon is not None:
                distances.append(geodesic((lat, lon), (item_lat, item_lon)).km)

        return (min(distances), len(distances)) if distances else (None, 0)
