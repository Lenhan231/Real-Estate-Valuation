"""Geo enrichment cho app — tái sử dụng dữ liệu pipeline ETL đã build.

Tọa độ + feature POI lấy từ Supabase real-time (address_cache table), với
CSV fallback (data/cache/localities.csv) nếu Supabase không available.
Chỉ khi gặp đường chưa có trong cache mới gọi Nominatim API.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Load .env file
try:
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parent.parent.parent  # project root
    load_dotenv(ROOT / ".env")
except ImportError:
    ROOT = Path(__file__).resolve().parent.parent.parent

RAW_CSV = ROOT / "data" / "cache" / "localities.csv"
DENSITY_CSV = ROOT / "data" / "external" / "density_data.csv"

# Supabase config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE_CACHE", "address_cache")

HCM_CENTER = (10.7769, 106.7009)

POI_COLS = [
    'nearest_school_km', 'school_count_3km',
    'nearest_hospital_km', 'hospital_count_5km',
    'nearest_marketplace_km', 'marketplace_count_3km',
    'nearest_supermarket_km', 'supermarket_count_3km',
    'nearest_mall_km', 'mall_count_3km',
    'nearest_bus_stop_km', 'bus_stop_count_1km',
    'nearest_metro_km', 'metro_count_5km',
]

# Cùng cấu hình với pipeline/transformation/poi_features.py (POI_TYPES)
POI_QUERIES = [
    ('amenity', 'school', 'nearest_school_km', 'school_count_3km', 3000),
    ('amenity', 'hospital', 'nearest_hospital_km', 'hospital_count_5km', 5000),
    ('amenity', 'marketplace', 'nearest_marketplace_km', 'marketplace_count_3km', 3000),
    ('shop', 'supermarket', 'nearest_supermarket_km', 'supermarket_count_3km', 3000),
    ('shop', 'mall', 'nearest_mall_km', 'mall_count_3km', 3000),
    ('highway', 'bus_stop', 'nearest_bus_stop_km', 'bus_stop_count_1km', 1000),
]


def _norm(s) -> str:
    return " ".join(str(s).lower().split())


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


class GeoLookup:
    def __init__(self):
        # Try Supabase first, fallback to CSV
        self.data_source = "CSV"  # Default

        print("\n=== GeoLookup Init ===")
        print(f"RAW_CSV: {RAW_CSV}")
        print(f"DENSITY_CSV: {DENSITY_CSV}")

        df = self._load_from_supabase()
        if df is not None:
            self.data_source = "Supabase"
        else:
            df = self._load_from_csv()
            self.data_source = "CSV"

        if df is None or df.empty:
            print(f"❌ ERROR: df={df}, empty={df.empty if df is not None else 'N/A'}")
            raise FileNotFoundError("No data loaded from Supabase or CSV")

        print(f"✅ Loaded {len(df)} rows, source={self.data_source}")

        df = df.dropna(subset=['lat', 'lon']).copy()
        df['street_n'] = df['street'].map(_norm)
        df['locality_n'] = df['locality'].map(_norm)
        # locality_square dùng dấu phẩy thập phân trong dữ liệu gốc
        df['locality_square'] = pd.to_numeric(
            df['locality_square'].astype(str).str.replace(',', '.', regex=False),
            errors='coerce',
        )
        self.df = df

    @staticmethod
    def _load_from_supabase():
        """Load từ Supabase address_cache table + merge density data (real-time)."""
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            print("⏭️  Supabase env vars not set, skipping...")
            return None
        try:
            from supabase import create_client
            client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            response = client.table(SUPABASE_TABLE).select("*").execute()
            if response.data:
                df = pd.DataFrame(response.data)
                print(f"✅ Loaded {len(df)} rows từ Supabase {SUPABASE_TABLE}")

                # Merge với density data để có locality_square & locality_population_density
                if DENSITY_CSV.exists():
                    print(f"Merging with density data...")
                    density = pd.read_csv(DENSITY_CSV)
                    df = df.merge(density, on=['locality', 'region'], how='left')
                    print(f"  ✅ Merged, now {len(df)} rows with density features")

                return df
            else:
                print(f"⚠️  Supabase table empty, falling back to CSV")
        except ImportError:
            print("⚠️  supabase library not installed: pip install supabase")
        except Exception as e:
            print(f"⚠️  Supabase error: {e}")
            import traceback
            traceback.print_exc()
        return None

    @staticmethod
    def _load_from_csv():
        """Load từ CSV + merge density data (fallback)."""
        try:
            print(f"Loading {RAW_CSV}...")
            if not RAW_CSV.exists():
                print(f"  ❌ File not found: {RAW_CSV}")
                return None

            df = pd.read_csv(RAW_CSV)
            print(f"  ✅ Loaded {len(df)} rows from localities.csv")

            # Merge với density data để có locality_square & locality_population_density
            if DENSITY_CSV.exists():
                print(f"Merging with {DENSITY_CSV}...")
                density = pd.read_csv(DENSITY_CSV)
                df = df.merge(density, on=['locality', 'region'], how='left')
                print(f"  ✅ Merged, now {len(df)} rows")
            else:
                print(f"  ⚠️  Density file not found: {DENSITY_CSV}")

            return df if not df.empty else None
        except Exception as e:
            print(f"❌ CSV load error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def localities(self) -> list[str]:
        return sorted(self.df['locality_n'].dropna().unique())

    def streets_of(self, locality: str) -> list[str]:
        sub = self.df[self.df['locality_n'] == _norm(locality)]
        return sorted(sub['street_n'].dropna().unique())

    def geocode(self, street: str, locality: str):
        """(street, locality) → (lat, lon, nguồn). Ưu tiên cache của pipeline,
        đường lạ mới gọi Nominatim, cuối cùng rơi về tâm phường, rồi HCM center."""
        street_n, locality_n = _norm(street), _norm(locality)

        sub = self.df[(self.df['street_n'] == street_n) & (self.df['locality_n'] == locality_n)]
        if len(sub):
            return float(sub['lat'].median()), float(sub['lon'].median()), 'cache pipeline (đúng đường)'

        if street_n:
            try:
                from geopy.geocoders import Nominatim
                loc = Nominatim(user_agent='hpp_app', timeout=5).geocode(
                    f"{street_n}, {locality_n}, Hồ Chí Minh, Vietnam"
                )
                if loc:
                    return float(loc.latitude), float(loc.longitude), 'Nominatim API'
            except Exception:
                pass

        sub = self.df[self.df['locality_n'] == locality_n]
        if len(sub):
            return float(sub['lat'].median()), float(sub['lon'].median()), 'cache pipeline (tâm phường)'

        # Fallback: use HCM city center
        print(f"⚠️  No location found for {street}/{locality}, using HCM center")
        return HCM_CENTER[0], HCM_CENTER[1], 'HCM center (fallback)'

    def poi_features(self, lat: float, lon: float, max_cache_km: float = 2.0, allow_live: bool = True):
        """Feature POI cho một vị trí. Trả về (dict feature, khoảng cách km tới
        điểm cache, nguồn 'cache'/'overpass').

        Đường nhanh: copy từ listing đã crawl gần nhất (tức thì). Nếu vị trí xa
        vùng crawl quá max_cache_km thì gọi Overpass client của pipeline để tính
        thật (chậm hơn nhưng chính xác), rồi lưu vào localities.csv cho lần sau.
        """
        d = _haversine_km(lat, lon, self.df['lat'].values, self.df['lon'].values)
        i = int(np.argmin(d))
        row = self.df.iloc[i]
        feats = {c: (float(row[c]) if pd.notna(row[c]) else None) for c in POI_COLS}
        cache_dist = float(d[i])

        if cache_dist <= max_cache_km or not allow_live:
            return feats, cache_dist, 'cache'

        try:
            live = self._poi_features_live(lat, lon)
            return live, 0.0, 'overpass'
        except Exception:
            return feats, cache_dist, 'cache'  # mạng lỗi thì đành dùng điểm gần nhất

    @staticmethod
    def _poi_features_live(lat: float, lon: float) -> dict:
        """Tính feature POI qua đúng client Overpass của pipeline ETL
        (tự đọc/ghi cache localities.csv nên chỉ chậm ở lần gọi đầu)."""
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from pipeline.transformation.poi_features import get_poi_features
        from pipeline.transformation.metro_features import get_metro_features
        from pipeline.ingestion.load_pois import append_to_localities_csv

        feats = {}
        for key, value, near_col, count_col, radius in POI_QUERIES:
            nearest, count = get_poi_features(lat, lon, key, value, radius)
            feats[near_col] = float(nearest) if nearest is not None else None
            feats[count_col] = float(count)
        nearest, count = get_metro_features(lat, lon, 5000)
        feats['nearest_metro_km'] = float(nearest) if nearest is not None else None
        feats['metro_count_5km'] = float(count)

        # Lưu lại để các lần chạy sau (và cả pipeline) dùng ngay khỏi gọi API
        append_to_localities_csv(lat, lon, {k: v for k, v in feats.items() if v is not None})
        return feats

    def locality_stats(self, locality: str):
        """(diện tích km², mật độ dân số) của phường, đúng đơn vị lúc train."""
        sub = self.df[self.df['locality_n'] == _norm(locality)]
        if not len(sub):
            return None, None
        sq = sub['locality_square'].median()
        dens = sub['locality_population_density'].median()
        return (
            float(sq) if pd.notna(sq) else None,
            float(dens) if pd.notna(dens) else None,
        )

    @staticmethod
    def distance_to_center(lat: float, lon: float) -> float:
        return float(_haversine_km(lat, lon, HCM_CENTER[0], HCM_CENTER[1]))
