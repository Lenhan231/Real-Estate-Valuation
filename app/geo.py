"""Geo enrichment cho app — tái sử dụng dữ liệu pipeline ETL đã build.

Tọa độ + feature POI lấy từ snapshot crawl (data/processed/alonhadat_features.csv, cùng
nguồn với dữ liệu train nên đơn vị luôn khớp với model). Chỉ khi gặp đường
chưa có trong cache mới gọi Nominatim API.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "processed" / "alonhadat_features.csv"

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
        df = pd.read_csv(RAW_CSV)
        df = df.dropna(subset=['lat', 'lon']).copy()
        df['street_n'] = df['street'].map(_norm)
        df['locality_n'] = df['locality'].map(_norm)
        # locality_square dùng dấu phẩy thập phân trong dữ liệu gốc
        df['locality_square'] = pd.to_numeric(
            df['locality_square'].astype(str).str.replace(',', '.', regex=False),
            errors='coerce',
        )
        self.df = df

    def localities(self) -> list[str]:
        return sorted(self.df['locality_n'].dropna().unique())

    def streets_of(self, locality: str) -> list[str]:
        sub = self.df[self.df['locality_n'] == _norm(locality)]
        return sorted(sub['street_n'].dropna().unique())

    def geocode(self, street: str, locality: str):
        """(street, locality) → (lat, lon, nguồn). Ưu tiên cache của pipeline,
        đường lạ mới gọi Nominatim, cuối cùng rơi về tâm phường."""
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

        return None, None, None

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
