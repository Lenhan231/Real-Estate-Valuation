"""Ghép feature cho 1 căn nhà và dự đoán giá, dùng đúng schema lúc train."""
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from geo import GeoLookup, POI_COLS

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "saved_models"
READY_CSV = ROOT / "models" / "data" / "model_ready_data.csv"

# Các cột numeric được khởi tạo bằng median của tập train (từ model_ready_data),
# sau đó ghi đè bằng input người dùng + tra cứu geo. Mọi cột khác khởi tạo 0.
NUM_COLS = POI_COLS + [
    'area_m2', 'distance_to_center_km',
    'num_floors', 'num_bedrooms', 'road_width_m', 'width_m', 'length_m',
    'locality_population_density', 'locality_square', 'lat', 'lon',
]


def load_model():
    files = sorted(MODEL_DIR.glob('*.joblib'))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy model trong {MODEL_DIR} — chạy notebook train trước")
    path = files[-1]
    model = joblib.load(path)
    with open(MODEL_DIR / f"{path.stem}_meta.pkl", 'rb') as f:
        meta = pickle.load(f)
    medians = pd.read_csv(READY_CSV).median(numeric_only=True)
    return model, meta, medians


def build_row(features, medians, geo: GeoLookup, *,
              street, locality, property_type, legal_status, direction,
              area_m2, width_m, length_m, num_floors, num_bedrooms, road_width_m,
              bin_flags: dict, title_flags: dict):
    """Trả về (row dict theo đúng thứ tự features, info dict để hiển thị).
    info = None nếu không geocode được."""
    lat, lon, source = geo.geocode(street, locality)
    if lat is None:
        return None, None

    row = {f: 0.0 for f in features}
    for f in NUM_COLS:
        if f in row and f in medians.index:
            row[f] = float(medians[f])

    # width/length có thể None (người dùng không rõ) — xử lý giống build_features:
    # bật cờ missing, suy từ diện tích / cạnh còn lại, bí quá thì median tập train
    if width_m is None and 'width_m_missing' in row:
        row['width_m_missing'] = 1
    if length_m is None and 'length_m_missing' in row:
        row['length_m_missing'] = 1
    if width_m is None and length_m:
        width_m = area_m2 / length_m
    if length_m is None and width_m:
        length_m = area_m2 / width_m
    if width_m is None:
        width_m = float(medians['width_m'])
    if length_m is None:
        length_m = float(medians['length_m'])

    row.update({
        'area_m2': area_m2, 'width_m': width_m, 'length_m': length_m,
        'num_floors': num_floors, 'num_bedrooms': num_bedrooms, 'road_width_m': road_width_m,
        'lat': lat, 'lon': lon,
        'distance_to_center_km': geo.distance_to_center(lat, lon),
    })

    # Feature POI: cache pipeline (nhanh) hoặc Overpass live nếu xa vùng crawl;
    # giá trị thiếu thì giữ median + bật cờ missing
    poi_result = geo.poi_features(lat, lon)
    if len(poi_result) == 3:
        pois, cache_dist, poi_source = poi_result
    elif len(poi_result) == 2:
        pois, cache_dist = poi_result
        poi_source = 'cache'
    else:
        raise ValueError(f"Unexpected poi_features result shape: {len(poi_result)}")
    for col, val in pois.items():
        miss = f'{col}_missing'
        if val is None:
            if miss in row:
                row[miss] = 1
        else:
            row[col] = val

    sq, dens = geo.locality_stats(locality)
    if sq is not None:
        row['locality_square'] = sq
    if dens is not None:
        row['locality_population_density'] = dens

    # One-hot: bật đúng cột của lựa chọn
    for prefix, val in [('property_type', property_type),
                        ('legal_status', legal_status),
                        ('direction', direction)]:
        key = f'{prefix}_{val}'
        if key in row:
            row[key] = 1

    for key, val in {**bin_flags, **title_flags}.items():
        if key in row:
            row[key] = int(bool(val))

    return row, {'lat': lat, 'lon': lon, 'source': source,
                 'pois': pois, 'cache_dist_km': cache_dist, 'poi_source': poi_source}


def predict_price(model, meta, row) -> float:
    """Dự đoán giá (tỷ VND), tự áp locality encoding + expm1 theo metadata."""
    features = meta['features']
    X = pd.DataFrame([row], columns=features)
    pred = float(model.predict(X)[0])
    if meta.get('target_transform') == 'log1p':
        pred = float(np.expm1(pred))
    return pred


def apply_locality_encoding(row, meta, locality):
    row['locality_price_median'] = float(
        meta.get('locality_price_map', {}).get(
            locality, meta.get('locality_price_global', 0.0))
    )
    return row
