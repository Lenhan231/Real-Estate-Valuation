"""Inference for v2.4 production model (3-tier price-only ensemble).

Model: LightGBM + XGBoost + CatBoost ensemble per price tier
Strategy: Price segmentation only (no property type split)
Features: 64 cleaned & optimized (boolean-to-int, low-impact removed)
Performance: 13.25% MAPE, 0.9187 R²
"""
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from geo import GeoLookup, POI_COLS

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "save_models"  # Updated path (lowercase)
READY_CSV = ROOT / "models" / "data" / "model_ready_data.csv"  # Updated path


def load_models():
    models = {}
    for path in MODEL_DIR.glob("*.pkl"):
        if "_meta" in path.name:
            continue
        try:
            models[path.stem] = joblib.load(path)
        except Exception:
            pass

    if not models:
        raise FileNotFoundError(f"No models in {MODEL_DIR} — run scripts/train_xgboost.py first")

    with open(MODEL_DIR / "ensemble_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    medians = pd.read_csv(READY_CSV).median(numeric_only=True)
    return models, meta, medians


# ---------------------------------------------------------------------------
# Feature engineering — mirrors train_xgboost.py:preprocess() for one row
# ---------------------------------------------------------------------------

def build_row(medians, geo: GeoLookup, *,
              street, locality, property_type, legal_status, direction,
              area_m2, width_m, length_m, num_floors, num_bedrooms, road_width_m,
              bin_flags: dict, text_flags: dict):
    """Return (row dict, info dict) or (None, None) if geocoding fails.

    row contains every feature preprocess() would produce for this property.
    info is for display only (lat, lon, source, pois, cache_dist_km, poi_source).
    """
    lat, lon, source = geo.geocode(street, locality)
    if lat is None:
        return None, None

    dist_km = geo.distance_to_center(lat, lon)

    # --- missing-value flags ---
    width_m_missing = int(width_m is None)
    length_m_missing = int(length_m is None)
    if width_m is None and length_m:
        width_m = area_m2 / length_m
    if length_m is None and width_m:
        length_m = area_m2 / width_m
    if width_m is None:
        width_m = float(medians.get("width_m", 4.0))
    if length_m is None:
        length_m = float(medians.get("length_m", 20.0))

    # --- POI features ---
    poi_result = geo.poi_features(lat, lon)
    pois, cache_dist, poi_source = poi_result if len(poi_result) == 3 else (*poi_result, "cache")

    def poi(col):
        v = pois.get(col)
        return float(v) if v is not None else float(medians.get(col, 999.0))

    nearest_metro = pois.get("nearest_metro_km")
    nearest_mall = pois.get("nearest_mall_km")
    nearest_supermarket = pois.get("nearest_supermarket_km")

    # --- locality stats ---
    sq, dens = geo.locality_stats(locality)
    loc_sq = float(sq) if sq is not None else float(medians.get("locality_square", 0.0))
    loc_dens = float(dens) if dens is not None else float(medians.get("locality_population_density", 0.0))

    # --- derived features (same formulas as preprocess()) ---
    perimeter_m = (width_m + length_m) * 2
    shape_ratio = (width_m + 0.1) / (length_m + 0.1)
    area_x_floors = area_m2 * num_floors
    area_x_bedrooms = area_m2 * num_bedrooms
    area_per_bedroom = area_m2 / (num_bedrooms + 1)
    distance_vs_area = dist_km / (area_m2 + 1)
    log_area = np.log1p(area_m2)
    log_dist = np.log1p(dist_km)
    log_dens = np.log1p(loc_dens)

    ns_km = poi("nearest_school_km")
    nh_km = poi("nearest_hospital_km")
    nm_km = float(nearest_mall) if nearest_mall is not None else float(medians.get("nearest_mall_km", 999.0))

    # v2.4: Removed location_score & amenity_score (low-impact per XAI analysis)
    # Models use raw POI distances + counts directly (more predictive)
    nearby_amenities = sum(poi(c) for c in [
        "school_count_3km", "hospital_count_5km", "marketplace_count_3km",
        "supermarket_count_3km", "mall_count_3km", "bus_stop_count_1km", "metro_count_5km"
    ])

    # --- post_day features (not available at inference; zero = neutral) ---
    post_day_year = 0
    post_day_month = 0
    post_day_day = 0

    row = {
        # core numeric
        "num_floors": num_floors,
        "num_bedrooms": num_bedrooms,
        "road_width_m": road_width_m,
        "width_m": width_m,
        "length_m": length_m,
        "area_m2": area_m2,
        # bin flags from form
        "dining_room_bin": int(bin_flags.get("dining_room_bin", 0)),
        "kitchen_bin": int(bin_flags.get("kitchen_bin", 0)),
        "terrace_bin": int(bin_flags.get("terrace_bin", 0)),
        "car_parking_bin": int(bin_flags.get("car_parking_bin", 0)),
        # locality
        "locality_square": loc_sq,
        "locality_population_density": loc_dens,
        "distance_to_center_km": dist_km,
        # POI
        "nearest_school_km": ns_km,
        "school_count_3km": poi("school_count_3km"),
        "nearest_hospital_km": nh_km,
        "hospital_count_5km": poi("hospital_count_5km"),
        "nearest_marketplace_km": poi("nearest_marketplace_km"),
        "marketplace_count_3km": poi("marketplace_count_3km"),
        "nearest_supermarket_km": float(nearest_supermarket) if nearest_supermarket is not None else float(medians.get("nearest_supermarket_km", 999.0)),
        "supermarket_count_3km": poi("supermarket_count_3km"),
        "nearest_mall_km": nm_km,
        "mall_count_3km": poi("mall_count_3km"),
        "nearest_bus_stop_km": poi("nearest_bus_stop_km"),
        "bus_stop_count_1km": poi("bus_stop_count_1km"),
        "nearest_metro_km": float(nearest_metro) if nearest_metro is not None else float(medians.get("nearest_metro_km", 999.0)),
        "metro_count_5km": poi("metro_count_5km"),
        # post_day (zero at inference)
        "post_day_year": post_day_year,
        "post_day_month": post_day_month,
        "post_day_day": post_day_day,
        # derived
        "perimeter_m": perimeter_m,
        "shape_ratio": shape_ratio,
        "area_x_floors": area_x_floors,
        "area_x_bedrooms": area_x_bedrooms,
        "area_per_bedroom": area_per_bedroom,
        "distance_vs_area": distance_vs_area,
        "log_area": log_area,
        "log_distance_to_center": log_dist,
        "log_population_density": log_dens,
        "nearby_amenities": nearby_amenities,
        "nearby_amenities_log": np.log1p(nearby_amenities),
        # v2.4 ratio features
        "frontage_ratio": (width_m + 0.1) / (road_width_m + 0.1) if road_width_m else 0.0,
        "depth_ratio": (length_m + 0.1) / (width_m + 0.1) if width_m else 0.0,
        "road_area_ratio": road_width_m / np.sqrt(area_m2 + 1) if road_width_m else 0.0,
        "amenity_density": nearby_amenities / (area_m2 + 1),
        # text-derived flags (from user checkboxes)
        "is_hem_xe_hoi": int(text_flags.get("is_hem_xe_hoi", 0)),
        "is_mat_tien": int(property_type == "nha_mat_tien"),
        "is_no_hau": int(text_flags.get("is_no_hau", 0)),
        "has_noi_that": int(text_flags.get("has_noi_that", 0)),
        "is_gap": int(text_flags.get("is_gap", 0)),
        "is_kinh_doanh": int(text_flags.get("is_kinh_doanh", 0)),
        # missing-value flags
        "nearest_metro_km_missing": int(nearest_metro is None),
        "nearest_mall_km_missing": int(nearest_mall is None),
        "nearest_supermarket_km_missing": int(nearest_supermarket is None),
        "width_m_missing": width_m_missing,
        "length_m_missing": length_m_missing,
        # locality encoding (filled later by apply_locality_encoding)
        "locality_price_median": 0.0,
        "price_per_sqm_market": 0.0,
    }

    # --- one-hot categorical columns (same as pd.get_dummies in preprocess) ---
    # property_type
    row["property_type_nha_mat_tien"] = int(property_type == "nha_mat_tien")
    row["property_type_nha_trong_hem"] = int(property_type == "nha_trong_hem")
    row["property_type_nan"] = 0

    # legal_status
    for ls in ["so_hong_so_do", "giay_to_hop_le", "unknown", "nan"]:
        row[f"legal_status_{ls}"] = int(legal_status == ls)

    # direction
    for d in ["bac", "dong", "dong_bac", "dong_nam", "nam", "tay", "tay_bac", "tay_nam", "unknown", "nan"]:
        row[f"direction_{d}"] = int(direction == d)

    # listing_type (app always uses "ban" / selling — one-hot with dummy_na)
    row["listing_type_nan"] = 0

    info = {
        "lat": lat, "lon": lon, "source": source,
        "pois": pois, "cache_dist_km": cache_dist, "poi_source": poi_source,
    }
    return row, info


def apply_locality_encoding(row, meta, locality):
    row["locality_price_median"] = float(
        meta.get("locality_price_map", {}).get(locality, meta.get("locality_price_global", 0.0))
    )
    row["price_per_sqm_market"] = float(
        meta.get("locality_sqm_map", {}).get(locality, meta.get("locality_sqm_global", 0.0))
    )
    return row


def predict_price(models, meta, row, price_tier) -> float:
    """Route to price tier, ensemble 3 models (LGBM+XGB+CatBoost), return price in VND.

    Args:
        models: Loaded model dict
        meta: Model metadata
        row: Feature dict (64 features, preprocessed)
        price_tier: 'low' (0-5B), 'mid' (5-20B), or 'high' (20B+)

    Returns:
        Predicted price in VND
    """
    # v2.4: 3-tier (price only), 3 models per tier
    lgbm_key = f"lgbm_{price_tier}"
    xgb_key = f"xgb_{price_tier}"
    cb_key = f"cb_{price_tier}"

    if lgbm_key not in models:
        raise ValueError(f"Model {lgbm_key} not found. Available: {list(models.keys())}")

    # Get feature list from first model (all use same features)
    feat = models[lgbm_key].feature_names_
    X = pd.DataFrame([{f: row.get(f, 0.0) for f in feat}])

    # Ensemble: average 3 models' log predictions
    predictions = []
    for key in [lgbm_key, xgb_key, cb_key]:
        if key in models:
            pred_log = float(models[key].predict(X)[0])
            predictions.append(pred_log)

    if not predictions:
        raise ValueError(f"No models available for {price_tier}")

    pred_log = np.mean(predictions)  # Average in log space
    pred = float(np.expm1(pred_log))  # Back to VND

    return pred
