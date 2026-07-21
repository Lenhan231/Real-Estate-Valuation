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
MODEL_DIR = ROOT / "models" / "saved_models"  # v2.4 production models
READY_CSV = ROOT / "data" / "processed" / "model_training_data.csv"  # v2.4 training data


def load_models():
    """Load v2.4 production models (9 models: 3-tier × 3-algorithm ensemble)."""
    models = {}
    for path in MODEL_DIR.glob("*.pkl"):
        try:
            models[path.stem] = joblib.load(path)
        except Exception as e:
            print(f"Warning: Failed to load {path.stem}: {e}")

    if not models:
        raise FileNotFoundError(f"No models in {MODEL_DIR} — ensure train_production.py has been run")

    # v2.4: No ensemble_meta.pkl needed (simpler 3-tier architecture)
    meta = {"version": "v2.4", "tiers": ["low", "mid", "high"], "models_per_tier": 3}

    medians = pd.read_csv(READY_CSV).median(numeric_only=True)
    return models, meta, medians


# ---------------------------------------------------------------------------
# Feature engineering — mirrors train_xgboost.py:preprocess() for one row
# ---------------------------------------------------------------------------

def build_row(medians, geo: GeoLookup, *,
              street, locality, property_type, legal_status, direction,
              area_m2, width_m, length_m, num_floors, num_bedrooms, road_width_m,
              bin_flags: dict, text_flags: dict):
    """Build exact 64-feature row matching training data. Return (row dict, info dict) or (None, None)."""
    lat, lon, source = geo.geocode(street, locality)
    if lat is None:
        return None, None

    dist_km = geo.distance_to_center(lat, lon)

    # Handle missing width/length
    width_m_missing = int(width_m is None)
    if width_m is None:
        width_m = float(medians.get("width_m", 4.0))
    if length_m is None:
        length_m = float(medians.get("length_m", 20.0))

    # POI features
    poi_result = geo.poi_features(lat, lon)
    pois, cache_dist, poi_source = poi_result if len(poi_result) == 3 else (*poi_result, "cache")

    def poi(col):
        v = pois.get(col)
        return float(v) if v is not None else float(medians.get(col, 999.0))

    # Locality stats
    sq, dens = geo.locality_stats(locality)
    loc_sq = float(sq) if sq is not None else float(medians.get("locality_square", 0.0))
    loc_dens = float(dens) if dens is not None else float(medians.get("locality_population_density", 0.0))

    # Derived features (exactly matching preprocessing.py)
    perimeter_m = (width_m + length_m) * 2
    shape_ratio = (width_m + 0.1) / (length_m + 0.1)
    shape_ratio_missing = int(shape_ratio is None or np.isnan(shape_ratio))
    road_width_m_missing = 0  # Always provided

    nearby_amenities = sum(poi(c) for c in [
        "school_count_3km", "hospital_count_5km", "marketplace_count_3km",
        "supermarket_count_3km", "mall_count_3km", "bus_stop_count_1km", "metro_count_5km"
    ])

    # Build exact 64-feature row
    row = {
        # 1-6: Core numeric
        "num_floors": float(num_floors),
        "num_bedrooms": float(num_bedrooms),
        "road_width_m": float(road_width_m),
        "width_m": float(width_m),
        "length_m": float(length_m),
        "area_m2": float(area_m2),
        # 7-8: Locality
        "locality_square": float(loc_sq),
        "locality_population_density": float(loc_dens),
        # 9: Distance
        "distance_to_center_km": float(dist_km),
        # 10-23: POI distances & counts
        "nearest_school_km": float(poi("nearest_school_km")),
        "school_count_3km": float(poi("school_count_3km")),
        "nearest_hospital_km": float(poi("nearest_hospital_km")),
        "hospital_count_5km": float(poi("hospital_count_5km")),
        "nearest_marketplace_km": float(poi("nearest_marketplace_km")),
        "marketplace_count_3km": float(poi("marketplace_count_3km")),
        "nearest_supermarket_km": float(poi("nearest_supermarket_km")),
        "supermarket_count_3km": float(poi("supermarket_count_3km")),
        "nearest_mall_km": float(poi("nearest_mall_km")),
        "mall_count_3km": float(poi("mall_count_3km")),
        "nearest_bus_stop_km": float(poi("nearest_bus_stop_km")),
        "bus_stop_count_1km": float(poi("bus_stop_count_1km")),
        "nearest_metro_km": float(poi("nearest_metro_km")),
        "metro_count_5km": float(poi("metro_count_5km")),
        # 24-25: Temporal (zero at inference)
        "post_day_month": 0.0,
        "post_day_day": 0.0,
        # 26-30: Missing flags + derived
        "road_width_m_missing": float(road_width_m_missing),
        "perimeter_m": float(perimeter_m),
        "shape_ratio": float(shape_ratio),
        "shape_ratio_missing": float(shape_ratio_missing),
        "width_m_missing": float(width_m_missing),
        # 31-34: Amenity & distance
        "nearby_amenities": float(nearby_amenities),
        "nearby_amenities_log": float(np.log1p(nearby_amenities)),
        "nearest_metro_km_missing": 0.0,  # Always provided
        "amenity_density": float(nearby_amenities / (area_m2 + 1)),
        # 35-40: Text features
        "is_hem_xe_hoi": float(text_flags.get("is_hem_xe_hoi", 0)),
        "is_mat_tien": float(property_type == "nha_mat_tien"),
        "is_no_hau": float(text_flags.get("is_no_hau", 0)),
        "has_noi_that": float(text_flags.get("has_noi_that", 0)),
        "is_gap": float(text_flags.get("is_gap", 0)),
        "is_kinh_doanh": float(text_flags.get("is_kinh_doanh", 0)),
        # 41-47: Interaction & log features
        "area_x_floors": float(area_m2 * num_floors),
        "area_x_bedrooms": float(area_m2 * num_bedrooms),
        "area_per_bedroom": float(area_m2 / (num_bedrooms + 1)),
        "distance_vs_area": float(dist_km / (area_m2 + 1)),
        "log_area": float(np.log1p(area_m2)),
        "log_distance_to_center": float(np.log1p(dist_km)),
        "log_population_density": float(np.log1p(loc_dens)),
        # 48-50: Ratio features
        "frontage_ratio": float((width_m + 0.1) / (road_width_m + 0.1)),
        "depth_ratio": float((length_m + 0.1) / (width_m + 0.1)),
        "road_area_ratio": float(road_width_m / np.sqrt(area_m2 + 1)),
        # 51-54: Direction one-hot
        "direction_dong_bac": float(direction == "dong_bac"),
        "direction_dong_nam": float(direction == "dong_nam"),
        "direction_nam": float(direction == "nam"),
        "direction_unknown": float(direction == "unknown"),
        # 55-56: Property type one-hot
        "property_type_nha_mat_tien": float(property_type == "nha_mat_tien"),
        "property_type_nha_trong_hem": float(property_type == "nha_trong_hem"),
        # 57-59: Legal status one-hot
        "legal_status_giay_to_hop_le": float(legal_status == "giay_to_hop_le"),
        "legal_status_so_hong_so_do": float(legal_status == "so_hong_so_do"),
        "legal_status_unknown": float(legal_status == "unknown"),
        # 60-64: Bin flags one-hot
        "dining_room_bin_False": float(bin_flags.get("dining_room_bin", 0) == 0),
        "terrace_bin_False": float(bin_flags.get("terrace_bin", 0) == 0),
        "terrace_bin_True": float(bin_flags.get("terrace_bin", 0) == 1),
        "car_parking_bin_False": float(bin_flags.get("car_parking_bin", 0) == 0),
        "car_parking_bin_True": float(bin_flags.get("car_parking_bin", 0) == 1),
    }

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
    model = models[lgbm_key]
    if hasattr(model, 'feature_name'):
        feat = model.feature_name()  # LightGBM: method call
    elif hasattr(model, 'feature_names'):
        feat = model.feature_names  # XGBoost/CatBoost
    elif hasattr(model, 'feature_names_'):
        feat = model.feature_names_
    else:
        feat = list(row.keys())  # Fallback: use all keys from row

    # Only use features that exist in the training data
    X_dict = {f: row.get(f, 0.0) for f in feat if f in row}
    # Add missing features as 0
    for f in feat:
        if f not in X_dict:
            X_dict[f] = 0.0

    X = pd.DataFrame([X_dict])

    # Ensure only the model's expected features
    if X.shape[1] != len(feat):
        print(f"⚠️  Warning: Expected {len(feat)} features, got {X.shape[1]}")
        X = X[[f for f in feat if f in X.columns]]

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
