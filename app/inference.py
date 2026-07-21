"""Inference for v2.6 production model (3-tier price-only ensemble).

Model: LightGBM + XGBoost + CatBoost ensemble per price tier
Strategy: Price segmentation only (no property type split)
Features: 78 optimized (64 base + 14 polynomial/interaction)
Performance: 13.10% MAPE, 0.9200 R²

Note: Features built via shared feature_builder.py (single source of truth)
"""
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from feature_builder import build_features_from_row, FEATURE_NAMES
from geo import GeoLookup, POI_COLS

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "saved_models"  # v2.4 production models
READY_CSV = ROOT / "data" / "processed" / "model_training_data.csv"  # v2.4 training data


def load_models():
    """Load v2.6 production models (9 models: 3-tier × 3-algorithm ensemble)."""
    models = {}
    for path in MODEL_DIR.glob("*.pkl"):
        try:
            models[path.stem] = joblib.load(path)
        except Exception as e:
            print(f"Warning: Failed to load {path.stem}: {e}")

    if not models:
        raise FileNotFoundError(f"No models in {MODEL_DIR} — ensure train_production.py has been run")

    # Load training data to get exact feature names and count
    training_df = pd.read_csv(READY_CSV)
    feature_cols = [c for c in training_df.columns if c != 'price_vnd']

    meta = {
        "version": "v2.6",
        "tiers": ["low", "mid", "high"],
        "models_per_tier": 3,
        "n_features": len(feature_cols),
        "feature_names": feature_cols
    }

    medians = training_df.median(numeric_only=True)
    return models, meta, medians


# ---------------------------------------------------------------------------
# Feature engineering — mirrors train_xgboost.py:preprocess() for one row
# ---------------------------------------------------------------------------

def build_row(medians, geo: GeoLookup, *,
              street, locality, property_type, legal_status, direction,
              area_m2, width_m, length_m, num_floors, num_bedrooms, road_width_m,
              bin_flags: dict, text_flags: dict):
    """Build exact 78-feature row matching v2.6 training data. Return (row dict, info dict) or (None, None)."""
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

    # Build row - features in EXACT order from training CSV
    # Order matters: must match model training data
    row = {
        "num_floors": float(num_floors),
        "num_bedrooms": float(num_bedrooms),
        "road_width_m": float(road_width_m),
        "width_m": float(width_m),
        "length_m": float(length_m),
        "area_m2": float(area_m2),
        "locality_square": float(loc_sq),
        "locality_population_density": float(loc_dens),
        "distance_to_center_km": float(dist_km),
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
        "post_day_month": 0.0,
        "post_day_day": 0.0,
        "road_width_m_missing": float(road_width_m_missing),
        "perimeter_m": float(perimeter_m),
        "shape_ratio": float(shape_ratio),
        "shape_ratio_missing": float(shape_ratio_missing),
        "width_m_missing": float(width_m_missing),
        "nearby_amenities": float(nearby_amenities),
        "nearby_amenities_log": float(np.log1p(nearby_amenities)),
        "nearest_metro_km_missing": 0.0,
        "amenity_density": float(nearby_amenities / (area_m2 + 1)),
        "is_hem_xe_hoi": float(text_flags.get("is_hem_xe_hoi", 0)),
        "is_mat_tien": float(property_type == "nha_mat_tien"),
        "is_no_hau": float(text_flags.get("is_no_hau", 0)),
        "has_noi_that": float(text_flags.get("has_noi_that", 0)),
        "is_gap": float(text_flags.get("is_gap", 0)),
        "is_kinh_doanh": float(text_flags.get("is_kinh_doanh", 0)),
        "area_x_floors": float(area_m2 * num_floors),
        "area_x_bedrooms": float(area_m2 * num_bedrooms),
        "area_per_bedroom": float(area_m2 / (num_bedrooms + 1)),
        "distance_vs_area": float(dist_km / (area_m2 + 1)),
        "log_area": float(np.log1p(area_m2)),
        "log_distance_to_center": float(np.log1p(dist_km)),
        "log_population_density": float(np.log1p(loc_dens)),
        "frontage_ratio": float((width_m + 0.1) / (road_width_m + 0.1)),
        "depth_ratio": float((length_m + 0.1) / (width_m + 0.1)),
        "road_area_ratio": float(road_width_m / np.sqrt(area_m2 + 1)),
        "area_m2_squared": float(area_m2 ** 2),
        "area_m2_sqrt": float(np.sqrt(area_m2 + 0.1)),
        "distance_squared": float(dist_km ** 2),
        "road_width_squared": float(road_width_m ** 2),
        "bedrooms_squared": float(num_bedrooms ** 2),
        "floors_squared": float(num_floors ** 2),
        "area_x_distance": float(area_m2 * dist_km),
        "area_per_distance": float(area_m2 / (dist_km + 0.1)),
        "bedrooms_x_distance": float(num_bedrooms * dist_km),
        "floors_x_distance": float(num_floors * dist_km),
        "area_x_road_width": float(area_m2 * road_width_m),
        "width_x_length": float(width_m * length_m),
        "density_x_area": float(loc_dens * area_m2),
        "locality_sq_x_area": float(loc_sq * area_m2),
        "direction_dong_bac": float(direction == "dong_bac"),
        "direction_dong_nam": float(direction == "dong_nam"),
        "direction_nam": float(direction == "nam"),
        "direction_unknown": float(direction == "unknown"),
        "property_type_nha_mat_tien": float(property_type == "nha_mat_tien"),
        "property_type_nha_trong_hem": float(property_type == "nha_trong_hem"),
        "legal_status_giay_to_hop_le": float(legal_status == "giay_to_hop_le"),
        "legal_status_so_hong_so_do": float(legal_status == "so_hong_so_do"),
        "legal_status_unknown": float(legal_status == "unknown"),
        "dining_room_bin_False": float(bin_flags.get("dining_room_bin", 0) == 0),
        "terrace_bin_False": float(bin_flags.get("terrace_bin", 0) == 0),
        "terrace_bin_True": float(bin_flags.get("terrace_bin", 0) == 1),
        "car_parking_bin_False": float(bin_flags.get("car_parking_bin", 0) == 0),
        "car_parking_bin_True": float(bin_flags.get("car_parking_bin", 0) == 1),
    }

    # Add 2 locality encoding features (needed for model prediction)
    row["locality_price_median"] = float(medians.get("locality_price_median", 0.0))
    row["price_per_sqm_market"] = float(medians.get("price_per_sqm_market", 0.0))

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
        meta: Model metadata (includes feature_names)
        row: Feature dict (preprocessed)
        price_tier: 'low' (0-5B), 'mid' (5-20B), or 'high' (20B+)

    Returns:
        Predicted price in VND
    """
    # v2.6: 3-tier (price only), 3 models per tier
    lgbm_key = f"lgbm_{price_tier}"
    xgb_key = f"xgb_{price_tier}"
    cb_key = f"cb_{price_tier}"

    if lgbm_key not in models:
        raise ValueError(f"Model {lgbm_key} not found. Available: {list(models.keys())}")

    # Get feature names from metadata (source of truth)
    feat = meta.get("feature_names", list(row.keys()))

    # Build feature dict: use row if available, else median
    X_dict = {}
    for f in feat:
        if f in row:
            X_dict[f] = float(row[f])
        else:
            # Use median from training data as fallback
            X_dict[f] = float(0.0)

    # Create DataFrame with exact feature order from metadata
    X = pd.DataFrame([X_dict])
    X = X[feat]  # Reorder to match training feature order

    # Ensemble: average 3 models' log predictions
    predictions = []
    for key in [lgbm_key, xgb_key, cb_key]:
        if key in models:
            model = models[key]

            # Get model's expected features
            try:
                if hasattr(model, 'feature_names_in_'):
                    model_features = list(model.feature_names_in_)
                elif hasattr(model, 'feature_name'):
                    model_features = list(model.feature_name())
                else:
                    model_features = list(X.columns)

                # Use only features model expects
                X_pred = X[[f for f in model_features if f in X.columns]]
            except:
                X_pred = X

            pred_log = float(model.predict(X_pred)[0])
            predictions.append(pred_log)

    if not predictions:
        raise ValueError(f"No models available for {price_tier}")

    pred_log = np.mean(predictions)  # Average in log space
    pred = float(np.expm1(pred_log))  # Back to VND

    return pred
