"""Inference for v2.6 production model (3-tier price-only ensemble).

Model: LightGBM + XGBoost + CatBoost ensemble per price tier
Strategy: Price segmentation only (no property type split)
Features: 78 optimized (64 base + 14 polynomial/interaction)
Performance: 13.10% MAPE, 0.9200 R²

Note: Features built via shared preprocessing.py (single source of truth).
"""
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "models" / "scripts"))

from .geo import GeoLookup, POI_COLS
from shared.preprocessing import preprocess

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
MODEL_DIR = ROOT / "models" / "saved_models"
READY_CSV = ROOT / "data" / "processed" / "model_training_data.csv"


def load_models():
    """Load v2.6 production models (9 models: 3-tier × 3-algorithm ensemble).

    Feature names sourced from trained model (single source of truth).
    Locality encoding stats computed from training data (reused from preprocessing.py).
    """
    models = {}
    for path in MODEL_DIR.glob("*.pkl"):
        try:
            models[path.stem] = joblib.load(path)
        except Exception as e:
            print(f"Warning: Failed to load {path.stem}: {e}")

    if not models:
        raise FileNotFoundError(f"No models in {MODEL_DIR} — ensure train_production.py has been run")

    # Get feature names from trained model (single source of truth!)
    first_model = models[list(models.keys())[0]]
    try:
        feature_names = list(first_model.feature_names_in_)
    except AttributeError:
        # Fallback if model doesn't have feature_names_in_
        training_df = pd.read_csv(READY_CSV)
        feature_names = [c for c in training_df.columns if c != 'price_vnd']
        feature_names += ["locality_price_median", "price_per_sqm_market"]

    # Load training data to compute locality encoding stats (reuses preprocessing.py logic)
    training_df = pd.read_csv(READY_CSV)

    # Compute locality price/sqm stats from training data (same as add_locality_features)
    # Handle case where locality column might not exist in CSV
    if 'locality' in training_df.columns and 'locality_price_median' in training_df.columns:
        locality_price_map = training_df.groupby('locality')['locality_price_median'].first().to_dict()
        locality_price_global = training_df['locality_price_median'].median()
    else:
        locality_price_map = {}
        locality_price_global = 0.0

    if 'locality' in training_df.columns and 'price_per_sqm_market' in training_df.columns:
        locality_sqm_map = training_df.groupby('locality')['price_per_sqm_market'].first().to_dict()
        locality_sqm_global = training_df['price_per_sqm_market'].median()
    else:
        locality_sqm_map = {}
        locality_sqm_global = 0.0

    meta = {
        "version": "v2.6",
        "tiers": ["low", "mid", "high"],
        "models_per_tier": 3,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        # Locality encoding (from training data, same as preprocessing.add_locality_features)
        "locality_price_map": locality_price_map,
        "locality_sqm_map": locality_sqm_map,
        "locality_price_global": locality_price_global,
        "locality_sqm_global": locality_sqm_global,
    }

    medians = training_df.median(numeric_only=True)
    return models, meta, medians


def build_row(meta, geo: GeoLookup, *,
              street, locality, property_type, legal_status, direction,
              area_m2, width_m, length_m, num_floors, num_bedrooms, road_width_m,
              bin_flags: dict, text_flags: dict):
    """Build exact 80-feature row (78 from preprocessing + 2 locality encoding).

    Reuses preprocessing.py for feature engineering (single source of truth).
    Applies locality encoding from training data (meta dict).
    Returns (row dict with 80 features, info dict) or (None, error_msg).
    """
    error_log = []
    try:
        lat, lon, source = geo.geocode(street, locality)
        if lat is None:
            error_msg = f"Geocode returned None for street='{street}', locality='{locality}'"
            error_log.append(error_msg)
            print(f"❌ {error_msg}")
            return None, "\n".join(error_log)
    except Exception as e:
        error_msg = f"Geocode error: {str(e)}"
        error_log.append(error_msg)
        print(f"❌ [BUILD_ROW] {error_msg}")
        import traceback
        traceback.print_exc()
        return None, "\n".join(error_log)

    try:
        dist_km = geo.distance_to_center(lat, lon)
        print(f"✅ [BUILD_ROW] Distance calculated: {dist_km:.2f} km")
    except Exception as e:
        error_msg = f"Distance error: {str(e)}"
        error_log.append(error_msg)
        print(f"❌ {error_msg}")
        return None, "\n".join(error_log)

    # POI features (geo lookup - may return None)
    try:
        poi_result = geo.poi_features(lat, lon)
        pois, cache_dist, poi_source = poi_result if len(poi_result) == 3 else (*poi_result, "cache")
        print(f"✅ [BUILD_ROW] POI features loaded: source={poi_source}")

        def poi(col):
            v = pois.get(col) if pois else None
            return v  # Pass None, let preprocessing handle imputation
    except Exception as e:
        error_msg = f"POI features error: {str(e)}"
        error_log.append(error_msg)
        print(f"❌ {error_msg}")
        return None, "\n".join(error_log)

    # Locality stats (geo lookup - may return None)
    try:
        sq, dens = geo.locality_stats(locality)
        print(f"✅ [BUILD_ROW] Locality stats: sq={sq}, dens={dens}")
    except Exception as e:
        print(f"❌ [BUILD_ROW] Locality stats error: {e}")
        sq, dens = None, None
    loc_sq = sq  # Pass None, let preprocessing handle imputation
    loc_dens = dens

    # Build single-row DataFrame to pass through preprocessing.preprocess()
    # This reuses the exact feature engineering logic from training
    # Note: Pass None for missing values so preprocessing can apply hierarchical imputation
    row_df = pd.DataFrame([{
        "listing_id": 0,
        "price_vnd": 1e9,  # Dummy price (not used, just for preprocessing)
        "property_type": property_type,
        "legal_status": legal_status,
        "direction": direction,
        "area_m2": float(area_m2),
        "num_floors": float(num_floors) if num_floors is not None else None,
        "num_bedrooms": float(num_bedrooms) if num_bedrooms is not None else None,
        "road_width_m": float(road_width_m) if road_width_m is not None else None,
        "width_m": float(width_m) if width_m is not None else None,
        "length_m": float(length_m) if length_m is not None else None,
        "locality_square": float(loc_sq) if loc_sq is not None else None,
        "locality_population_density": float(loc_dens) if loc_dens is not None else None,
        "distance_to_center_km": float(dist_km),
        "nearest_school_km": poi("nearest_school_km"),
        "school_count_3km": poi("school_count_3km"),
        "nearest_hospital_km": poi("nearest_hospital_km"),
        "hospital_count_5km": poi("hospital_count_5km"),
        "nearest_marketplace_km": poi("nearest_marketplace_km"),
        "marketplace_count_3km": poi("marketplace_count_3km"),
        "nearest_supermarket_km": poi("nearest_supermarket_km"),
        "supermarket_count_3km": poi("supermarket_count_3km"),
        "nearest_mall_km": poi("nearest_mall_km"),
        "mall_count_3km": poi("mall_count_3km"),
        "nearest_bus_stop_km": poi("nearest_bus_stop_km"),
        "bus_stop_count_1km": poi("bus_stop_count_1km"),
        "nearest_metro_km": poi("nearest_metro_km"),
        "metro_count_5km": poi("metro_count_5km"),
        "post_day_month": 0,
        "post_day_day": 0,
        # Text flags
        "is_hem_xe_hoi": int(text_flags.get("is_hem_xe_hoi", 0)),
        "is_mat_tien": int(text_flags.get("is_mat_tien", 0)),
        "is_no_hau": int(text_flags.get("is_no_hau", 0)),
        "has_noi_that": int(text_flags.get("has_noi_that", 0)),
        "is_gap": int(text_flags.get("is_gap", 0)),
        "is_kinh_doanh": int(text_flags.get("is_kinh_doanh", 0)),
        # Bin flags
        "dining_room_bin": int(bin_flags.get("dining_room_bin", 0)),
        "terrace_bin": int(bin_flags.get("terrace_bin", 0)),
        "car_parking_bin": int(bin_flags.get("car_parking_bin", 0)),
    }])

    # Run through preprocessing.preprocess() (single source of truth for 78 features)
    try:
        preprocessed, _, _ = preprocess(row_df)
        if preprocessed.empty:
            error_msg = "Preprocessing returned empty dataframe"
            error_log.append(error_msg)
            print(f"❌ {error_msg}")
            return None, "\n".join(error_log)
        row_dict = preprocessed.iloc[0].to_dict()
        row = {k: float(v) for k, v in row_dict.items() if k != 'price_vnd'}
    except Exception as e:
        error_msg = f"Preprocessing error: {str(e)}"
        error_log.append(error_msg)
        print(f"❌ {error_msg}")
        import traceback
        tb = traceback.format_exc()
        error_log.append(tb)
        return None, "\n".join(error_log)

    # Add 2 locality encoding features from training data (in meta)
    locality_price_map = meta.get("locality_price_map", {})
    locality_sqm_map = meta.get("locality_sqm_map", {})
    row["locality_price_median"] = float(
        locality_price_map.get(locality, meta.get("locality_price_global", 0.0))
    )
    row["price_per_sqm_market"] = float(
        locality_sqm_map.get(locality, meta.get("locality_sqm_global", 0.0))
    )

    info = {
        "lat": lat, "lon": lon, "source": source,
        "pois": pois, "cache_dist_km": cache_dist, "poi_source": poi_source,
    }
    return row, info


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
