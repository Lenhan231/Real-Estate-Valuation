"""Shared feature engineering for both training & inference.

This module is the single source of truth for feature construction.
Both preprocessing.py and inference.py should use this to stay in sync.
"""

import numpy as np


def build_features_from_row(row_dict, geo_lookup=None):
    """Build all engineered features from base properties.

    Args:
        row_dict: Dict with keys: area_m2, width_m, length_m, num_floors, num_bedrooms,
                  road_width_m, distance_to_center_km, locality_square,
                  locality_population_density, nearby_amenities, POI data, etc.
        geo_lookup: GeoLookup instance (optional, for POI features)

    Returns:
        Dict with all 78 engineered features in exact order
    """

    # Extract base features
    area_m2 = row_dict.get("area_m2", 0.0)
    width_m = row_dict.get("width_m", 0.0)
    length_m = row_dict.get("length_m", 0.0)
    num_floors = row_dict.get("num_floors", 1.0)
    num_bedrooms = row_dict.get("num_bedrooms", 1.0)
    road_width_m = row_dict.get("road_width_m", 0.0)
    dist_km = row_dict.get("distance_to_center_km", 0.0)
    loc_sq = row_dict.get("locality_square", 0.0)
    loc_dens = row_dict.get("locality_population_density", 0.0)
    nearby_amenities = row_dict.get("nearby_amenities", 0.0)

    # Build feature dict in exact order (matching training CSV)
    features = {}

    # Core numeric (6)
    features["num_floors"] = float(num_floors)
    features["num_bedrooms"] = float(num_bedrooms)
    features["road_width_m"] = float(road_width_m)
    features["width_m"] = float(width_m)
    features["length_m"] = float(length_m)
    features["area_m2"] = float(area_m2)

    # Locality (2)
    features["locality_square"] = float(loc_sq)
    features["locality_population_density"] = float(loc_dens)

    # Distance (1)
    features["distance_to_center_km"] = float(dist_km)

    # POI distances & counts (14)
    poi_cols = [
        "nearest_school_km", "school_count_3km",
        "nearest_hospital_km", "hospital_count_5km",
        "nearest_marketplace_km", "marketplace_count_3km",
        "nearest_supermarket_km", "supermarket_count_3km",
        "nearest_mall_km", "mall_count_3km",
        "nearest_bus_stop_km", "bus_stop_count_1km",
        "nearest_metro_km", "metro_count_5km"
    ]
    for col in poi_cols:
        features[col] = float(row_dict.get(col, 0.0))

    # Temporal (2)
    features["post_day_month"] = float(row_dict.get("post_day_month", 0.0))
    features["post_day_day"] = float(row_dict.get("post_day_day", 0.0))

    # Missing flags & derived (5)
    features["road_width_m_missing"] = float(row_dict.get("road_width_m_missing", 0.0))

    # Derived: perimeter & shape
    perimeter_m = (width_m + length_m) * 2
    shape_ratio = (width_m + 0.1) / (length_m + 0.1) if length_m > 0 else 1.0
    features["perimeter_m"] = float(perimeter_m)
    features["shape_ratio"] = float(shape_ratio)
    features["shape_ratio_missing"] = float(row_dict.get("shape_ratio_missing", 0.0))
    features["width_m_missing"] = float(row_dict.get("width_m_missing", 0.0))

    # Amenity features (4)
    features["nearby_amenities"] = float(nearby_amenities)
    features["nearby_amenities_log"] = float(np.log1p(nearby_amenities))
    features["nearest_metro_km_missing"] = float(row_dict.get("nearest_metro_km_missing", 0.0))
    features["amenity_density"] = float(nearby_amenities / (area_m2 + 1))

    # Text features (6)
    text_flags = [
        "is_hem_xe_hoi", "is_mat_tien", "is_no_hau",
        "has_noi_that", "is_gap", "is_kinh_doanh"
    ]
    for col in text_flags:
        features[col] = float(row_dict.get(col, 0.0))

    # Base interactions & logs (7)
    features["area_x_floors"] = float(area_m2 * num_floors)
    features["area_x_bedrooms"] = float(area_m2 * num_bedrooms)
    features["area_per_bedroom"] = float(area_m2 / (num_bedrooms + 1))
    features["distance_vs_area"] = float(dist_km / (area_m2 + 1))
    features["log_area"] = float(np.log1p(area_m2))
    features["log_distance_to_center"] = float(np.log1p(dist_km))
    features["log_population_density"] = float(np.log1p(loc_dens))

    # Ratio features (3)
    features["frontage_ratio"] = float((width_m + 0.1) / (road_width_m + 0.1))
    features["depth_ratio"] = float((length_m + 0.1) / (width_m + 0.1))
    features["road_area_ratio"] = float(road_width_m / np.sqrt(area_m2 + 1))

    # Polynomial features (6)
    features["area_m2_squared"] = float(area_m2 ** 2)
    features["area_m2_sqrt"] = float(np.sqrt(area_m2 + 0.1))
    features["distance_squared"] = float(dist_km ** 2)
    features["road_width_squared"] = float(road_width_m ** 2)
    features["bedrooms_squared"] = float(num_bedrooms ** 2)
    features["floors_squared"] = float(num_floors ** 2)

    # Additional interactions (8)
    features["area_x_distance"] = float(area_m2 * dist_km)
    features["area_per_distance"] = float(area_m2 / (dist_km + 0.1))
    features["bedrooms_x_distance"] = float(num_bedrooms * dist_km)
    features["floors_x_distance"] = float(num_floors * dist_km)
    features["area_x_road_width"] = float(area_m2 * road_width_m)
    features["width_x_length"] = float(width_m * length_m)
    features["density_x_area"] = float(loc_dens * area_m2)
    features["locality_sq_x_area"] = float(loc_sq * area_m2)

    # Direction one-hot (4)
    direction = row_dict.get("direction", "unknown")
    features["direction_dong_bac"] = float(direction == "dong_bac")
    features["direction_dong_nam"] = float(direction == "dong_nam")
    features["direction_nam"] = float(direction == "nam")
    features["direction_unknown"] = float(direction == "unknown")

    # Property type one-hot (2)
    property_type = row_dict.get("property_type", "unknown")
    features["property_type_nha_mat_tien"] = float(property_type == "nha_mat_tien")
    features["property_type_nha_trong_hem"] = float(property_type == "nha_trong_hem")

    # Legal status one-hot (3)
    legal_status = row_dict.get("legal_status", "unknown")
    features["legal_status_giay_to_hop_le"] = float(legal_status == "giay_to_hop_le")
    features["legal_status_so_hong_so_do"] = float(legal_status == "so_hong_so_do")
    features["legal_status_unknown"] = float(legal_status == "unknown")

    # Bin flags one-hot (5)
    dining_room_bin = row_dict.get("dining_room_bin", 0)
    terrace_bin = row_dict.get("terrace_bin", 0)
    car_parking_bin = row_dict.get("car_parking_bin", 0)

    features["dining_room_bin_False"] = float(dining_room_bin == 0)
    features["terrace_bin_False"] = float(terrace_bin == 0)
    features["terrace_bin_True"] = float(terrace_bin == 1)
    features["car_parking_bin_False"] = float(car_parking_bin == 0)
    features["car_parking_bin_True"] = float(car_parking_bin == 1)

    return features


# Feature names in exact order (78 total)
FEATURE_NAMES = [
    "num_floors", "num_bedrooms", "road_width_m", "width_m", "length_m", "area_m2",
    "locality_square", "locality_population_density",
    "distance_to_center_km",
    "nearest_school_km", "school_count_3km", "nearest_hospital_km", "hospital_count_5km",
    "nearest_marketplace_km", "marketplace_count_3km", "nearest_supermarket_km",
    "supermarket_count_3km", "nearest_mall_km", "mall_count_3km", "nearest_bus_stop_km",
    "bus_stop_count_1km", "nearest_metro_km", "metro_count_5km",
    "post_day_month", "post_day_day",
    "road_width_m_missing", "perimeter_m", "shape_ratio", "shape_ratio_missing", "width_m_missing",
    "nearby_amenities", "nearby_amenities_log", "nearest_metro_km_missing", "amenity_density",
    "is_hem_xe_hoi", "is_mat_tien", "is_no_hau", "has_noi_that", "is_gap", "is_kinh_doanh",
    "area_x_floors", "area_x_bedrooms", "area_per_bedroom", "distance_vs_area",
    "log_area", "log_distance_to_center", "log_population_density",
    "frontage_ratio", "depth_ratio", "road_area_ratio",
    "area_m2_squared", "area_m2_sqrt", "distance_squared", "road_width_squared",
    "bedrooms_squared", "floors_squared",
    "area_x_distance", "area_per_distance", "bedrooms_x_distance", "floors_x_distance",
    "area_x_road_width", "width_x_length", "density_x_area", "locality_sq_x_area",
    "direction_dong_bac", "direction_dong_nam", "direction_nam", "direction_unknown",
    "property_type_nha_mat_tien", "property_type_nha_trong_hem",
    "legal_status_giay_to_hop_le", "legal_status_so_hong_so_do", "legal_status_unknown",
    "dining_room_bin_False", "terrace_bin_False", "terrace_bin_True",
    "car_parking_bin_False", "car_parking_bin_True"
]

assert len(FEATURE_NAMES) == 78, f"Expected 78 features, got {len(FEATURE_NAMES)}"
