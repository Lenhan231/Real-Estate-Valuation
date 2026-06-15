import pandas as pd
from pipeline.transformation.metro_features import _extractor as metro_extractor
from pipeline.transformation.poi_features import _extractor as poi_extractor

def get_additional_features(df, school_radius=3000, hospital_radius=5000, marketplace_radius=3000, supermarket_radius=3000, mall_radius=3000, bus_stop_radius=1000, metro_radius=5000) -> pd.DataFrame:
    """Vectorized feature extraction using BallTree batch queries"""

    lats = df["lat"].values
    lons = df["lon"].values

    # Education
    df["nearest_school_km"] = poi_extractor.get_nearest_distances_batch(lats, lons, "schools")
    df[f"school_count_{school_radius//1000}km"] = poi_extractor.get_counts_batch(lats, lons, "schools", school_radius)

    # Healthcare
    df["nearest_hospital_km"] = poi_extractor.get_nearest_distances_batch(lats, lons, "hospitals")
    df[f"hospital_count_{hospital_radius//1000}km"] = poi_extractor.get_counts_batch(lats, lons, "hospitals", hospital_radius)

    # Marketplace
    df["nearest_marketplace_km"] = poi_extractor.get_nearest_distances_batch(lats, lons, "marketplaces")
    df[f"marketplace_count_{marketplace_radius//1000}km"] = poi_extractor.get_counts_batch(lats, lons, "marketplaces", marketplace_radius)

    # Supermarket
    df["nearest_supermarket_km"] = poi_extractor.get_nearest_distances_batch(lats, lons, "supermarkets")
    df[f"supermarket_count_{supermarket_radius//1000}km"] = poi_extractor.get_counts_batch(lats, lons, "supermarkets", supermarket_radius)

    # Mall
    df["nearest_mall_km"] = poi_extractor.get_nearest_distances_batch(lats, lons, "malls")
    df[f"mall_count_{mall_radius//1000}km"] = poi_extractor.get_counts_batch(lats, lons, "malls", mall_radius)

    # Bus stop
    df["nearest_bus_stop_km"] = poi_extractor.get_nearest_distances_batch(lats, lons, "bus_stops")
    df[f"bus_stop_count_{bus_stop_radius//1000}km"] = poi_extractor.get_counts_batch(lats, lons, "bus_stops", bus_stop_radius)

    # Metro
    df["nearest_metro_km"] = metro_extractor.get_nearest_distances_batch(lats, lons)
    df[f"metro_count_{metro_radius//1000}km"] = metro_extractor.get_counts_batch(lats, lons, metro_radius)

    return df