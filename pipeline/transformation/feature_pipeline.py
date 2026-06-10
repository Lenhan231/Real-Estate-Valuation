import pandas as pd
from pipeline.transformation.metro_features import get_metro_features
from pipeline.transformation.poi_features import get_poi_features

def get_additional_features(df, school_radius=3000, hospital_radius=5000, marketplace_radius=3000, supermarket_radius=3000, mall_radius=3000, bus_stop_radius=1000, metro_radius=5000) -> pd.DataFrame:
    # Education
    df[["nearest_school_km", f"school_count_{school_radius//1000}km"]] = df.apply(
        lambda row: pd.Series(
            get_poi_features(
                row["lat"],
                row["lon"],
                "amenity",
                "school",
                school_radius 
            )
        ),
        axis=1
    )

    # Healthcare
    df[["nearest_hospital_km", f"hospital_count_{hospital_radius//1000}km"]] = df.apply(
        lambda row: pd.Series(
            get_poi_features(
                row["lat"],
                row["lon"],
                "amenity",
                "hospital",
                hospital_radius
            )
        ),
        axis=1
    )

    # Marketplace
    df[["nearest_marketplace_km", f"marketplace_count_{marketplace_radius//1000}km"]] = df.apply(
        lambda row: pd.Series(
            get_poi_features(
                row["lat"],
                row["lon"],
                "amenity",
                "marketplace",
                marketplace_radius
            )
        ),
        axis=1
    )

    # Supermarket
    df[["nearest_supermarket_km", f"supermarket_count_{supermarket_radius//1000}km"]] = df.apply(
        lambda row: pd.Series(
            get_poi_features(
                row["lat"],
                row["lon"],
                "shop",
                "supermarket",
                supermarket_radius
            )
        ),
        axis=1
    )

    # Mall
    df[["nearest_mall_km", f"mall_count_{mall_radius//1000}km"]] = df.apply(
        lambda row: pd.Series(
            get_poi_features(
                row["lat"],
                row["lon"],
                "shop",
                "mall",
                mall_radius
            )
        ),
        axis=1
    )

    # Bus stop
    df[["nearest_bus_stop_km", f"bus_stop_count_{bus_stop_radius//1000}km"]] = df.apply(
        lambda row: pd.Series(
            get_poi_features(
                row["lat"],
                row["lon"],
                "highway",
                "bus_stop",
                bus_stop_radius
            )
        ),
        axis=1
    )

    # Metro
    df[["nearest_metro_km", f"metro_count_{metro_radius//1000}km"]] = df.apply(
        lambda row: pd.Series(
            get_metro_features(
                row["lat"],
                row["lon"],
                metro_radius
            )
        ),
        axis=1
    )
    return df