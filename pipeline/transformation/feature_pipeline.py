import pandas as pd
from pipeline.transformation.get_nearest_metro import get_nearest_metro
from pipeline.transformation.get_nearest_poi import get_nearest_poi

def get_additional_features(df) -> pd.DataFrame:
    # Education
    df[["nearest_school_km", "school_count_3km"]] = df.apply(
        lambda row: pd.Series(
            get_nearest_poi(
                row["lat"],
                row["lon"],
                "amenity",
                "school",
                3000
            )
        ),
        axis=1
    )

    # Healthcare
    df[["nearest_hospital_km", "hospital_count_5km"]] = df.apply(
        lambda row: pd.Series(
            get_nearest_poi(
                row["lat"],
                row["lon"],
                "amenity",
                "hospital",
                5000
            )
        ),
        axis=1
    )

    # Marketplace
    df[["nearest_marketplace_km", "marketplace_count_3km"]] = df.apply(
        lambda row: pd.Series(
            get_nearest_poi(
                row["lat"],
                row["lon"],
                "amenity",
                "marketplace",
                3000
            )
        ),
        axis=1
    )

    # Supermarket
    df[["nearest_supermarket_km", "supermarket_count_3km"]] = df.apply(
        lambda row: pd.Series(
            get_nearest_poi(
                row["lat"],
                row["lon"],
                "shop",
                "supermarket",
                3000
            )
        ),
        axis=1
    )

    # Mall
    df[["nearest_mall_km", "mall_count_3km"]] = df.apply(
        lambda row: pd.Series(
            get_nearest_poi(
                row["lat"],
                row["lon"],
                "shop",
                "mall",
                3000
            )
        ),
        axis=1
    )

    # Bus stop
    df[["nearest_bus_stop_km", "bus_stop_count_1km"]] = df.apply(
        lambda row: pd.Series(
            get_nearest_poi(
                row["lat"],
                row["lon"],
                "highway",
                "bus_stop",
                1000
            )
        ),
        axis=1
    )

    # Metro
    df[["nearest_metro_km", "metro_count_5km"]] = df.apply(
        lambda row: pd.Series(
            get_nearest_metro(
                row["lat"],
                row["lon"],
                5000
            )
        ),
        axis=1
    )
    return df