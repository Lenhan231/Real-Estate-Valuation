import pandas as pd

from pipeline.transformation.cleaning import clean_data, final_clean
from pipeline.ingestion.load_density import (
    load_density,
    merge_density_with_alonhadat
)
from pipeline.ingestion.load_pois import (
    add_coordinates,
    distance_to_center
)
from pipeline.transformation.feature_pipeline import (
    get_additional_features
)

if __name__ == "__main__":
    df = pd.read_csv(r"data\raw\alonhadat_details.csv")
    df = clean_data(df)
    
    # Density
    density_df = load_density()
    df = merge_density_with_alonhadat(df, density_df)

    # Lat, lon, distance to center
    df = add_coordinates(df) 
    df = distance_to_center(df)

    # Additional features
    df = get_additional_features(df, 
                                school_radius=3000, 
                                hospital_radius=5000, 
                                marketplace_radius=3000,
                                supermarket_radius=3000, 
                                mall_radius=3000, 
                                bus_stop_radius=1000, 
                                metro_radius=5000)

    df.to_csv(r"data\processed\alonhadat_features.csv", index=False)