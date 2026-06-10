import pandas as pd

from pipeline.transformation.cleaning import clean_data
from pipeline.ingestion.load_density import load_density
from pipeline.ingestion.load_density import merge_density_with_alonhadat

from pipeline.ingestion.load_pois import add_coordinates
from pipeline.ingestion.load_pois import distance_to_center



if __name__ == "__main__":
    df = pd.read_csv(r"data\raw\alonhadat_details.csv")
    df = clean_data(df)
    
    # Density
    density_df = load_density()
    df = merge_density_with_alonhadat(df, density_df)

    # Lat, lon, distance to center
    df = add_coordinates(df) 
    df = distance_to_center(df)
    
    