import pandas as pd

from pipeline.ingestion.load_density import load_density
from pipeline.ingestion.load_density import merge_density_with_alonhadat
from pipeline.ingestion.load_pois import geocode_with_fallback

if __name__ == "__main__":
    df = pd.read_csv(r"data\raw\alonhadat_details.csv")
    density_df = load_density()
    df = merge_density_with_alonhadat(df, density_df)
    
    df[["lat", "lon", "matched_address"]] = df.apply(geocode_with_fallback, axis=1)
    df.to_csv(r"data\processed\alonhadat_with_density_and_poi.csv", index=False)

