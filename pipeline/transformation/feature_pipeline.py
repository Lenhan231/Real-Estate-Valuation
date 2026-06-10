from air_quality import get_aqi


# 1) dedupe coordinates first
coords = (
    df[["lat", "lon"]]
    .dropna()
    .drop_duplicates()
    .reset_index(drop=True)
)

# 2) fetch in parallel
results = []
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [
        executor.submit(get_aqi, row.lat, row.lon)
        for row in coords.itertuples(index=False)
    ]

    for fut in as_completed(futures):
        results.append(fut.result())

aqi_df = pd.DataFrame(results)

# 3) merge back to original df
df = df.merge(aqi_df, on=["lat", "lon"], how="left", suffixes=("", "_new"))