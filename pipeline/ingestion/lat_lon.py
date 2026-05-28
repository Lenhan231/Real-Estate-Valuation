import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time

geolocator = Nominatim(user_agent="housing_project")

# Hồ Chí Minh CBD (Quận 1)
HCM_CENTER = (10.7769, 106.7009)
HN_CENTER = (21.0285, 105.8542)

failed = []

def geocode_with_fallback(row):
    candidates = [
        f"{row['street']}, {row['locality']}, {row['region']}",
        f"{row['street']}, {row['region']}",
        f"{row['locality']}, {row['region']}",
        f"{row['region']}"
    ]

    for addr in candidates:
        try:
            time.sleep(1)
            location = geolocator.geocode(addr, timeout=10)
            if location:
                return pd.Series([location.latitude, location.longitude, addr])
        except:
            pass

    return pd.Series([None, None, None])

df["full_address"] = (
    df["street"].fillna("").str.strip() + ", " +
    df["locality"].fillna("").str.strip() + ", " +
    df["region"].fillna("").str.strip()
)

df[["lat", "lon", "matched_address"]] = df.apply(geocode_with_fallback, axis=1)

df["distance_to_center_km"] = df.apply(
    lambda r: geodesic(
        (r["lat"], r["lon"]),
        HCM_CENTER if str(r["region"]).strip().lower() in ["hồ chí minh", "ho chi minh"]
        else HN_CENTER
    ).km
    if pd.notnull(r["lat"]) and pd.notnull(r["lon"])
    else None,
    axis=1
)

# Save
df.to_csv("khoangcach.csv", index=False)

print(df[[
    "full_address",
    "lat",
    "lon",
    "distance_to_center_km"
]].head())