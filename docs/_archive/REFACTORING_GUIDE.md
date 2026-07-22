# POI Feature Engineering Refactoring

## Architecture Change

### Before (❌ Inefficient)
```
feature_pipeline.py
    ↓ (for each row)
poi_features.py → Overpass API call
metro_features.py → Overpass API call
```
- 🐢 **5000+ API calls** per feature engineering run
- ⏱️ **Hours** to complete (rate limits, timeouts)
- 🚫 Fails if API is down

---

### After (✅ Production-Ready)
```
download_pois.py (run once/month)
    ↓
schools.parquet
hospitals.parquet
metro_stations.parquet
... (cached locally)
    ↓
poi_features.py (BallTree indices)
metro_features.py (BallTree indices)
    ↓
feature_pipeline.py (vectorized queries)
```
- ⚡ **Instant** feature computation
- 🔄 **Reproducible** (same data every run)
- 📈 **Scalable** (works with 50k+ records)

---

## How to Use

### Step 1: Download POI Data (⚠️ Run ONCE)

```bash
python pipeline/ingestion/download_pois.py
```

This creates:
```
data/pois/
├── schools.parquet
├── hospitals.parquet
├── marketplaces.parquet
├── supermarkets.parquet
├── malls.parquet
├── bus_stops.parquet
└── metro_stations.parquet
```

**Time:** ~5-10 minutes (API calls)  
**Frequency:** Once per month/quarter (data doesn't change that fast)

---

### Step 2: Feature Engineering (Normal flow)

```python
from pipeline.transformation.feature_pipeline import get_additional_features

df = pd.read_csv("data/raw/houses.csv")
df = get_additional_features(df)  # ⚡ Fast! ~1 second for 5000 rows
```

**No API calls.** Only local BallTree queries.

---

## Technical Details

### POI Features (poi_features.py)
- **Loader:** `POIFeatureExtractor()` loads all parquet files on init
- **Indices:** BallTree (haversine metric) for each POI type
- **Operations:**
  - `get_nearest_distance_and_count(lat, lon, poi_type)` → single point
  - `get_nearest_distances_batch(lats, lons, poi_type)` → vectorized
  - `get_counts_batch(lats, lons, poi_type)` → vectorized

### Metro Features (metro_features.py)
- Same pattern for metro stations only
- `MetroFeatureExtractor()`

### Feature Pipeline (feature_pipeline.py)
- **Vectorized:** Calls batch methods instead of looping
- **Speed:** 10x faster than original `.apply()` version

---

## Troubleshooting

### "Warning: metro_stations.parquet not found"
→ Run `python pipeline/ingestion/download_pois.py` first

### "No POIs found in area"
→ Check bbox in `download_pois.py`. Expand or adjust coordinates.

### Slow download
→ Normal. Overpass API throttles requests. Each POI type takes ~30s-1min.

---

## Architecture Benefits for Capstone

✅ **Data Engineering perspective:**
- Separates ingestion (batch) from transformation (online)
- Demonstrates understanding of data pipelines
- Cacheable + reproducible

✅ **ML perspective:**
- Pre-computed spatial indices (no feature computation delays)
- Vectorized operations (NumPy instead of pandas.apply)
- Production-ready

✅ **Presentation talking points:**
> "External geospatial data was pre-ingested from OpenStreetMap via Overpass API and persisted locally as Parquet files. Feature engineering uses BallTree spatial indices for O(log n) nearest-neighbor queries instead of live API requests, enabling fast reproducible pipelines."

---

## Future Improvements

- [ ] Cache downloaded POIs with timestamps
- [ ] Parallelize Overpass downloads (async)
- [ ] Add data quality checks (min/max coordinates)
- [ ] Support for multiple cities in one run
