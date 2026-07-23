# 🔄 Data Pipeline

Complete ETL (Extract, Transform, Load) pipeline for real estate data processing.

---

## 📊 Pipeline Overview

```
Raw Data (Scraper)
    ↓
[CLEANING] Remove outliers, duplicates, invalid values
    ↓
[GEOCODING] Add coordinates with caching
    ↓
[DISTANCE CALC] Distance to city center
    ↓
[POI FEATURES] Points of interest (schools, hospitals, metro, etc.)
    ↓
[FEATURE ENG] Create 78 optimized features
    ↓
[DENSITY MERGE] Merge with locality density data
    ↓
Processed Data (CSV) → Supabase
```

**Total Processing:** ~10,432 properties with 78 engineered features

---

## 🗂️ Directory Structure

```
pipeline/
├── run.py                          ← Main ETL orchestrator (run this!)
├── __init__.py
├── cache_handler.py                ← Local cache management
├── supabase_handler.py             ← Cloud database operations
│
├── ingestion/                      ← Data acquisition
│   ├── __init__.py
│   ├── load_density.py             ← Load locality density stats
│   ├── load_pois.py                ← Add coordinates & POI features
│   └── scrapers/
│       └── Alonhadat/              ← Web scraper for alonhadat.com
│           ├── scheduling.py       ← Scraping scheduler
│           ├── link_each_status.py ← Extract listing links
│           └── link_to_details.py  ← Scrape property details
│
└── transformation/                 ← Feature engineering
    ├── __init__.py
    ├── cleaning.py                 ← Data validation & cleaning
    ├── feature_pipeline.py         ← POI & distance features
    ├── metro_features.py           ← Metro/transportation features
    ├── overpass_client.py          ← Query OpenStreetMap API
    └── poi_features.py             ← Points of interest extraction
```

---

## 🚀 Running the Pipeline

### Quick Start

```bash
# From project root
python pipeline/run.py

# Or with arguments
python pipeline/run.py --mode full  # Full reload (slow)
python pipeline/run.py --mode incremental  # Incremental (default, fast)
```

### What It Does

1. **Loads** raw data from `data/raw/alonhadat_details.csv`
2. **Cleans** invalid addresses, duplicates, outliers
3. **Geocodes** addresses using Nominatim API (with local cache)
4. **Calculates** distance to Saigon center (10.7769, 106.7009)
5. **Extracts** POI features:
   - Nearest school/hospital/marketplace/supermarket/mall/bus stop/metro
   - Count of amenities within radius (1-5km)
6. **Engineers** 78 features (polynomial interactions, normalization)
7. **Merges** density data (population, land area per locality)
8. **Saves** to `data/processed/model_training_data.csv`
9. **Uploads** to Supabase `Raw_Features` table

---

## 📁 Input/Output Files

### Inputs

| File | Source | Size | Description |
|------|--------|------|-------------|
| `data/raw/alonhadat_details.csv` | Web scraper | ~100MB | Raw property listings |
| `data/external/density_data.csv` | External | ~100KB | Locality density stats |
| `data/cache/localities.csv` | Cache | ~10MB | Pre-computed POI features |

### Outputs

| File | Destination | Size | Description |
|------|-------------|------|-------------|
| `data/processed/model_training_data.csv` | Local | ~50MB | Cleaned + featured dataset |
| `Supabase.Raw_Features` | Cloud DB | Indexed | Production data source |
| `data/cache/localities.csv` | Local | ~10MB | Updated POI cache |

---

## ⚙️ Core Modules

### 1. `ingestion/` — Data Acquisition

#### `scrapers/Alonhadat/`
Web scraper for **alonhadat.com** (Vietnamese real estate site).

```python
from pipeline.ingestion.scrapers.Alonhadat.scheduling import crawl_list_pages

# Scrape all listings
crawl_list_pages(
    start_page=1,
    end_page=100,
    output_file="data/raw/alonhadat_details.csv"
)
```

**Features:**
- Pagination handling
- Duplicate detection
- Error recovery
- Rate limiting

#### `load_pois.py`
Add geographic features using Nominatim & OpenStreetMap.

```python
from pipeline.ingestion.load_pois import add_coordinates, distance_to_center

# Geocode addresses
df = add_coordinates(df, cache_file="data/cache/localities.csv")

# Calculate distances
df = distance_to_center(df, center=(10.7769, 106.7009))
```

#### `load_density.py`
Load and merge locality-level statistics.

```python
from pipeline.ingestion.load_density import merge_density_with_alonhadat

df = merge_density_with_alonhadat(
    data_df,
    density_file="data/external/density_data.csv"
)
```

---

### 2. `transformation/` — Feature Engineering

#### `cleaning.py`
Data validation and cleaning.

```python
from pipeline.transformation.cleaning import clean_data, final_clean

# Initial cleaning
df = clean_data(df)

# Final validation before training
df = final_clean(df)
```

**Operations:**
- Remove NaN values
- Drop duplicates (by address + price)
- Remove outliers (price > 500M VND)
- Format normalization

#### `feature_pipeline.py`
Main feature engineering orchestrator.

```python
from pipeline.transformation.feature_pipeline import get_additional_features

# Generate all features
features = get_additional_features(
    row,
    geo_lookup=geo_lookup,  # For address matching
    cache=cache,            # For POI caching
)
```

**78 Features:**
- 64 base (price, area, location, amenities, type)
- 14 polynomial/interaction (area², price/m², etc.)

#### `poi_features.py`
Extract points of interest features.

```python
from pipeline.transformation.poi_features import extract_poi_features

pois = extract_poi_features(
    lat, lon,
    radius_km=3,
    poi_types=['school', 'hospital', 'metro']
)
```

**POI Types:**
- Amenities: school, hospital, marketplace, supermarket, mall
- Transportation: bus stop, metro station
- Radius: 1-5km depending on type

#### `metro_features.py`
Specialized features for metro/transportation.

```python
from pipeline.transformation.metro_features import get_metro_features

metro_dist, metro_count = get_metro_features(lat, lon)
```

---

### 3. `cache_handler.py` — Smart Caching

Avoid re-computing features by caching locality-level results.

```python
from pipeline.cache_handler import LocalityCache

cache = LocalityCache("data/cache/localities.csv")

# Check cache first
if cache.has(locality, region):
    features = cache.get(locality, region)
else:
    # Compute and cache
    features = compute_poi_features(...)
    cache.put(locality, region, features)
    cache.save()  # Save to disk
```

**Cache Strategy:**
1. ✅ Exact match (old_address) → instant
2. ✅ Locality + region match → instant
3. ⏱️ API fallback → cache result
4. ❌ Missing → drop row

---

### 4. `supabase_handler.py` — Cloud Database

Upload processed data to Supabase (PostgreSQL).

```python
from pipeline.supabase_handler import push_csv_to_supabase

# Upload cleaned dataset
push_csv_to_supabase(
    csv_file="data/processed/model_training_data.csv",
    table_name="Raw_Features",
    batch_size=100
)
```

**Supabase Tables:**
- `Raw_Features` - All property data with features
- `address_cache` - Cached coordinates & POI
- `listings` - Original listings (before feature engineering)

---

## 🔧 Configuration

Edit `pipeline/run.py` to customize:

```python
# Processing parameters
BATCH_SIZE = 2              # Records per checkpoint
OUTPUT_FILE = "data/processed/alonhadat_features.csv"
CACHE_FILE = "data/cache/localities.csv"

# Geocoding (Nominatim)
NOMINATIM_TIMEOUT = 10     # seconds
CACHE_RADIUS = 3            # km for POI lookup

# Feature engineering
POI_TYPES = [
    'school', 'hospital', 'marketplace',
    'supermarket', 'mall', 'bus_stop', 'metro'
]
RADIUS_KM = {
    'school': 3,
    'hospital': 5,
    'metro': 5,
}
```

---

## 📊 Performance

### Typical Run Times

| Stage | Records | Time |
|-------|---------|------|
| Load raw | 10,432 | 5s |
| Clean | 10,432 | 10s |
| Geocode (w/ cache) | 10,432 | 30s |
| POI features (API) | 10,432 | 5-10m |
| Feature engineering | 10,432 | 20s |
| Density merge | 10,432 | 10s |
| Supabase upload | 10,432 | 2-3m |
| **TOTAL** | | **10-15m** |

**First run:** Slower (API calls, no cache)
**Incremental:** Faster (cached features)

### Memory Usage
- ~500MB for 10k records with 78 features
- In-memory processing (no chunking)
- Can scale to 50k+ records with chunks

---

## 🐛 Troubleshooting

### "API rate limited"
```python
# Increase delay between requests
import time
time.sleep(1)  # 1 second between Nominatim calls
```

### "Memory error"
```python
# Process in batches
BATCH_SIZE = 500  # Process 500 records at a time
```

### "Supabase upload fails"
```python
# Check connection
from pipeline.supabase_handler import test_connection
test_connection()  # Returns True if OK

# Check .env variables
# SUPABASE_URL, SUPABASE_SERVICE_KEY must be set
```

### "Cache file corrupted"
```bash
# Remove and rebuild
rm data/cache/localities.csv
python pipeline/run.py  # Will rebuild from scratch
```

---

## 📈 Monitoring

### Check Progress
```bash
# Watch the output CSV grow
watch -n 5 'wc -l data/processed/model_training_data.csv'

# Monitor memory
top  # Press 'p' to sort by Python process
```

### Verify Quality
```python
# Load processed data
import pandas as pd
df = pd.read_csv("data/processed/model_training_data.csv")

# Check stats
print(df.describe())
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
```

---

## 🚨 Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| "Nominatim API timeout" | Network slow or API down | Increase `NOMINATIM_TIMEOUT` or retry |
| "SSL certificate error" | Python SSL config | `pip install certifi` |
| "Out of memory" | Too many records in RAM | Reduce `BATCH_SIZE` |
| "Column not found" | Schema mismatch | Check raw CSV columns |
| "Supabase auth failed" | Wrong credentials | Update `.env` |

---

## 🔄 Data Flow Diagram

```
┌─────────────────────┐
│  Web Scraper        │
│  (alonhadat.com)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Raw CSV            │
│  (~100MB, messy)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  [CLEANING]         │
│  Remove outliers    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  [GEOCODING]        │
│  Add lat/lon        │
│  (Nominatim cache)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  [POI FEATURES]     │
│  Schools, hospitals │
│  (OpenStreetMap)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  [FEATURE ENG]      │
│  78 engineered      │
│  features           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  [DENSITY MERGE]    │
│  Add locality stats │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Processed CSV      │
│  (~50MB, clean)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Supabase Upload    │
│  Raw_Features table │
└─────────────────────┘
```

---

## 📚 Related Documentation

- [Main README](../README.md) - Project overview
- [DATA.md](../DATA.md) - Data sources & structure
- [MODELS.md](../MODELS.md) - Model training

---

**Last Updated:** 2026-07-23
**Maintainer:** DSP391m Capstone Team
