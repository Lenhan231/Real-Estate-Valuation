# 📊 Data Directory

Complete guide to data layers: raw ingestion → processing → training ready.

---

## 🗂️ Directory Structure

```
data/
├── README.md                          ← This file
├── raw/                               ← Source data (never modify)
│   ├── alonhadat_listings.csv         (Property listings - metadata)
│   └── alonhadat_details.csv          (Full property details - main source)
│
├── processed/                         ← Cleaned & feature-engineered data
│   ├── alonhadat_cleaned.csv          (After cleaning, before features)
│   ├── model_training_data.csv        (Ready for ML - 78 features)
│   └── model_training_data_*.csv      (Versioned backups)
│
├── external/                          ← Reference & static data
│   ├── density_data.csv               (Locality population & area density)
│   └── location_reference.csv         (Locality coordinates & regions)
│
└── cache/                             ← Computed cache (regeneratable)
    └── localities.csv                 (POI features by locality - reused)
```

---

## 📋 Data Layers Explained

### 1. **raw/** — Source Data (Immutable)

**Files:**
- `alonhadat_listings.csv` - Listing metadata
- `alonhadat_details.csv` - Complete property data (~100MB)

**Schema:**
```csv
id,url,old_address,street,locality,region,price,area_m2,width_m,
length_m,num_floors,num_bedrooms,legal_status,direction,
property_type,posted_date,last_updated
```

**Characteristics:**
- ✓ Direct from web scraper (alonhadat.com)
- ✓ 10,432+ properties
- ✓ Some missing values (normal)
- ✓ Possible duplicates (handled in cleaning)
- ✓ Raw text (no normalization)

**DO NOT EDIT** - Archive only, source of truth for reproducibility.

**Size:** ~100MB (uncompressed)

---

### 2. **processed/** — Cleaned & Engineered Data

#### `alonhadat_cleaned.csv`

**After cleaning, before features.**

**Transformations Applied:**
- ✓ Remove duplicates (by address + price)
- ✓ Drop rows with missing coordinates
- ✓ Remove price outliers (>500M VND)
- ✓ Normalize text fields (lowercase, trim whitespace)
- ✓ Fix data types (strings → numeric where needed)

**Size:** ~50MB

---

#### `model_training_data.csv`

**FINAL TRAINING DATASET - Ready for ML models.**

**Includes:**
- ✓ All cleaned data from alonhadat_cleaned.csv
- ✓ 78 engineered features (see Feature List below)
- ✓ Density statistics merged
- ✓ No missing values in key columns
- ✓ Normalized numeric features

**Columns (Partial):**
```
address, street, locality, region, price, area_m2, ...
nearest_school_km, school_count_3km,
nearest_hospital_km, hospital_count_5km,
nearest_metro_km, metro_count_5km,
locality_population_density, locality_area_km2,
... (78 total features)
```

**Used By:**
- Model training (`models/scripts/train_production.py`)
- Model evaluation & testing
- Prediction API (at runtime uses features extracted from input)

**Size:** ~50MB

**Update Frequency:** When new data scraped, re-run `pipeline/run.py`

---

### 3. **external/** — Reference Data (Static)

#### `density_data.csv`

**Locality-level population and area statistics.**

**Schema:**
```csv
locality, region, locality_population, locality_area_km2,
population_density_per_km2
```

**Source:** Vietnam statistics (CSO)

**Used For:** Merge with processed data for density features

**Update Frequency:** Rarely (annual government releases)

---

#### `location_reference.csv` (Optional)

**Locality geo-coordinates and region mapping.**

**Schema:**
```csv
locality, region, center_lat, center_lon, area_km2
```

**Source:** Generated from Nominatim/OpenStreetMap

---

### 4. **cache/** — Computed Cache (Regeneratable)

#### `localities.csv`

**Pre-computed POI features by locality to avoid repeated API calls.**

**Schema:**
```csv
street, locality, region, lat, lon, old_address,
nearest_school_km, school_count_3km,
nearest_hospital_km, hospital_count_5km,
... (POI features)
```

**Purpose:**
- Speed up inference (avoid re-querying OpenStreetMap)
- Consistency across runs
- Fallback when API unavailable

**Updated:** After each pipeline run

**Size:** ~10MB (1,000+ unique localities)

**Can be safely deleted** → Will be regenerated on next `pipeline/run.py`

---

## 🔄 Data Processing Pipeline

```
┌─────────────────────────────┐
│  raw/alonhadat_details.csv  │  (raw web scraper output)
│  (~100MB, messy)            │
└──────────────┬──────────────┘
               │
               ▼
        ┌──────────────┐
        │   CLEANING   │
        │ Remove dups  │
        │ Fix types    │
        └──────┬───────┘
               │
               ▼
┌─────────────────────────────┐
│ processed/cleaned.csv       │  (after cleaning)
│ (~50MB, valid data)         │
└──────────────┬──────────────┘
               │
               ▼
        ┌──────────────┐
        │  GEOCODING   │
        │  Add lat/lon │
        │  (w/ cache)  │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │  POI LOOKUP  │
        │  Schools,    │
        │  hospitals   │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │  FEATURE     │
        │  ENGINEERING │
        │  78 features │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │  DENSITY     │
        │  MERGE       │
        └──────┬───────┘
               │
               ▼
┌─────────────────────────────┐
│ processed/model_training_.. │  (final training data)
│ data.csv (~50MB, ready)     │
└─────────────────────────────┘
         ↓
   ┌─────────────────┐
   │  Training/Test  │
   │  ML Models      │
   └─────────────────┘
```

---

## 📊 Data Statistics

### raw/alonhadat_details.csv

| Metric | Value |
|--------|-------|
| **Records** | 10,432 |
| **Columns** | 17 |
| **Size** | ~100MB |
| **Duplicates** | ~50-100 |
| **Missing (%)** | 5-15% per column |
| **Price range** | 1B - 500B VND |
| **Area range** | 10 - 1000 m² |

### processed/model_training_data.csv

| Metric | Value |
|--------|-------|
| **Records** | 10,300 (after cleaning) |
| **Columns** | 78 (engineered) |
| **Size** | ~50MB |
| **Duplicates** | 0 |
| **Missing** | <1% |
| **Price median** | 14.5B VND |
| **Area median** | 85 m² |

---

## 📈 Feature List (78 Total)

### Base Features (11)
- `price` - Property price (VND)
- `area_m2` - Total area (m²)
- `width_m` - Width (m)
- `length_m` - Length (m)
- `num_floors` - Number of floors
- `num_bedrooms` - Bedrooms
- `road_width_m` - Road width (m)
- `property_type` - Category (apartment, house, etc.)
- `legal_status` - Sổ hồng, sổ đỏ, etc.
- `direction` - Facing direction
- `posted_date` - Days since posted

### POI Features (14)
- `nearest_school_km` - Distance to nearest school
- `school_count_3km` - Schools within 3km
- `nearest_hospital_km` - Distance to nearest hospital
- `hospital_count_5km` - Hospitals within 5km
- `nearest_marketplace_km`
- `marketplace_count_3km`
- `nearest_supermarket_km`
- `supermarket_count_3km`
- `nearest_mall_km`
- `mall_count_3km`
- `nearest_bus_stop_km`
- `bus_stop_count_1km`
- `nearest_metro_km`
- `metro_count_5km`

### Geographic Features (15)
- `lat`, `lon` - Coordinates
- `distance_to_center_km` - Distance to Saigon center
- `locality_population` - Locality population
- `locality_area_km2` - Locality area
- `population_density_per_km2` - Density
- `region_north`, `region_south`, `region_center` - Region dummies (12)

### Engineered Features (38)
- Price/area interactions
- Polynomial features (area², distance²)
- Temporal features (month, season)
- Locality statistics
- Categorical encodings
- ... (see `pipeline/transformation/feature_pipeline.py` for full list)

---

## 🔄 Data Versioning

**Backup versioned datasets:**

```bash
# Archive old training data
cp data/processed/model_training_data.csv \
   data/processed/model_training_data_2026-07-23.csv

# Keep last 3 versions
ls -t data/processed/model_training_data_*.csv | tail -n +4 | xargs rm
```

---

## 📥 Importing Data

### In Python

```python
import pandas as pd

# Load training data
df = pd.read_csv('data/processed/model_training_data.csv')

# Load raw data (for debugging)
raw = pd.read_csv('data/raw/alonhadat_details.csv')

# Load cache
cache = pd.read_csv('data/cache/localities.csv')
```

### From Database

```python
from pipeline.supabase_handler import load_from_supabase

# Load from Supabase
df = load_from_supabase('Raw_Features')
```

---

## 🛡️ Data Quality Checklist

Before using data:

- [ ] `model_training_data.csv` exists
- [ ] No NaN in key columns (price, area, lat, lon)
- [ ] 10,000+ records
- [ ] All 78 features present
- [ ] Price > 0 and < 500B
- [ ] Area > 10 m²
- [ ] Valid lat/lon (-90 to 90, -180 to 180)

```python
# Quick validation
def validate_training_data(df):
    assert len(df) > 10000, "Too few records"
    assert df.isnull().sum().max() < 10, "Too many NaN"
    assert df['price'].min() > 0, "Invalid prices"
    return True
```

---

## 🔐 Privacy & Ethics

### Data Considerations

- ✓ Property listings are public (scraped from public website)
- ✓ Price data is publicly available
- ✓ No personal information collected
- ✓ Aggregated for BI/analytics only

### Usage

- ✓ Use for model training & research
- ✓ Respect copyright of alonhadat.com
- ❌ Do not resell raw data
- ❌ Do not republish without attribution

---

## 📚 Related Documentation

- [pipeline/README.md](../pipeline/README.md) - How data is processed
- [MODELS.md](../MODELS.md) - Features used in training
- [notebooks/README.md](../notebooks/README.md) - Data analysis notebooks

---

## 🚨 Common Issues

### "ModuleNotFoundError in pipeline"
→ Run from project root: `python pipeline/run.py`

### "Supabase connection error"
→ Check `.env`: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`

### "Missing localities.csv"
→ Run pipeline: `python pipeline/run.py --mode incremental`

### "File too large to load"
→ Sample: `pd.read_csv('file.csv', nrows=10000)`

### "Encoding error"
→ Specify encoding: `pd.read_csv('file.csv', encoding='utf-8')`

---

**Last Updated:** 2026-07-23
**Data Version:** v2.6 (training data with 78 features)
