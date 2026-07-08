# House Price Prediction - ETL Pipeline Implementation

## 📋 Architecture Overview

The project implements an ETL (Extract, Transform, Load) pipeline for Vietnamese real estate data from Alonhadat.com with geospatial feature engineering.

**Processing Flow:**
```
[1/5] Web Scraping      → Extract listings from Alonhadat
[2/5] Data Cleaning     → Validate, deduplicate records
[3/5] Base Features     → Add density, geocoding
[4/5] Geospatial        → Query POI/metro distances
[5/5] Output            → Push to Supabase + CSV
```

---

## 🏗️ System Architecture

### Data Pipeline Layers

**Layer 1: Ingestion** (`pipeline/ingestion/`)
- **`scrapers/Alonhadat/`** - Web scraper for real estate listings
  - `link_each_status.py` - Parse listing pages
  - `scheduling.py` - Orchestrate multi-page crawls
  - `link_to_details.py` - Extract property detail pages
- **`load_density.py`** - Geocoding with caching via geopy
- **`load_pois.py`** - Coordinate transformation & city-center distance

**Layer 2: Transformation** (`pipeline/transformation/`)
- **`overpass_client.py`** ⭐ NEW - Base class for Overpass API queries
  - Handles rate limiting, retry logic, coordinate extraction
  - Manages persistent cache from `localities.csv`
- **`poi_features.py`** - Query nearest POIs (schools, hospitals, markets)
  - Inherits from `OverpassAPIClient`
  - Returns distance & count within radius
- **`metro_features.py`** - Query metro stations
  - Inherits from `OverpassAPIClient`
  - Metro-specific Overpass queries
- **`feature_pipeline.py`** - Batch feature computation
- **`cleaning.py`** - Data validation & deduplication

**Layer 3: Output** (`pipeline/supabase_handler.py`)
- Push processed CSV to Supabase (cloud database)

---

## 🔄 Geospatial Feature Extraction

### Current Implementation (API-based)

Each property gets queried for:

**POI Features** (6 types):
- Schools: nearest distance + count within 3km
- Hospitals: nearest distance + count within 5km
- Marketplaces: nearest distance + count within 3km
- Supermarkets: nearest distance + count within 3km
- Malls: nearest distance + count within 3km
- Bus stops: nearest distance + count within 1km

**Transit Features**:
- Metro stations: nearest distance + count within 5km

### Caching Strategy

**Level 1: CSV Cache** (`data/localities.csv`)
- Stores precomputed features indexed by rounded lat/lon (5 decimals)
- Auto-populated on first feature extraction run
- Reused in subsequent runs → ~instant lookups

**Level 2: In-Memory Cache**
- Loaded at module init from CSV
- Fast repeated queries within same process
- Dictionary keyed by `(lat, lon, key, value)` or `(lat, lon, key, value, radius)`

### Query Logic

```python
def get_poi_features(lat, lon, key, value, radius=500):
    # 1. Check in-memory cache
    if (lat, lon, key, value) in cache:
        return cache[(lat, lon, key, value)]
    
    # 2. Query Overpass API
    data = query_overpass(lat, lon, key, value)
    
    # 3. Calculate metrics
    distances = [geodesic(property, poi) for poi in data]
    nearest = min(distances)
    count = len([d for d in distances if d <= radius])
    
    # 4. Store in cache for next run
    cache[(lat, lon, key, value)] = (nearest, count)
    return (nearest, count)
```

---

## 🔧 Code Refactoring (Phase 2)

### A) Base Class Extraction
**Before:** 90+ lines of duplicated code in `poi_features.py` and `metro_features.py`
- `_load_persistent_cache()` (35 lines)
- `_query_overpass_api()` (14 lines)
- `_extract_coordinates()` (5 lines)
- `_calculate_metrics()` (8 lines)
- Rate limit handling, retry logic

**After:** Created `OverpassAPIClient` base class
- Both modules now inherit common functionality
- ~60 lines eliminated
- Easier to maintain & extend for new POI types

### B) Scraper Reorganization
**Before:** `scaper/Alonhadat/` at root level
**After:** `pipeline/ingestion/scrapers/Alonhadat/`
- Clearer separation of concerns
- All ingestion logic colocated under `pipeline/ingestion/`
- Updated imports in `main.py`

### C) Documentation Alignment
This document now reflects **actual implementation**, not planned architecture.

---

## 📊 Performance Characteristics

| Component | Approach | Cost |
|-----------|----------|------|
| Scraping | Live requests to alonhadat.com | ~5-10 min (50 pages) |
| Geocoding | geopy cache check + fallback | ~15 sec for 2500 addresses |
| POI Features | Overpass API + CSV cache | Depends on cache hit rate |
| Feature Pipeline | Row-by-row processing | ~5-10 min per batch |
| **Total** | **End-to-end** | **~30-45 min** |

**Cache Performance:**
- Cache hit: `<1ms` per property
- Cache miss: `500ms-2s` per property (API query + backoff)
- Typical hit rate on repeat runs: `>90%`

---

## 🔐 Data Privacy & Security

- ✅ Supabase credentials in `.env` (git-ignored)
- ✅ Rate-limit handling respects Overpass API terms
- ✅ User-Agent headers identify data processor
- ✅ Exponential backoff on 429/504 responses
- ✅ Cloudflare WARP integration for VPN routing (optional)

---

## 🧪 Testing & Validation

**Current Coverage:**
- ✅ Scraper works on actual Alonhadat site
- ✅ Geocoding returns valid lat/lon
- ✅ POI features fallback gracefully on API errors
- ✅ Cache persistence works across runs
- ✅ Batch checkpointing prevents data loss

**Not Yet Implemented:**
- Unit tests for feature extraction
- Integration tests for full pipeline
- Data quality validation (coordinate bounds, outliers)
- Performance regression testing

---

## 🚀 Usage

### Quick Start
```bash
# Run entire pipeline
python main.py --start-page 1 --end-page 50

# Output: data/processed/alonhadat_features.csv
```

### Partial Runs
```bash
# Just scrape pages 10-20
python main.py --start-page 10 --end-page 20

# Features are incrementally cached in data/localities.csv
```

---

## 📁 File Structure

```
pipeline/
├── ingestion/
│   ├── scrapers/Alonhadat/          ← Web scraping
│   ├── load_pois.py                 ← Geocoding + distance
│   └── load_density.py              ← Demographic data
├── transformation/
│   ├── overpass_client.py           ← Base API client (NEW)
│   ├── poi_features.py              ← POI queries (refactored)
│   ├── metro_features.py            ← Metro queries (refactored)
│   ├── feature_pipeline.py          ← Batch processing
│   └── cleaning.py                  ← Data validation
└── supabase_handler.py              ← Cloud output

data/
├── raw/
│   ├── alonhadat_listings.csv       ← Scraped listings
│   └── alonhadat_details.csv        ← Property details
├── processed/
│   ├── alonhadat_cleaned.csv        ← Cleaned data
│   └── alonhadat_features.csv       ← Final output
├── external/
│   └── density_data.csv             ← External demographics
└── localities.csv                   ← POI feature cache
```

---

## 🔮 Future Improvements

**Phase 3: Optimization**
- [ ] Async Overpass queries (concurrent requests)
- [ ] Parquet output format (smaller files)
- [ ] Support multiple cities (Hanoi, Da Nang, etc.)
- [x] Dashboard visualization

**Phase 4: Production**
- [ ] Unit test suite
- [ ] CI/CD pipeline
- [ ] Scheduled scraper runs
- [ ] Data quality monitoring
- [ ] Model training pipeline

---

**Last Updated:** 2026-07-02  
**Status:** ✅ Phase 2 Complete (Code Refactoring)  
**Next Phase:** Production Optimization
