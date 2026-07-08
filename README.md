# House Price Prediction

A real estate valuation ETL pipeline featuring geospatial feature engineering for property price prediction.

## Quick Start

```bash
# Install dependencies
pip install -r docs/requirements.txt

# Run the pipeline
python main.py
```

> **Note:** Install **Warp CLI** before running the crawler.

## Documentation

Detailed documentation is available in the `docs/` directory:

- **[Documentation Index](docs/INDEX.md)** — All docs and quick reference
- **[Project README](docs/README.md)** — Full project overview
- **[Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)** — Architecture & technical details
- **[Refactoring Guide](docs/REFACTORING_GUIDE.md)** — Code improvements & guidelines
- **[Model Splitting Guide](docs/MODEL_SPLITTING_GUIDE.md)** — Property type splitting, IQR outlier filtering, and model leaderboards

## Features

* **Fast Geocoding** — Hybrid local mapping and Nominatim geocoding with persistent caching
* **Geospatial Feature Engineering** — Distance-based features for schools, hospitals, supermarkets, shopping malls, and public transportation using BallTree spatial indices
* **Batch Processing** — Processes data in configurable batches with checkpoint support
* **Incremental Saving** — Automatically saves progress after each batch to minimize data loss
* **Persistent Geocoding Cache** — Reuses previously geocoded addresses to reduce API requests and improve performance

## Pipeline

1. Load and clean raw real estate data.
2. Generate population density features.
3. Geocode property addresses with cache support.
4. Extract nearby points of interest (POI) features.
5. Save the enriched dataset for downstream analysis and modeling.

## Cache

The geocoding cache is stored at:

```text
data/geocode_cache.csv
```

This cache is automatically updated during pipeline execution and reused in subsequent runs.
