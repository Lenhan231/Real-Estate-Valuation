# House Price Prediction

Real estate valuation ETL pipeline with geospatial feature engineering.

## Quick Start

```bash
# Install dependencies
pip install -r docs/requirements.txt

# Download POI data (first run only)
python pipeline/ingestion/download_pois.py

# Run pipeline
python main.py
```

## Documentation

See [docs/](docs/) for detailed documentation:

- **[Documentation Index](docs/INDEX.md)** — All docs and quick reference
- **[Project README](docs/README.md)** — Full project overview
- **[Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)** — Architecture & technical details
- **[Refactoring Guide](docs/REFACTORING_GUIDE.md)** — Code improvements & guidelines
- **[Model Splitting Guide](docs/MODEL_SPLITTING_GUIDE.md)** — Property type splitting, IQR outlier filtering, and model leaderboards

## Features

- **Fast Geocoding** — Local mapping + Nominatim API with persistent cache
- **POI Features** — Schools, hospitals, supermarkets, malls, transit (via BallTree spatial indices)
- **Batch Processing** — Efficient pipeline with checkpointing (50 records/batch)
- **Incremental Output** — Saves progress after each batch to prevent data loss

## Pipeline Steps

1. Load & clean real estate data
2. Add density features
3. Geocode addresses (with caching)
4. Extract geospatial POI features in batches
5. Save final dataset with all engineered features

## Performance

- **First run**: ~10-15 min (includes geocoding API calls)
- **Subsequent runs**: ~2-5 min (uses cached coordinates)

Geocoding cache is stored at `data/geocode_cache.csv`.
