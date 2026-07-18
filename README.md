# 🏠 House Price Prediction - 6-Bucket Ensemble (MAPE 13.47%)

An end-to-end Automated Valuation Model (AVM) pipeline for the Vietnamese real estate market (specifically Ho Chi Minh City). The production model uses a 6-bucket segmented router ensembling **LightGBM** and **CatBoost** to capture localized market dynamics and mitigate extreme pricing variance.

---

## ⚙️ Setup

### 1. Environment Variables
Create a `.env` file based on `.env.example`:

```bash
# Copy template
cp .env.example .env

# Edit .env and add your Supabase credentials:
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key
SUPABASE_TABLE=Raw_Features
```

**⚠️ Important:** Never commit `.env` to git. It's in `.gitignore` for security.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. ETL Pipeline (Data Ingestion & Processing)
Crawls real estate listings, geocodes addresses with caching, queries Points of Interest (POIs) using spatial indices, and uploads coordinates to Supabase:
```bash
python main.py --start-page 1 --end-page 50
```

### 2. Model Training & Pipeline Serialization
All model code and training pipelines reside in the `Models/` directory.
```bash
cd Models
python train_xgboost.py      # Train LightGBM + CatBoost 6-bucket ensemble
python train_tabpfn.py       # Train TabPFN comparative baseline models
python save_meta.py          # Process final dataset and save lookup metadata
```

### 3. Web UI & BI Dashboard
To launch the user interface for live appraisals or explore the analytics dashboard:
```bash
# From the project root
streamlit run app/app.py          # Interactive appraisal form
streamlit run app/dashboard.py    # Market analysis dashboard
```

---

## 📁 Repository Structure

```text
Real-Estate-Valuation/
├── README.md                    ← You are here
├── main.py                      # Main ETL pipeline orchestration
├── requirements.txt             # Project dependencies
├── .env                         # Configuration & API secrets
│
├── models/                      ← ALL MODEL DEVELOPMENT & ARTIFACTS
│   ├── train_xgboost.py         # Script: Trains LightGBM + CatBoost ensemble
│   ├── train_tabpfn.py          # Script: Trains TabPFN baseline models
│   ├── save_meta.py             # Script: Generates locality lookup metadata
│   ├── data/                    # Local training & validation datasets
│   │   ├── raw_data.csv
│   │   ├── alonhadat_features_cleaned.csv
│   │   └── model_ready_data.csv
│   └── *.pkl                    # Serialized model binaries (12 files for ensemble)
│
├── data/                        ← DATASETS & PROCESSED DATA
│   ├── raw/                     # Raw scraped data
│   │   ├── alonhadat_listings.csv
│   │   └── alonhadat_details.csv
│   ├── processed/               # Cleaned & feature-engineered datasets
│   │   ├── alonhadat_features.csv
│   │   ├── alonhadat_features_cleaned.csv
│   ├── external/                # External reference data
│   │   └── density_data.csv     # Population density data Crawling from Wikipedia
│   └── localities.csv           # Location metadata
│
├── app/                         ← STREAMLIT UI & SERVING LAYER
│   ├── app.py                   # Property appraisal user interface
│   ├── dashboard.py             # BI Market dashboard
│   ├── inference.py             # Single-property feature pipeline & router
│   ├── geo.py                   # Spatial queries & local geocode lookup
│   ├── api.py                   # Full REST API
│   ├── api_simple.py            # Simple REST API
│   └── README.md                # UI application guide
│
├── docs/                        ← ARCHITECTURE & REFERENCE DOCUMENTATION
│   ├── INDEX.md                 # Documentation catalog
│   ├── MODEL_SPLITTING_GUIDE.md # Property type segmentation & outlier guidelines
│   ├── IMPLEMENTATION_SUMMARY.md# Tech stack and pipeline phases
│   ├── REFACTORING_GUIDE.md     # Code refactoring guidelines
│   └── README.md                # Documentation overview
│
└── pipeline/                    ← ETL PIPELINE MODULES
    ├── supabase_handler.py      # Supabase cloud database interface
    ├── ingestion/               # Data scrapers and loaders
    │   ├── load_density.py      # Load population density data
    │   ├── load_pois.py         # Load Points of Interest
    │   └── scrapers/            # Web scrapers
    │       ├── Alonhadat/
    │       │   ├── link_each_status.py
    │       │   ├── link_to_details.py
    │       │   └── scheduling.py
    │       └── __init__.py
    └── transformation/          # Data cleaning & feature engineering
        ├── cleaning.py          # Data validation and cleaning
        ├── feature_pipeline.py  # Core feature engineering
        ├── poi_features.py      # POI-based features
        ├── metro_features.py    # Metro station features
        └── overpass_client.py   # OpenStreetMap/Overpass API client
```

---

## 📊 Performance & Modeling

Rather than fitting a single global model, the dataset is dynamically segmented into **6 specialized sub-market buckets** based on property type and estimated price tier to resolve high pricing variance:

* **Property Type Splits:** Alley houses (*nhà trong hẻm*) vs. Frontage houses (*nhà mặt tiền*)
* **Price Tier Splits:** Low (0-5B VND), Mid (5-20B VND), and High (>20B VND)

### Model Evaluation Results
* **Global $R^2$**: `0.9138` (Model explains 91.3% of HCMC price variance)
* **Global MAPE**: `13.47%`
* **Low-segment (0-5B VND) MAPE**: `10.48%` (Close to the 10% target threshold)

### Data Pruning & Outlier Bounds
To prevent ultra-luxury outliers from distorting target learning, the pipeline applies segment-specific IQR fences (×1.5 for frontage, ×3.0 for alleys) alongside strict baseline boundaries:
* **Prices:** Bounded between `2.0B` and `50.0B` VND
* **Property Area:** Capped between `15m²` and `500m²`
* **Price per m²:** Bounded between `30M` and `800M` VND/m²

---

## 🛠️ Developer Checklist

* **To run full ETL:** `python main.py`
* **To retrain ensemble:** `cd Models && python train_xgboost.py`
* **To rebuild metadata:** `cd Models && python save_meta.py`
* **To launch UI:** `streamlit run app/app.py`
* **To launch REST API:** `python app/api_simple.py`
