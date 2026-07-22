# 🏠 Real Estate Valuation - DSP391m Capstone Project

**Automated real estate price prediction for Ho Chi Minh City properties**

Predicts property prices using Machine Learning based on area, location, amenities, and geospatial features.

| Metric | Value |
|--------|-------|
| **Model** | XGBoost |
| **MAPE** | 18.01% |
| **R²** | 0.8663 |
| **Dataset** | 10,432 properties |
| **Features** | 166 engineered |

---

## 📁 Repository Organization

### `/data` — Data Layers
- `raw/` — Source exports from scraping / ingestion
- `processed/` — Cleaned data and model-ready datasets
- `external/` — Reference data (density, POI)
- `cache/` — Local cache for repeat lookups

**Why separate?** Keep source data, cleaned data, and train-ready data distinct so EDA and training do not mix responsibilities.

### `/pipeline` — Data Processing (ETL)
```
pipeline/
├── ingestion/           ← Data scrapers & loaders
├── transformation/      ← Feature engineering (166 features from 11 base)
├── supabase_handler.py  ← Fetch data from cloud database
└── cache_handler.py     ← Local cache management
```

**What it does:**
1. Load ingested data from Supabase `Raw_Features`
2. Clean outliers, duplicates, and invalid values
3. Engineer model features
4. Save cleaned and model-ready datasets to `/data/processed/`

See [DATA.md](DATA.md) for detailed pipeline steps.

### `/notebooks` — Analysis & Training
```
notebooks/
├── 01_eda/
│   ├── 01_eda_complete.ipynb    ← Exploratory data analysis
│   └── output/                  ← Generated visualizations
└── README.md
```

**01_eda:** Explore `Raw_Features` first, then compare against cleaned data  
Training now lives in `models/scripts/`

### `/models` — Trained ML Models
```
models/
├── production/          ← XGBoost (best model) ⭐
│   ├── production_model.pkl
│   └── model_results.csv
├── archive/             ← Previous models
│   ├── lightgbm_model.pkl (18.76% MAPE)
│   └── catboost_model.pkl (19.52% MAPE)
└── README.md           ← Model usage guide
```

**Why organized?** Keep production separate from experiments. Never delete working models.

### `/app` — User Interfaces
```
app/
├── app_simple.py        ← Streamlit web app 🏠
├── dashboard.py         ← BI dashboard 📊
├── api_simple.py        ← REST API 🔌
├── inference_simple.py  ← Model wrapper
└── geo.py              ← Geospatial helpers
```

---

## 🚀 Quick Start

**Start both API and Streamlit UI with one command:**

**Windows (PowerShell):**
```bash
.\run.ps1
```

**Linux/Mac (Bash):**
```bash
bash run.sh
```

Then open:
- 📡 **API Docs**: http://localhost:8000/docs (Swagger)
- 🎨 **Web UI**: http://localhost:8500 (Streamlit)

**Or run manually (2 terminals):**

Terminal 1:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2:
```bash
streamlit run app/ui/streamlit_app.py --server.port 8500
```

---

## 🏗️ Project Architecture

```
FastAPI Backend (app/main.py)
    ├── routers/      (API endpoints)
    ├── services/     (Business logic)
    ├── schemas/      (Pydantic models)
    └── core/         (ML models, config)
         
Streamlit Frontend (app/ui/streamlit_app.py)
    └── Calls API endpoints via requests
```

See [app/README.md](app/README.md) for detailed architecture documentation.

---

## 📚 How to Use

### 1. Run Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_eda/01_eda_complete.ipynb
```
Analyzes 12,814 properties: price ranges, feature correlations, geographic clusters.

### 2. Train Models
```bash
python models/scripts/train_ensemble.py --data-source supabase
```
Trains the 6-bucket ensemble on Supabase data and saves the best model to `models/production/`.

### 3. Use Web App
```bash
streamlit run app/app_simple.py
# Visit: http://localhost:8501
```
Input: area, floors, bedrooms, type → Output: predicted price + confidence interval

### 4. Use BI Dashboard
```bash
streamlit run app/dashboard.py
```
Visualize market trends, price heatmaps, top localities.

### 5. Use REST API
```bash
python app/api_simple.py
# POST http://localhost:5000/api/predict
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| [DATA.md](DATA.md) | Data pipeline: sources → cleaning → features → cache |
| [MODELS.md](MODELS.md) | Model training: architecture, hyperparameters, results |
| [notebooks/README.md](notebooks/README.md) | How to run EDA & training notebooks |
| [models/README.md](models/README.md) | Model selection & usage examples |

---

## 🔄 Data Processing Pipeline

```
Raw source files
    ↓ [ingestion]
Supabase Raw_Features
    ↓ [EDA / cleaning]
Cleaned dataset
    ↓ [feature engineering / model prep cache]
model_ready_data.csv
    ↓ [model training]
Production model
    ↓
Web App / Dashboard / API
```

**Why this structure?**
- Clear separation between data, processing, models, apps
- Each folder has one responsibility
- Easy to trace where errors come from
- Documentation explains the "why", not just "what"
 - `model_ready_data.csv` is a derived cache for inspection, not the source of truth for training

---

## 🤖 Model Performance

**Best Model: XGBoost**
- Training: 8,345 properties (80%)
- Testing: 2,087 properties (20%)
- Features: 166 engineered features
- Target: Property price (log-transformed)

**Metrics:**
- MAPE: 18.01% (mean absolute percentage error)
- R²: 0.8663 (explains 86.63% of price variance)
- MAE: 2.67B VND (±1.33B @ 95% confidence)

**Compared to:**
- LightGBM: 18.76% MAPE
- CatBoost: 19.52% MAPE

---

## 🚨 Troubleshooting

**"Model not found"**
```bash
ls models/production/production_model.pkl
# If missing, re-run: `python models/scripts/train_ensemble.py --data-source supabase`
```

**"Feature mismatch"**
- Ensure `/pipeline/transformation/feature_pipeline.py` hasn't changed
- Re-run training notebook to regenerate model

**"Supabase connection error"**
- Check `.env` file has correct `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`
- Verify Supabase table exists and has data

---

## 📋 Capstone Deliverables

| Item | Status |
|------|--------|
| Data Pipeline | ✅ Complete |
| EDA Analysis | ✅ Complete |
| Model Training | ✅ Complete |
| Web App | ✅ Live |
| BI Dashboard | ✅ Ready |
| REST API | ✅ Ready |
| Organization | ✅ Clear structure |
| Research Paper | ⏳ Next |

---

**Last Updated:** 2026-07-18  
**Model Version:** v1.0 (XGBoost)  
**Data Source:** alonhadat.com (web scraping)
