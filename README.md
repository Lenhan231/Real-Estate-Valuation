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

### `/data` — Input Data & Features
- `raw/` — Original web-scraped listings (12.8k properties)
- `processed/` — Cleaned & feature-engineered datasets (10.4k after filtering)
- `external/` — Reference data (density, POI)
- `cache/` — Supabase cache (1,208 locality records)

**Why separate?** Raw data is messy; processed is the clean version used by models.

### `/pipeline` — Data Processing (ETL)
```
pipeline/
├── ingestion/           ← Data scrapers & loaders
├── transformation/      ← Feature engineering (166 features from 11 base)
├── supabase_handler.py  ← Fetch data from cloud database
└── cache_handler.py     ← Local cache management
```

**What it does:**
1. Load raw data from Supabase
2. Clean outliers & missing values
3. Engineer 166 features (geometric, temporal, POI, text)
4. Save to `/data/processed/` & cache

See [DATA.md](DATA.md) for detailed pipeline steps.

### `/notebooks` — Analysis & Training
```
notebooks/
├── 01_eda/
│   ├── 01_eda_complete.ipynb    ← Exploratory data analysis
│   └── output/                  ← Generated visualizations
├── 02_model_training/
│   └── 02_model_training.ipynb  ← Train & compare 3 models
└── README.md
```

**01_eda:** Price distributions, correlations, geographic patterns  
**02_model_training:** Train LightGBM, XGBoost, CatBoost; save production model

### `/models` — Trained ML Models
```
models/
├── production/          ← XGBoost (best model) ⭐
│   ├── production_model.pkl
│   └── model_results.csv
├── archive/             ← Previous models
│   ├── lightgbm_model.pkl (18.76% MAPE)
│   └── catboost_model.pkl (19.52% MAPE)
├── data/                ← Training datasets
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

## 🚀 How to Use

### 1. Run Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_eda/01_eda_complete.ipynb
```
Analyzes 12,814 properties: price ranges, feature correlations, geographic clusters.

### 2. Train Models
```bash
jupyter notebook notebooks/02_model_training/02_model_training.ipynb
```
Trains 3 algorithms, compares MAPE, saves best model to `models/production/`.

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
Scraped Data (12,814 properties)
    ↓ [pipeline/ingestion/]
Raw CSV Files
    ↓ [pipeline/transformation/]
Clean Data (10,432 after outlier removal)
    ↓ [Feature Engineering]
166 Features (geometric, temporal, POI, interactions)
    ↓
Supabase Cache (Raw_Features table)
    ↓ [Model Training]
Production Model (XGBoost)
    ↓
Web App / Dashboard / API
```

**Why this structure?**
- Clear separation between data, processing, models, apps
- Each folder has one responsibility
- Easy to trace where errors come from
- Documentation explains the "why", not just "what"

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
# If missing, re-run: notebooks/02_model_training/02_model_training.ipynb
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
