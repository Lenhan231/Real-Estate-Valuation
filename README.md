# 🏠 Real Estate Valuation - DSP391m Capstone Project

**Automated real estate price prediction for Ho Chi Minh City properties**

Predicts property prices using Machine Learning based on area, location, amenities, and geospatial features.

| Metric | Value |
|--------|-------|
| **Model** | 9-Model Ensemble: LightGBM + XGBoost + CatBoost (3-tier price segmentation) |
| **MAPE** | 13.10% (v2.6 production) |
| **R²** | 0.9200 (explains 92% of variance) |
| **MAE** | 2.15B VND |
| **Training Data** | 10,421 properties (80-20 split) |
| **Features** | 79 (78 engineered + 1 target price) |
| **Training Time** | ~149 seconds |

---

## ✨ Latest Features (v2.6)

### **🎯 Price Tier Segmentation**
- ✅ 3-tier ensemble model (low/mid/high price segments)
- ✅ Specialized model per price range for better accuracy
- ✅ 13.10% MAPE - 2.15% improvement over v2.4
- ✅ Feature optimization: 78 features (down from 166, better signal)

### **📝 Feedback Collection System**

### **📝 Feedback Collection System**
- ✅ Persistent feedback form in Streamlit UI
- ✅ Users rate predictions (Accurate/Not Sure/Not Accurate)
- ✅ Optional actual price input for model learning
- ✅ Feedback stored in Supabase with full feature context
- ✅ Session state management to preserve form across reruns

### **📈 Feedback Analytics Dashboard**
- ✅ **Summary Metrics**: Total feedback, MAPE, Model Bias %, Rating Distribution
- ✅ **Trends Over Time**: Daily feedback count & accuracy trends
- ✅ **Segmentation Analysis**: Performance by price bucket, property type, top 10 localities
- ✅ **Distribution Charts**: Rating breakdown, error distribution
- ✅ **Best vs Worst Predictions**: Side-by-side comparison for learning

**Dashboard Functions:**
- `get_feedback_trends()` — Time series data
- `get_feedback_by_segment()` — Grouped performance metrics
- `get_feedback_distribution()` — Rating & error distribution
- `get_best_predictions()` — Most accurate predictions

### **🔄 Active Learning & Model Retraining**
- ✅ **Auto Retraining**: Extract feedback data, rebuild ensemble
- ✅ **Drift Detection**: Monitor performance degradation
- ✅ **Admin Dashboard**: One-click retraining button
- ✅ **Performance Comparison**: Old vs new model metrics
- ✅ **Minimum 3 samples** required for retraining

**Retraining Pipeline:**
```
Feedback Data (Supabase)
    ↓ [extract features + actual prices]
Retraining Dataset
    ↓ [train LightGBM, XGBoost, CatBoost]
New Ensemble Model
    ↓ [compare MAPE/MAE vs old]
Performance Report
    ↓ [deploy if better]
Updated Model
```

**New API Endpoints:**
- `POST /api/admin/retrain` — Trigger retraining
- `GET /api/admin/drift-status` — Check model drift
- `GET /api/admin/model-comparison` — Get metrics

**New Streamlit Tabs:**
1. **💰 Định giá** — Price prediction (2 modes: paste description or detailed form)
2. **📊 Phân tích thị trường** — Market analysis & heatmaps
3. **📈 Feedback Analytics** — Dashboards with trends & segmentation
4. **🔧 Model Management** — Retrain, drift detection, performance comparison

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

## 🚀 Quick Start - Local Development

### Prerequisites

- Python 3.11+
- Git
- Supabase account (for database)

### Setup (5 minutes)

**1. Clone & setup environment:**

```bash
git clone https://github.com/Lenhan231/Real-Estate-Valuation.git
cd Real-Estate-Valuation

# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**2. Configure environment:**

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your Supabase credentials
nano .env
# Required:
# - SUPABASE_URL
# - SUPABASE_SERVICE_KEY
# - WANDB_API_KEY (optional, for experiment tracking)
```

**3. Start both services:**

```bash
# Option A: One command (if run.sh exists)
bash run.sh
# or Windows:
.\run.ps1

# Option B: Manual (2 terminals)
# Terminal 1 - API:
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Streamlit UI:
streamlit run app/ui/streamlit_app.py --server.port 8501
```

**4. Access:**

- 📡 **API Swagger Docs**: http://localhost:8000/docs
- 🎨 **Streamlit UI**: http://localhost:8501
- 📊 **API Root**: http://localhost:8000

### Testing

```bash
# Test API health
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"street":"Nguyễn Hữu Cảnh","locality":"Bình Thạnh","area_m2":100}'

# List available localities
curl http://localhost:8000/api/localities
```

### Troubleshooting

**"Module not found" error:**
```bash
pip install -r requirements.txt
```

**"Supabase connection failed":**
- Check `.env` file has correct credentials
- Verify Supabase tables exist: `address_cache`, `Raw_Features`
- Test connection: `python -c "from app.core.models import get_models; get_models()"`

**Port already in use:**
```bash
# Kill process on port 8000
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

---

## 🌐 Live Deployment (Render)

**Production URLs:**

- 🔗 **API**: https://real-estate-valuation-88yg.onrender.com
  - API Docs: https://real-estate-valuation-88yg.onrender.com/docs
  - Status: https://real-estate-valuation-88yg.onrender.com/health

- 🎨 **Streamlit Web App**: https://real-estate-valuation-2.onrender.com

**For deployment to other platforms (DigitalOcean, AWS, self-hosted), see [DEPLOYMENT.md](DEPLOYMENT.md)**

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
