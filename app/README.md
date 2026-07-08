# 🎨 Web & API - User Interfaces

## Usage

### 1. Interactive Web UI
```bash
streamlit run app.py
```
Opens: http://localhost:8501  
Features: Property valuation, instant predictions with confidence bounds

### 2. BI Dashboard
```bash
streamlit run dashboard.py
```
Opens: http://localhost:8501  
Features: Heatmaps, trends, locality rankings, segment analysis

### 3. REST API
```bash
python api_simple.py
```
API: http://localhost:5000

**Endpoints:**
- `GET /health` - Status check
- `GET /api/info` - Model metrics
- `POST /api/predict` - Predict price

**Example:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"area_m2": 80, "num_floors": 3, "locality": "phường bình thạnh"}'
```

## Files

```
├── app.py              # Interactive web UI
├── dashboard.py        # BI dashboard
├── api_simple.py       # REST API
├── geo.py              # Geospatial utilities
├── inference.py        # Original inference (legacy)
└── README.md           # This file
```

## Model Integration

All interfaces load models from:
```
models/saved_models/
  ├── hybrid_meta.pkl
  ├── ensemble_meta_learner.joblib
  ├── ensemble_base_models.joblib
  └── segment_models.joblib
```

Predictions are generated from:
```
models/data/predictions_latest.csv
```

## Architecture

```
User Interface
    ↓
Load Pre-trained Models
    ↓
Feature Engineering
    ↓
Hybrid Predictor
    (segment models + ensemble fallback)
    ↓
Return Prediction ± Confidence
```
