# 🏠 Real Estate Valuation - Backend + Frontend

Refactored architecture with FastAPI backend and Streamlit frontend.

## 🚀 Quick Start

**Terminal 1 - API Backend:**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
API: http://localhost:8000/docs (Swagger)

**Terminal 2 - Streamlit UI:**
```bash
streamlit run app/ui/streamlit_app.py --server.port 8500
```
UI: http://localhost:8500

## 🏗️ Architecture

```
app/
├── main.py                 # FastAPI application
├── schemas/                # Pydantic request/response models
│   ├── predict.py         # PredictRequest, PredictResponse
│   └── feedback.py        # FeedbackRequest, FeedbackResponse
├── routers/               # API endpoints (organized by feature)
│   ├── predict.py         # POST /api/predict, /api/parse, /api/localities
│   └── feedback.py        # POST /api/feedback, GET /api/feedback/stats
├── services/              # Business logic layer
│   ├── inference.py       # Price prediction logic
│   └── feedback.py        # Feedback submission & analytics
├── core/                  # Configuration & ML utilities
│   ├── config.py          # Settings & environment
│   ├── models.py          # Singleton model loader
│   ├── inference.py       # build_row(), predict_price() (imported)
│   ├── geo.py             # GeoLookup class
│   ├── feedback.py        # Supabase feedback functions
│   ├── parsers.py         # Text parsing
│   └── explainability.py  # XAI calculations
├── ui/                    # Frontend application
│   └── streamlit_app.py   # Streamlit UI (calls API)
├── utils/                 # Utility functions
└── __init__.py
```

## 🔄 Data Flow

```
User Input (Streamlit UI)
    ↓
API Call (requests library)
    ↓
Router (predict.py or feedback.py)
    ↓
Service (inference.py or feedback.py)
    ↓
Core Logic (models.py loads ML models, geo.py geocodes, etc.)
    ↓
Response (Pydantic schema)
    ↓
Display in Streamlit
```

## 📡 API Endpoints

### Prediction
- `POST /api/predict` - Price prediction with XAI
- `POST /api/parse` - Parse listing description
- `GET /api/localities` - Get available districts/wards

### Feedback
- `POST /api/feedback` - Submit user feedback
- `GET /api/feedback/stats` - Get analytics dashboard data

### Info
- `GET /` - API info

## 🎯 Key Components

### Schemas (`schemas/`)
Pydantic models for type safety and validation:
- `PredictRequest` - Property details for prediction
- `PredictResponse` - Price + XAI data
- `FeedbackRequest` - User feedback data
- `FeedbackStats` - Analytics metrics

### Services (`services/`)
Pure business logic (no framework dependencies):
- `inference.predict_property_price()` - Core prediction logic
- `feedback.submit_feedback()` - Save feedback to Supabase
- `feedback.get_feedback_analytics()` - Get stats for dashboard

### Routers (`routers/`)
FastAPI endpoint handlers:
- Map HTTP requests to services
- Validate input with schemas
- Return responses with Pydantic models

### Core (`core/`)
Low-level utilities and configuration:
- `models.py` - Singleton pattern for ML model loading (load once, use everywhere)
- `config.py` - Settings from environment variables
- Imported modules from original `app/core/` (inference, geo, parsers, etc.)

## 🔧 Development Workflow

### Adding a New Endpoint

1. **Create Schema** (`schemas/new_feature.py`):
```python
from pydantic import BaseModel
class NewRequest(BaseModel):
    param1: str
    param2: float
```

2. **Create Service** (`services/new_feature.py`):
```python
def handle_new_feature(param1: str, param2: float) -> dict:
    # Business logic here
    return result
```

3. **Create Router** (`routers/new_feature.py`):
```python
@router.post("/new-endpoint")
async def endpoint(request: NewRequest):
    result = handle_new_feature(...)
    return result
```

4. **Include in Main** (`main.py`):
```python
from app.routers import new_feature
app.include_router(new_feature.router)
```

### Debugging

**API logs:**
```bash
python -m uvicorn app.main:app --log-level debug
```

**Streamlit logs:**
```bash
streamlit run app/ui/streamlit_app.py --logger.level=debug
```

## 📊 Model Info

- **Ensemble**: LightGBM + XGBoost + CatBoost (3-tier by price)
- **Features**: 80 (78 engineered + 2 locality encoding)
- **Performance**: 13.1% MAPE, 0.92 R²

See `models/README.md` for details.

## 🔐 Environment Variables

```bash
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key
```

## 📦 Dependencies

Backend:
```
fastapi>=0.100
uvicorn>=0.23
pydantic>=2.0
```

Frontend:
```
streamlit>=1.28
requests>=2.31
pandas>=2.0
```

ML:
```
lightgbm>=4.0
xgboost>=2.0
catboost>=1.2
scikit-learn>=1.3
```

## ✅ Benefits of This Structure

- ✅ **Separation of Concerns**: Backend logic independent of UI
- ✅ **Testability**: Services can be tested without Streamlit
- ✅ **Reusability**: API can serve multiple frontends (web, mobile, etc.)
- ✅ **Scalability**: Easy to add features (just add schema → service → router)
- ✅ **Type Safety**: Pydantic validates all inputs/outputs
- ✅ **Documentation**: Swagger docs auto-generated from schemas

## 🚀 Next Steps

1. **Model Retraining Script** - Use accumulated feedback data
2. **Parser Improvements** - Better Vietnamese NLP
3. **Active Learning** - Highlight low-confidence predictions
4. **Web Frontend** - React/Vue frontend calling same API

---

**Status**: ✅ Production-Ready  
**Last Updated**: 2026-07-22
