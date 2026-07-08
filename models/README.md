# 🤖 Models - Hybrid Ensemble

## Performance

```
MAPE:  3.36%  (target: <10%) ✅
MAE:   0.66B VND
R²:    0.9939
```

## Usage

### 1. Train
```bash
cd models
python train.py
```
Creates: `saved_models/hybrid_*.pkl` + `saved_models/ensemble_*.joblib`

### 2. Predict
```bash
python predict.py
```
Creates: `data/predictions_latest.csv`

## Architecture

**Hybrid = Segment Models + Ensemble Stacking**

```
Input (65 features)
    ↓
├─ Segment 0-5B (XGBoost)      MAPE: 6.6%
├─ Segment 5-20B (XGBoost)     MAPE: 2.8% ← BEST
├─ Segment 20-100B (XGBoost)   MAPE: 3.2%
│
├─ Ensemble Base:
│  ├─ XGBoost
│  └─ RandomForest
│  └─ Ridge Meta-Learner
│
→ Hybrid Prediction
  (segment if available, else ensemble)
```

## Features (65)

- **Numeric (25):** Amenity distances, property dimensions
- **Binary (5):** Kitchen, parking, terrace, etc.
- **Categorical (15):** Type, legal status, direction
- **Engineered (3):** Price/m², amenity score, location
- **Keywords (10):** Modern, luxury, investment flags
- **Missing (5):** Imputation indicators
- **Locality (1):** Median price per ward

## Files

```
├── train.py                    # Training script
├── predict.py                  # Inference script
├── saved_models/
│   ├── hybrid_meta.pkl         # Metadata + metrics
│   ├── ensemble_meta_learner.joblib
│   ├── ensemble_base_models.joblib
│   └── segment_models.joblib
├── data/
│   ├── raw_data.csv            # Original (8,643)
│   ├── model_ready_data.csv    # Cleaned (7,692)
│   └── predictions_latest.csv  # Full predictions
└── README.md                   # This file
```

## Train Details

- **Data:** 8,643 properties
- **After cleaning:** 7,692 (dropped 944 outliers)
- **Split:** 6,153 train / 1,539 test (80/20)
- **Training time:** ~2 minutes
- **Prediction time:** ~30ms per property

## Retrain Monthly

```bash
cd models
python train.py    # Update models with new data
python predict.py  # Generate fresh predictions
```
