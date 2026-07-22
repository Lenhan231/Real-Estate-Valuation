# 📦 Model Versioning

**Tracking model versions, hyperparameters, và performance**

## 🎯 Tại sao cần versioning?

- Track which model version is in production
- Compare performance across versions
- Rollback to previous version if needed
- Document what changed & why

## 📋 Model Version Template

```
Model: XGBoost v1.0
Date: 2026-07-20
Training Data: 8,345 properties (80%)
Test Data: 2,087 properties (20%)

Hyperparameters:
  - max_depth: 7
  - learning_rate: 0.1
  - n_estimators: 100
  - subsample: 0.8
  - colsample_bytree: 0.8

Performance:
  - MAPE: 18.01%
  - R²: 0.8663
  - MAE: 2.67B VND
  - RMSE: 4.37B VND

Features: 166 engineered features
  - Location features (district, ward, proximity to POI)
  - Property features (area, bedrooms, floors, age)
  - Market features (market segment, price per sqm)

Changes from v0.9:
  - Added 10 new geospatial features
  - Tuned max_depth from 8 to 7
  - Improved MAPE from 18.5% to 18.01%

Status: PRODUCTION
File: models/production/production_model.pkl
Deployed: 2026-07-18
```

## 📊 Version History File

Create `models/VERSION_HISTORY.csv`:

```csv
version,date,mape,r2,status,notes
v1.0,2026-07-20,18.01,0.8663,production,XGBoost baseline
v0.9,2026-07-15,18.50,0.8600,archived,LightGBM experiment
v0.8,2026-07-10,19.52,0.8529,archived,CatBoost experiment
```

## 🚀 How to Version a Model

```python
import json
from datetime import datetime

model_metadata = {
    "version": "v1.1",
    "date_trained": datetime.now().isoformat(),
    "model_type": "XGBoost",
    "hyperparameters": {
        "max_depth": 7,
        "learning_rate": 0.1,
        "n_estimators": 100
    },
    "performance": {
        "mape": 0.1801,
        "r2": 0.8663,
        "mae": 2.67e9
    },
    "training_data": {
        "n_samples": 8345,
        "n_features": 166,
        "feature_version": "v2.0"
    },
    "notes": "Added geospatial features"
}

# Save metadata with model
with open("models/production/metadata_v1.1.json", "w") as f:
    json.dump(model_metadata, f, indent=2)
```

## 📝 Best Practices

1. **Never overwrite** - Always save with new version number
2. **Document changes** - What's different from previous version
3. **Keep history** - Archive old versions for 6+ months
4. **Tag in git** - Use git tags: `git tag model-v1.0`
5. **Store metadata** - JSON file with hyperparameters & performance
