# House Price Prediction Models

## Overview

This directory contains machine learning models for predicting Vietnamese house prices based on geospatial features extracted from OpenStreetMap and property characteristics.

## 📁 Structure

```
models/
├── training/          ← Training pipeline
│   ├── train.py      ← Train models (Random Forest, Gradient Boosting)
│   └── __init__.py
├── inference/        ← Prediction pipeline
│   ├── predict.py    ← Load model and make predictions
│   └── __init__.py
├── saved_models/     ← Trained model artifacts
│   ├── random_forest.joblib
│   ├── random_forest_meta.pkl
│   └── ...
└── README.md
```

## 🚀 Usage

### Step 1: Train Models

Open and run: **`models/training/train_model.ipynb`**

**What it does:**
- Fetches data from Supabase database
- Prepares features (18 geospatial + property features)
- Trains 2 models:
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Evaluates on test set (20% of data)
- Visualizes feature importance
- Saves best model + metadata

**Output:**
- `models/saved_models/random_forest.joblib`
- `models/saved_models/random_forest_meta.pkl`

### Step 2: Make Predictions

Open and run: **`models/inference/predict.ipynb`**

**What it does:**
- Loads best trained model
- Fetches all data from Supabase
- Makes price predictions
- Calculates prediction errors (if actual prices available)
- Visualizes predictions vs actual prices
- Saves predictions to `data/predictions/predictions_latest.csv`

**Output:**
- `data/predictions/predictions_latest.csv` (includes actual vs predicted prices)

### Alternative: Command Line Scripts

If you prefer CLI, use the Python scripts:
- `models/training/train.py` - Non-interactive training
- `models/inference/predict.py` - Non-interactive prediction

```bash
python models/training/train.py
python models/inference/predict.py
```

## 📊 Features Used

**Geospatial Features (from Overpass API):**
- `nearest_school_km`, `school_count_3km`
- `nearest_hospital_km`, `hospital_count_5km`
- `nearest_marketplace_km`, `marketplace_count_3km`
- `nearest_supermarket_km`, `supermarket_count_3km`
- `nearest_mall_km`, `mall_count_3km`
- `nearest_bus_stop_km`, `bus_stop_count_1km`
- `nearest_metro_km`, `metro_count_5km`

**Property Features:**
- `area_m2` - Property area in square meters
- `distance_to_center_km` - Distance to city center
- `locality_square` - Locality area
- `locality_population_density` - Population density

**Target:**
- `price_billion_vnd` - House price in billion Vietnamese Dong

## 📈 Model Performance

Models are evaluated on:
- **R² Score** - Variance explained (0-1, higher is better)
- **RMSE** - Root Mean Squared Error in billion VND
- **MAE** - Mean Absolute Error in billion VND

Current best models typically achieve:
- R² Score: 0.75-0.85
- RMSE: 1.5-2.0 billion VND (depending on data quality)

## 🔄 Workflow

```
Pipeline (ETL)
    ↓
Supabase Database
    ↓
Train Models
    ↓
Saved Model Artifacts
    ↓
Make Predictions
    ↓
data/predictions/
```

## 🛠️ Customization

### Add New Features

Edit `train.py`:
```python
FEATURE_COLS = [
    # Add new column names here
    'your_new_feature',
]
```

### Change Model Hyperparameters

Edit `train.py`:
```python
rf_model = RandomForestRegressor(
    n_estimators=200,      # Increase trees
    max_depth=25,          # Deeper trees
    # ... other params
)
```

### Change Train/Test Split

Edit `train.py`:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,  # Use 30% for testing
    random_state=42
)
```

## ⚠️ Common Issues

### ModuleNotFoundError
**Problem:** Cannot import supabase_handler
**Solution:** Make sure you're running from project root:
```bash
cd /path/to/HousePricePrediction
python models/training/train.py
```

### No trained models found
**Problem:** Can't make predictions, no models saved
**Solution:** Run training first:
```bash
python models/training/train.py
```

### Supabase connection error
**Problem:** Cannot fetch data from Supabase
**Solution:** Check `.env` file has valid credentials:
```
SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...
```

## 📝 Output Files

### After Training:
- `models/saved_models/random_forest.joblib` - Model binary
- `models/saved_models/random_forest_meta.pkl` - Features & metrics

### After Prediction:
- `data/predictions/predictions_latest.csv` - Full predictions with errors

**CSV Columns:**
```
area_m2, distance_to_center_km, ... (all input features)
predicted_price_billion_vnd       (model prediction)
actual_price_billion_vnd          (true price if available)
error_billion_vnd                 (difference)
error_pct                         (error as percentage)
```

## 🔮 Future Improvements

- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Feature engineering (polynomial, interactions)
- [ ] Ensemble methods (stacking, voting)
- [ ] Cross-validation for better estimates
- [ ] Feature importance analysis
- [ ] Model versioning & experiment tracking
- [ ] Batch prediction API endpoint

## 📚 References

- [scikit-learn RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [scikit-learn GradientBoosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [Model Persistence with joblib](https://joblib.readthedocs.io/)
