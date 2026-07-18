# 🤖 Models - Trained ML Models

## Structure

```
models/
├── production/                     # Production-ready models
│   ├── production_model.pkl       # ⭐ XGBoost (MAPE: 18.01%)
│   ├── xgboost_model.pkl          # Same as production_model
│   └── model_results.csv          # Evaluation metrics
│
├── archive/                        # Previous/experimental models
│   ├── lightgbm_model.pkl         # LightGBM (MAPE: 18.76%)
│   ├── catboost_model.pkl         # CatBoost (MAPE: 19.52%)
│   └── xgboost_tuned_model.pkl    # Tuning experiment (not better)
│
├── segment_models/                # (Optional) Segmented by price/type
│   ├── low/                       # Budget segment (< 5B VND)
│   ├── mid/                       # Mid-range (5-20B VND)
│   └── high/                      # Premium (> 20B VND)
│
├── data/                          # Training data & cache
│   ├── model_ready_data.csv       # Preprocessed training data
│   └── predictions.csv            # Saved predictions
│
├── save_models/                   # (Legacy) Raw training outputs
│   └── [original saved files]
│
└── README.md                       # This file
```

## Model Selection

### ⭐ Production Model (Recommended)
```python
import joblib
model = joblib.load('models/production/production_model.pkl')
predictions = model.predict(X_test)
```

**Specifications:**
- **Type:** XGBoost Regressor
- **Training data:** 8,345 properties (80% split)
- **Features:** 166 engineered features
- **Target:** Price (log-transformed)

**Performance:**
- MAPE: 18.01% (Mean Absolute Percentage Error)
- R²: 0.8663 (Explains 86.63% variance)
- RMSE: 4.37B VND
- MAE: 2.67B VND

**Confidence Interval:**
- 95% CI: ±5.24B VND (±1.96 × MAE)

### Alternative Models (Archive)
- **LightGBM:** MAPE 18.76%, R² 0.8642
- **CatBoost:** MAPE 19.52%, R² 0.8529

## Model Usage

### In Python
```python
from app.inference_simple import load_model, predict_price

model = load_model()
predictions = predict_price(features_df)  # Returns prices in VND
```

### In Streamlit App
```bash
streamlit run app/app_simple.py
```

### Via REST API
```bash
python app/api_simple.py
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"area_m2": 80, "num_floors": 3, ...}'
```

## Feature Engineering Pipeline

### Input Features (11 base)
- `num_floors`: Number of stories
- `num_bedrooms`: Bedroom count
- `area_m2`: Total area
- `width_m`, `length_m`: Lot dimensions
- `road_width_m`: Road frontage width
- `direction`: Cardinal direction
- `property_type`: Type (mặt tiền / hẻm)
- `legal_status`: Documentation status
- Geographic: lat, lon, distance_to_center

### Engineered Features (155 additional)
- **Geometric:** perimeter, shape_ratio, area × floors, area × bedrooms
- **Temporal:** post_day_year, post_day_month, post_day_day
- **Location:** location_score (distance-weighted amenity access)
- **Amenity:** amenity_score, school_count, hospital_count, etc.
- **Text:** is_hem_xe_hoi, is_mat_tien, is_gap (from description)
- **Log transforms:** log_area, log_distance
- **Interactions:** location × amenity interactions
- **Categorical:** One-hot encoded direction, type, status

## Training Details

### Data Preparation
1. Filter outliers: Price (2-50B VND), Area (15-500 m²)
2. Calculate price per sqm: 30-800M VND/m²
3. Remove duplicates: 260 rows
4. Clean invalid values: 1 row with NULL street
5. **Result:** 10,432 clean properties

### Model Training
```
XGBoost Hyperparameters:
- n_estimators: 500
- max_depth: 8
- learning_rate: 0.03
- subsample: 0.8
- colsample_bytree: 0.8
- Target transform: log1p (log(1 + price))
```

### Evaluation Metrics
```
Test Set (2,087 properties):
- MAPE: 18.01%
- R²: 0.8663
- RMSE: 4.37B VND
- MAE: 2.67B VND
```

## Model Updates

To retrain with new data:
```bash
python run_training.py
```

New model will be saved to `models/save_models/` then moved to appropriate folder.

## Notes

- All models trained on Supabase `Raw_Features` table
- Predictions are clipped to [0, ∞) to avoid negative prices
- Model expects 166 features in specific order (see feature_pipeline.py)
- Production model is stateless and can be parallelized

## Troubleshooting

**Model won't load:**
```python
import joblib
model = joblib.load('models/production/production_model.pkl')
# If fails: check file exists and is < 10MB
```

**Poor predictions:**
- Verify feature engineering matches training pipeline
- Check for out-of-distribution inputs
- Consider segmented models for extreme price ranges
