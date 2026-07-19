# 🤖 Models - Real Estate Valuation

## Directory Structure

```
models/
├── scripts/                        # Training and utility scripts
│   ├── shared/                    # Shared utilities (imported by all scripts)
│   │   ├── preprocessing.py       # Data preprocessing and feature engineering
│   │   ├── plotting.py            # Visualization functions
│   │   ├── wandb_logging.py       # Weights & Biases integration
│   │   └── __init__.py            # Module initialization
│   ├── train_xgboost.py           # XGBoost model training
│   └── train_ensemble_3model.py   # ⭐ Recommended: 3-model ensemble (LGBM+CatBoost+XGB)
│
├── saved_models/                   # Trained model artifacts
│   ├── lgbm_*.pkl                 # LightGBM models (6 buckets)
│   ├── cb_*.pkl                   # CatBoost models (6 buckets)
│   ├── xgb_*.pkl                  # XGBoost models (6 buckets)
│   ├── xgboost_*.pkl              # Single XGBoost model (alternative)
│   └── plots/                     # Training visualizations & plots
│
└── README.md                       # This file
```

## Models

### ⭐ 3-Model Ensemble (LGBM + CatBoost + XGBoost) - RECOMMENDED

```bash
python models/scripts/train_ensemble_3model.py --data-source supabase
```

**Specifications:**

- **Type:** 3-Model Ensemble (LightGBM + CatBoost + XGBoost)
- **Strategy:** 6-bucket stratification by price & property type
- **Dataset:** 12,814 Supabase records
- **Training data:** 8,345 properties (80% split)
- **Test data:** 2,087 properties (20% split)
- **Features:** 39+ engineered features
- **Target:** Price in VND (log-transformed for training)
- **Training Time:** ~100 seconds
- **Improvements:** Early stopping + weighted averaging by validation performance

**Performance Metrics:**

| Metric | Value |
| --- | --- |
| Global RMSE | 3.35 Billion VND |
| Global MAE | 2.12 Billion VND |
| Global R² | 0.9214 |
| Global MAPE | 13.53% |
| RMSE(log) | 0.1752 |

**Performance by Price Segment:**

| Segment | Price Range | MAPE | Test Samples |
| --- | --- | --- | --- |
| Budget | 0-5B VND | 11.92% | 225 |
| Mid-range | 5-20B VND | 14.09% | 1,270 |
| Premium | 20B+ VND | 13.21% | 592 |

### XGBoost Model

```python
import joblib
model = joblib.load('models/saved_models/xgboost_cleaned_supabase.pkl')
predictions = model.predict(X_test)
```

**Specifications:**

- **Type:** XGBoost Regressor
- **Dataset:** 12,814 Supabase records
- **Training data:** 8,345 properties (80% split)
- **Test data:** 2,087 properties (20% split)
- **Features:** 39+ engineered features
- **Target:** Price in VND (log-transformed for training)

**Performance Metrics:**

| Metric | Value |
| --- | --- |
| Global RMSE | 4.12 Billion VND |
| Global MAE | 2.51 Billion VND |
| Global R² | 0.8808 |
| Global MAPE | 17.00% |
| RMSE(log) | 0.2294 |
| Training Time | ~24 seconds |

**Performance by Price Segment:**

| Segment | Price Range | MAPE | Test Samples |
| --- | --- | --- | --- |
| Budget | 0-5B VND | 26.16% | 225 |
| Mid-range | 5-20B VND | 16.28% | 1,270 |
| Premium | 20B+ VND | 15.05% | 592 |

## Training Data

**Where:** `data/processed/model_training_data.csv`

**What:** All 8,345 training samples with 39+ preprocessed features + target price

**Why saved:**

- ✅ **Reproducibility** — exact data used to train the models
- ✅ **Inspection** — validate feature engineering and preprocessing
- ✅ **Analysis** — analyze model inputs and feature distributions
- ✅ **Debugging** — trace any data issues back to source
- ✅ **Comparison** — compare against test predictions

**Note:** This is the 80% training split only. Test data (20%) is not saved since ground truth should stay separate from model validation.

## Training Scripts

### XGBoost Training

```bash
python models/scripts/train_xgboost.py --data-source supabase
```

Trains an XGBoost regressor on preprocessed real estate features.

### Ensemble Training

```bash
python models/scripts/train_ensemble.py --data-source supabase
```

Trains an ensemble combining LightGBM and CatBoost with 6-bucket stratification.

## Feature Engineering Pipeline

### Base Features

- `num_floors`: Number of stories
- `num_bedrooms`: Bedroom count
- `area_m2`: Total area
- `width_m`, `length_m`: Lot dimensions
- Geographic: distance_to_center_km, nearest amenities
- `locality`: Area/district information

### Engineered Features (39+)

- **Geometric:** perimeter, shape_ratio, area × floors, area × bedrooms
- **Temporal:** post_day_year, post_day_month, post_day_day
- **Location:** location_score, distance_vs_area
- **Amenity:** amenity_score, nearby_amenities_count
- **Text:** is_hem_xe_hoi, is_mat_tien, is_gap (from description)
- **Log transforms:** log_area, log_distance
- **Locality:** locality_price_median, price_per_sqm_market
- **Categorical:** One-hot encoded for categorical variables

## Training Details

### Data Preparation

1. Source: Supabase `Raw_Features` table (12,814 records)
2. Preprocessing:
   - Remove duplicates
   - Filter outliers (2B-50B VND price range, 15-500 m² area)
   - Drop rows with missing target
3. Feature Engineering: 39+ engineered features
4. Encoding: One-hot for categorical, log-transform for skewed features
5. Output: `data/processed/model_training_data.csv`

### Model Training

**XGBoost Hyperparameters:**

```yaml
n_estimators: 1000
max_depth: 8
learning_rate: 0.03
subsample: 0.8
colsample_bytree: 0.8
Target transform: log1p (log(1 + price))
```

**Train/Test Split:**

- Train: 8,345 properties (80%)
- Test: 2,087 properties (20%)
- Random state: 42 (reproducible)

## Data Output

### Saved Artifacts

- **Models:** `models/saved_models/xgboost_*.pkl`, `ensemble_*.pkl`
- **Plots:** `models/saved_models/plots/`
  - Feature importance charts
  - Predicted vs actual scatter plots
- **Data:** `data/processed/model_training_data.csv`

## Notes

- All models use Supabase as primary data source
- Local CSV fallback available if Supabase connection fails
- Predictions are in Vietnamese Dong (VND)
- Feature order matters when using saved models
- Models are stateless and can be parallelized

## Troubleshooting

**Model won't load:**

```python
import joblib
model = joblib.load('models/saved_models/xgboost_cleaned_supabase.pkl')
# Check: file exists, is readable, correct path
```

**Poor predictions:**

- Verify all features are present
- Check feature engineering matches training pipeline
- Consider price range segmentation for outliers
- Validate against `data/processed/model_training_data.csv`
