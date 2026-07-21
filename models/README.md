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

### 📊 Model Performance Comparison

| Model | MAPE | R² | MAE (B VND) | RMSE (B VND) | Training Time | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| **3-Model Ensemble** 🏆 | **13.47%** | **0.9142** | **2.22** | **3.54** | 81.6s | Best overall, 6-bucket strategy |
| Single XGBoost | 18.16% | 0.8645 | 2.72 | 4.44 | 12.7s | Baseline single model |
| Baseline (Raw Features) | 20.96% | 0.8528 | 2.93 | 4.63 | ~10s | No feature engineering |

**Winner:** 3-Model Ensemble is **25% better** than single model, **33% better** than baseline! 🎯

---

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
- **Advantage over single-model:** +25% better MAPE (13.47% vs 18.01% XGBoost alone)

**Performance Metrics (Optimized v2.1):**

| Metric | Value | Status |
| --- | --- | --- |
| Global RMSE | 3.48 Billion VND | ⬇️ Improved |
| Global MAE | 2.18 Billion VND | ⬇️ Improved |
| Global R² | 0.9168 | ⬆️ Improved |
| Global MAPE | **13.28%** | ⬇️ Improved |
| RMSE(log) | 0.1755 | ⬇️ Improved |

**Performance by Price Segment (Global Aggregation):**

| Segment | Price Range | MAPE | Test Samples |
| --- | --- | --- | --- |
| Budget | 0-5B VND | **10.83%** ✅ | 232 |
| Mid-range | 5-20B VND | 13.82% | 1,250 |
| Premium | 20B+ VND | 13.76% | 603 |

**⭐ Key Achievement:**

- Optimized model MAPE: **13.28%** (down from initial 13.47%)
- R² improved to **0.9168** (up from 0.9142)
- Budget segment achieves **10.83%** (within 0.83% of <10% target)
- Training time: 106s (includes all 3 tiers × 3 models)

**Detailed Breakdown: Individual Bucket Performance (Price × Property Type)**

The ensemble uses 6 specialized buckets. Here's the MAPE for each:

| Price Tier | Property Type | Train | Test | MAPE | R² | MAE (B) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Low (0-5B) | Nha Mat Tien | 68 | 16 | **7.43%** | -0.13 | 0.32 | 🏆 BEST |
| Low (0-5B) | Nha Trong Hem | 856 | 216 | 11.09% | 0.39 | 0.43 | ✅ Good |
| **Mid (5-20B)** | **Nha Mat Tien** | **1,457** | **359** | **17.53%** | 0.55 | 2.03 | ⚠️ WORST |
| Mid (5-20B) | Nha Trong Hem | 3,612 | 891 | 12.32% | 0.78 | 1.22 | ✅ Good |
| High (20B+) | Nha Mat Tien | 1,878 | 462 | 13.51% | 0.49 | 4.55 | ✅ Good |
| High (20B+) | Nha Trong Hem | 465 | 141 | 14.56% | 0.45 | 4.30 | ✅ Good |

#### Insights from Bucket Breakdown

- 🏆 **Best:** Low + Nha Mat Tien (7.43% MAPE) — excellent but small segment (16 test samples)
- ⚠️ **Worst:** Mid + Nha Mat Tien (17.53% MAPE) — significant size (359 test samples), needs improvement
- 📊 **Pattern:** Nha Trong Hem consistently outperforms Nha Mat Tien across all price tiers
- 💡 **Key Issue:** Mid-price "Nha Mat Tien" properties are hardest to predict (17.53% vs 7-14% elsewhere)

### Model Optimization History

**Version 1.0 (Initial):** 13.47% MAPE (6-bucket price×property type)

**Version 2.0 (Simplified):** 13.38% MAPE (3-bucket price only)

- Removed property type segmentation (marginal benefit)
- 43% faster training (54.8s vs 95.2s)
- Better R² (0.9164 vs 0.9142)

**Version 2.1 (Optimized):** 13.28% MAPE ✅ **CURRENT**

- Phase 1: Hyperparameter tuning applied
  - LGBM: lr 0.03→0.05, n_est 1000
  - XGBoost: n_est 1000→1500
  - CatBoost: lr 0.03→0.05, depth 6→8, iter 1000→1500
- Improvement: 0.10% MAPE reduction
- Trade-off: Training time increased to 106s

**Testing Results:**

- Phase 2 (Stacking): 0.11% improvement (not worth complexity)
- Phase 3 (Data Quality): No critical issues found

---

### Segmentation Strategy Analysis

We tested three different segmentation approaches to understand which strategy provides the best performance-to-speed tradeoff:

| Strategy | Buckets | Models | MAPE | R² | Time | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **No Segmentation (Global)** | 1 | 3 | 18.35% | 0.8661 | 21.2s | ❌ Worst |
| **Price Only (3 buckets)** | 3 | 9 | 13.38% | 0.9164 | 54.8s | ⚡ Nearly ties |
| **Price + Type (6 buckets)** 🏆 | 6 | 18 | 13.47% | 0.9142 | 95.2s | ✅ Current |

**Key Finding:** Price-Only strategy is **nearly identical to current** (0.09% MAPE difference) but **43% faster** and simpler to maintain. Could be considered as an alternative for production if speed is critical.

---

### XGBoost Model (Single Model Alternative)

```python
import joblib
model = joblib.load('models/saved_models/xgboost_cleaned_supabase.pkl')
predictions = model.predict(X_test)
```

**Specifications:**

- **Type:** XGBoost Regressor (single model)
- **Dataset:** 12,814 Supabase records
- **Training data:** 8,336 properties (80% split)
- **Test data:** 2,085 properties (20% split)
- **Features:** 93 engineered features
- **Target:** Price in VND (log-transformed for training)

**Performance Metrics:**

| Metric | Value |
| --- | --- |
| Global RMSE | 4.44 Billion VND |
| Global MAE | 2.72 Billion VND |
| Global R² | 0.8645 |
| Global MAPE | 18.16% |
| RMSE(log) | 0.2446 |
| Training Time | 12.7 seconds |

**Performance by Price Segment:**

| Segment | Price Range | MAPE | Test Samples |
| --- | --- | --- | --- |
| Budget | 0-5B VND | 29.86% | 232 |
| Mid-range | 5-20B VND | 17.16% | 1,250 |
| Premium | 20B+ VND | 15.73% | 603 |

**Note:** Single model performs worse than ensemble on budget segment (29.86% vs 10.83%)

### Baseline Model (Raw Features Only)

**Purpose:** Establish a minimum performance baseline without feature engineering.

**Specifications:**

- **Type:** XGBoost Regressor (no feature engineering)
- **Dataset:** 12,814 Supabase records
- **Training data:** 8,336 properties (80% split)
- **Test data:** 2,085 properties (20% split)
- **Features:** 93 raw features only (no feature engineering)
- **Target:** Price in VND

**Performance Metrics:**

| Metric | Value |
| --- | --- |
| Global RMSE | 4.63 Billion VND |
| Global MAE | 2.93 Billion VND |
| Global R² | 0.8528 |
| Global MAPE | 20.96% |

**Key Insight:** Shows that feature engineering is critical — ensemble achieves **35.7% improvement** over baseline (13.47% vs 20.96%).

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

### 🚀 Production Model (CURRENT)

```bash
python models/scripts/train_production.py --data-source supabase
```

**PRODUCTION-READY** — Price-Only Ensemble (3 price tiers × 3-model ensemble):

- **Architecture:** LightGBM + CatBoost + XGBoost per price tier
- **Strategy:** Simple price segmentation (Low/Mid/High) — no property type split
- **Performance:** **13.38% MAPE**, 0.9164 R²
- **Speed:** 54.8s training (43% faster than previous)
- **Complexity:** Simple, maintainable, production-ready
- **Data Balance:** No tiny segments (improves all models)
- **Why Chosen:** Best R², faster training, balanced data quality

### Archived Experimental Scripts

See [DEPRECATED_SCRIPTS.md](models/scripts/DEPRECATED_SCRIPTS.md) for previous approaches tested:

- Baseline (raw features): 20.96% MAPE
- Single XGBoost: 18.16% MAPE
- 6-bucket ensemble: 13.47% MAPE (more complex, similar performance)
- Property-type only: 18.22% MAPE
- Deep Learning: Not tested (requires TensorFlow/TabNet)

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
