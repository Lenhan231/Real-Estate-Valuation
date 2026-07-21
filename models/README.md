# 🏠 Real Estate Valuation Models - v2.6 Production

## Overview

**v2.6** is the production-ready ensemble model for automated real estate price prediction in Vietnam. Achieves **13.10% MAPE** (Mean Absolute Percentage Error) with 78 optimized features and intelligent 3-tier price segmentation.

```
🎯 Performance
  MAPE:  13.10%        Mean Absolute Percentage Error
  R²:    0.9200        Coefficient of Determination  
  MAE:   2.15B VND     Mean Absolute Error in billions
  RMSE:  3.41B VND     Root Mean Squared Error

⚙️ Architecture
  Models:   9 (LightGBM + XGBoost + CatBoost × 3 price tiers)
  Features: 78 (64 base + 14 polynomial/interaction)
  Data:     10,421 training samples from Supabase
  Speed:    ~149 seconds training time

📊 Segmentation Strategy
  Price Tier Strategy: Low (0-5B) | Mid (5-20B) | High (20B+)
  Ensemble Method:     Weighted average of 3 algorithms per tier
```

---

## 🚀 Quick Start

### Training
```bash
python models/scripts/train_production.py
```

### Inference (via Streamlit App)
```bash
cd app/
streamlit run app.py
```

### Direct Python Usage
```python
from app.inference import load_models, build_row, predict_price
from app.geo import GeoLookup

# Load models (9 files)
models, meta, medians = load_models()
geo = GeoLookup()

# Build 78-feature row
row, info = build_row(
    medians, geo,
    street="Nguyễn Hữu Cảnh",
    locality="Phường Bình Thạnh",
    property_type="nha_mat_tien",
    legal_status="so_hong_so_do",
    direction="dong_nam",
    area_m2=100, width_m=4, length_m=25,
    num_floors=3, num_bedrooms=3, road_width_m=20,
    bin_flags={"terrace_bin": 1, "car_parking_bin": 1},
    text_flags={"is_gap": 0, "is_kinh_doanh": 1}
)

# Predict price for mid-tier
price = predict_price(models, meta, row, price_tier="mid")
print(f"Predicted: {price/1e9:.2f}B VND")
```

---

## 📂 Model Files

### Production Models (9 files)
```
saved_models/
├── lgbm_low.pkl        LightGBM for low-price (0-5B VND)
├── lgbm_mid.pkl        LightGBM for mid-price (5-20B VND)
├── lgbm_high.pkl       LightGBM for high-price (20B+ VND)
├── xgb_low.pkl         XGBoost for low-price tier
├── xgb_mid.pkl         XGBoost for mid-price tier
├── xgb_high.pkl        XGBoost for high-price tier
├── cb_low.pkl          CatBoost for low-price tier
├── cb_mid.pkl          CatBoost for mid-price tier
└── cb_high.pkl         CatBoost for high-price tier
```

### Analysis & Documentation
```
saved_models/
├── feature_analysis/              Feature importance analysis
│   ├── top30_features.png         Top 30 features visualization
│   ├── cumulative_importance.png  Cumulative curve
│   └── feature_importance_report.txt
│
└── redundancy_analysis/           VIF + correlation analysis (v2.8 experiment)
    ├── pca_cumulative_variance.png
    ├── redundancy_analysis_report.txt
    └── features_to_drop.txt
```

---

## 📊 Features (78 Total)

### Core (6)
`num_floors`, `num_bedrooms`, `road_width_m`, `width_m`, `length_m`, `area_m2`

### Locality (2)
`locality_square`, `locality_population_density`

### Distance & POI (15)
`distance_to_center_km`, `nearest_school_km`, `school_count_3km`, `nearest_hospital_km`, `hospital_count_5km`, `nearest_marketplace_km`, `marketplace_count_3km`, `nearest_supermarket_km`, `supermarket_count_3km`, `nearest_mall_km`, `mall_count_3km`, `nearest_bus_stop_km`, `bus_stop_count_1km`, `nearest_metro_km`, `metro_count_5km`

### Temporal (2)
`post_day_month`, `post_day_day` (set to 0 at inference)

### Flags & Derived (5)
`road_width_m_missing`, `perimeter_m`, `shape_ratio`, `shape_ratio_missing`, `width_m_missing`

### Text Features (6)
`is_hem_xe_hoi`, `is_mat_tien`, `is_no_hau`, `has_noi_that`, `is_gap`, `is_kinh_doanh`

### Base Interactions (7)
`area_x_floors`, `area_x_bedrooms`, `area_per_bedroom`, `distance_vs_area`, `log_area`, `log_distance_to_center`, `log_population_density`

### Ratios (3)
`frontage_ratio`, `depth_ratio`, `road_area_ratio`

### Polynomial Features ⭐ (6)
`area_m2_squared`, `area_m2_sqrt`, `distance_squared`, `road_width_squared`, `bedrooms_squared`, `floors_squared`

### Advanced Interactions ⭐ (8)
`area_x_distance`, `area_per_distance`, `bedrooms_x_distance`, `floors_x_distance`, `area_x_road_width`, `width_x_length`, `density_x_area`, `locality_sq_x_area`

### Categorical One-Hot (14)
- Direction (4): `direction_dong_bac`, `direction_dong_nam`, `direction_nam`, `direction_unknown`
- Property (2): `property_type_nha_mat_tien`, `property_type_nha_trong_hem`
- Legal (3): `legal_status_giay_to_hop_le`, `legal_status_so_hong_so_do`, `legal_status_unknown`
- Amenities (5): `dining_room_bin_False`, `terrace_bin_False`, `terrace_bin_True`, `car_parking_bin_False`, `car_parking_bin_True`

---

## 📈 Performance Comparison

| Version | Features | Strategy | MAPE | R² | Approach |
|---------|----------|----------|------|----|----|
| v2.4 | 64 | Baseline | 13.25% | 0.9164 | Base features |
| v2.5 | 45 | Remove low-impact | 13.36% | 0.9167 | ❌ Noise was useful |
| **v2.6** | **78** | **+Polynomial** | **13.10%** | **0.9200** | ✅ **BEST** |
| v2.7 | 88 | +Cubic/aggressive | 13.17% | 0.9193 | ❌ Overfitting |
| v2.8 | 59 | Remove redundant | 13.16% | 0.9192 | ❌ Lost signal |

**Why v2.6 Wins:**
- ✅ **Lowest MAPE**: 13.10% (polynomial features capture non-linearity)
- ✅ **Highest R²**: 0.9200 (explains 92% of variance)
- ✅ **No Overfitting**: Adds just polynomial terms (v2.7 was too aggressive)
- ✅ **Balanced**: 78 features = good signal-to-noise ratio

---

## 🔧 Training Pipeline

### Data Processing (`shared/preprocessing.py`)

1. **Load**: Supabase `Raw_Features` + CSV fallback
2. **Clean**: Remove duplicates, filter outliers (2-50B VND, 15-500 m² area)
3. **Encode**: Boolean → int, categorical → one-hot
4. **Impute**: Hierarchical (property_type + area_segment → property_type → global)
5. **Engineer**: Polynomial, interactions, ratios, log transforms
6. **Output**: 10,421 × 78 features

### Hyperparameters

```python
LGBM:
  n_estimators: 1000, max_depth: 8, learning_rate: 0.05
  subsample: 0.8, colsample_bytree: 0.8

XGBoost:
  n_estimators: 1500, max_depth: 8, learning_rate: 0.03
  subsample: 0.8, colsample_bytree: 0.8

CatBoost:
  iterations: 1500, depth: 8, learning_rate: 0.05
  loss_function: RMSE, early_stopping_rounds: 50
```

### Price Tiers
- **Low**: 0-5 billion VND (924 train, 232 test)
- **Mid**: 5-20 billion VND (5,069 train, 1,250 test)
- **High**: 20+ billion VND (2,343 train, 603 test)

---

## ⚡ Scripts

### Main Training
```bash
python models/scripts/train_production.py
```
- Trains all 9 models (3 tiers × 3 algorithms)
- Saves to `saved_models/`
- Outputs training data to `data/processed/model_training_data.csv`
- Time: ~149 seconds

### Feature Analysis
```bash
python models/scripts/feature_importance_analysis.py
```
- Extracts importance from all 9 models
- Identifies top 30 and bottom 30% features
- Generates visualizations + detailed report

### Redundancy Analysis (Experimental)
```bash
python models/scripts/feature_redundancy_analysis.py
```
- Calculates VIF for multicollinearity
- Correlation matrix for redundant features
- PCA variance explained
- Note: v2.8 removed redundancy but hurt MAPE → kept v2.6

---

## 🎯 Why Polynomial Features Work

**Non-linear Relationships:**
```python
# Linear doesn't capture real-world dynamics
price ≈ 100M × area_m2  # Wrong!

# Polynomial captures diminishing/accelerating returns
price ≈ 100M × area_m2 + 50k × area_m2²  # More realistic

# Examples:
- Large plots far from center: less valuable per sqm
- Small plots in center: more valuable per sqm
- Distance decay is curved, not linear
```

**Results:**
- v2.4 (no polynomial): 13.25% MAPE
- v2.6 (with polynomial): 13.10% MAPE
- **Improvement: +0.15% MAPE**

---

## ⚠️ Known Limitations

1. **Target Not Met**: 13.10% MAPE vs 10% goal
   - Real estate has inherent variance (market timing, subjective valuations)
   - 10% is very high bar (top-tier performance)
   - v2.6 is best achievable with current data

2. **Data Quality**: Some listings incomplete
   - Mitigation: Hierarchical imputation + median filling
   - No critical issues found in analysis

3. **Temporal**: Post-day set to 0 at inference
   - Rationale: Inference doesn't know listing date
   - Impact: Minimal (~1.8% importance)

4. **Market Changes**: Trained on 2025 data
   - Recommend quarterly retraining
   - Monitor prediction residuals

---

## 📋 Files & Directories

```
models/
├── README.md                          This file
├── scripts/
│   ├── train_production.py            Main training (v2.6)
│   ├── feature_importance_analysis.py Feature analysis
│   ├── feature_redundancy_analysis.py Redundancy detection
│   ├── train_stacking.py              Stacking (experimental)
│   ├── shared/
│   │   └── preprocessing.py           Feature engineering
│   └── data/
│       └── raw_data.csv               Raw training data
├── saved_models/
│   ├── *.pkl                          9 production models
│   ├── feature_analysis/              Importance analysis
│   ├── redundancy_analysis/           Redundancy results
│   └── plots/                         Visualizations
├── data/
│   ├── raw/                           Raw data
│   ├── processed/                     model_training_data.csv
│   └── external/                      Density data
└── docs/
    ├── DEPLOYMENT.md                  Deployment guide
    ├── OPTIMIZATION_SUMMARY.md        Optimization history
    ├── PRODUCTION_STATUS.md           Current status
    └── XAI_ANALYSIS_SUMMARY.md        Explainability
```

---

## 🚀 Production Deployment

### Checklist
- ✅ Models trained and saved (9 files)
- ✅ Feature schema documented (78 features)
- ✅ Inference pipeline tested (build_row → predict)
- ✅ Streamlit app deployed
- ✅ Fallbacks implemented (CSV if Supabase fails)

### Monitoring
- Track prediction errors vs actual prices
- Monitor residuals for bias
- Alert if MAPE drifts >15%
- Retrain quarterly with fresh data

---

## 📞 Support

**Version**: v2.6 (Production)  
**Last Updated**: 2026-07-21  
**Data Source**: Supabase Raw_Features + Local CSV fallback  
**Status**: ✅ Production Ready  

For issues, refer to `/docs/` or check the Streamlit app logs.

---

## 📜 License

Internal use only. Real Estate Valuation Capstone Project.
