# 🏠 Real Estate Valuation App - v2.6

Production Streamlit application for automated real estate price prediction in Vietnam.

## 🚀 Quick Start

```bash
streamlit run app.py
```

Opens: `http://localhost:8501`

## 📱 Features

### Tab 1: 💰 Định giá (Valuation)

- **Listing Parser**: Paste Vietnamese real estate listing text → automatic feature extraction
- **Manual Form**: Fill property details manually (area, floors, bedrooms, location, etc.)
- **Feature Visualization**: 80-feature table showing all engineered features
- **Price Prediction**: 3-tier ensemble model (low/mid/high price ranges)
- **Confidence Bounds**: MAPE-based error estimation

### Tab 2: 📊 Phân tích thị trường (Market Analysis)

- **Price Heatmaps**: Visualize prices across districts/localities
- **Market Trends**: Price trends by property type over time
- **Top Localities**: Rankings by average price and density
- **Segment Analysis**: Compare across property types and price tiers

## 🏗️ Architecture

```
User Input (Text or Form)
    ↓
Geolocation Lookup (Supabase + CSV fallback)
    ↓
POI & Locality Stats (Distance, amenities, density)
    ↓
Preprocessing Pipeline (shared/preprocessing.py)
    - Feature engineering (78 base features)
    - Hierarchical imputation
    - Polynomial/interaction features
    ↓
Locality Encoding (from training data)
    - Add 2 locality-based features
    ↓
3-Tier Ensemble (LightGBM + XGBoost + CatBoost)
    - Route by price tier (0-5B | 5-20B | 20B+)
    - Average 3 model predictions
    ↓
Return Price Estimate ± MAPE Confidence Bounds
```

## 📊 Model Architecture (v2.6)

**3-Tier Price Segmentation:**

- **Low**: 0-5 billion VND (924 samples)
- **Mid**: 5-20 billion VND (5,069 samples)
- **High**: 20+ billion VND (2,343 samples)

**Ensemble per Tier:**

- LightGBM (n_estimators=1000, max_depth=8)
- XGBoost (n_estimators=1500, max_depth=8)
- CatBoost (iterations=1500, depth=8)

**Performance:**

```
Global MAPE:  13.10%
Global R²:    0.9200
Global MAE:   2.15 Billion VND
Global RMSE:  3.41 Billion VND
```

## 🎯 Features (80 Total)

**78 from preprocessing:**

- Core (6): num_floors, num_bedrooms, road_width_m, width_m, length_m, area_m2
- Locality (2): locality_square, locality_population_density
- Distance & POI (15): distance_to_center_km + 14 POI features
- Temporal (2): post_day_month, post_day_day
- Flags & Derived (5): road_width_m_missing, perimeter_m, shape_ratio, etc.
- Text Features (6): is_hem_xe_hoi, is_mat_tien, is_no_hau, has_noi_that, is_gap, is_kinh_doanh
- Base Interactions (7): area_x_floors, area_x_bedrooms, log_area, etc.
- Ratios (3): frontage_ratio, depth_ratio, road_area_ratio
- Polynomial (6): area_m2_squared, area_m2_sqrt, distance_squared, etc.
- Advanced Interactions (8): area_x_distance, density_x_area, etc.
- Categorical One-Hot (14): direction (4), property_type (2), legal_status (3), amenities (5)

**2 from locality encoding:**

- locality_price_median: Market price in locality
- price_per_sqm_market: Market price per sqm

See [models/README.md](../models/README.md) for detailed feature documentation.

## 🔧 Files

```
app/
├── app.py                  # Main Streamlit UI (2 tabs)
├── inference.py            # Model loading & prediction pipeline
├── geo.py                  # Geolocation & POI lookup (Supabase + CSV)
├── constants.py            # Shared constants (patterns, configurations)
├── parsers.py              # Listing text parsing
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## 💾 Data Flow

### Loading Phase (app startup)

```python
models, meta, medians = load_models()
# - Load 9 models (3 tiers × 3 algorithms)
# - Extract feature names from model.feature_names_in_
# - Compute locality price/sqm maps from training data
# - Load training medians for fallback imputation
```

### Prediction Phase (per listing)

```python
row, info = build_row(meta, geo, ...)
# - Geocode address → lat, lon, district
# - Lookup POIs (schools, hospitals, markets, etc.)
# - Get locality stats (area, population density)
# - Run through preprocessing.preprocess() → 78 features
# - Add locality encoding → 2 features
# - Return 80-feature dict ready for models

price = predict_price(models, meta, row, price_tier)
# - Route to correct tier model set
# - Average 3 model predictions
# - Return VND price estimate
```

## 🛠️ Development

### Adding Features

1. Edit `models/scripts/shared/preprocessing.py`
2. Retrain models: `python models/scripts/train_production.py`
3. Models auto-sync (feature_names_in_ is source of truth)
4. App auto-picks up new features on next run

### Debugging

- Enable Streamlit logger: `streamlit run app.py --logger.level=debug`
- Check geolocation fallbacks in `geo.py`
- Monitor preprocessing steps in `inference.py`

### Configuration

Edit `app/constants.py` for:

- Text pattern matching
- Numeric extraction patterns
- Price tier boundaries (future)
- Model paths (future)

## ⚙️ Dependencies

```
streamlit>=1.28
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
lightgbm>=4.0
xgboost>=2.0
catboost>=1.2
postgrest-py  # Supabase
```

See `requirements.txt` for exact versions.

## 🔐 Environment Variables

For Supabase geolocation lookup:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

If not set, app falls back to CSV (`data/external/locality_stats.csv`).

## 📋 Testing Checklist

- [ ] Listing parser extracts features correctly
- [ ] Manual form submission works
- [ ] Geolocation resolves addresses
- [ ] Predictions are in expected range
- [ ] Confidence bounds make sense (±13.25% MAPE)
- [ ] Market dashboard loads without errors
- [ ] Feature table shows 80 features

## 🐛 Known Issues

1. **Geolocation**: Vietnamese spelling sensitivity (try variations)
2. **Missing POIs**: Falls back to median if lookup fails
3. **Outliers**: Extreme properties may have poor predictions
4. **Temporal**: post_day set to 0 (listing date unknown at inference)

## 📞 Support

- Model training docs: [models/README.md](../models/README.md)
- Deployment guide: [models/docs/DEPLOYMENT.md](../models/docs/DEPLOYMENT.md)
- Feature analysis: [models/saved_models/feature_analysis/](../models/saved_models/feature_analysis/)

---

**Version**: v2.6 (Production)  
**Last Updated**: 2026-07-22  
**Status**: ✅ Production Ready
