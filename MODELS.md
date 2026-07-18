# 🤖 Model Training & Selection Guide

**Everything you need to know about model architecture, training, and production usage**

---

## 🎯 Quick Summary

| Aspect | Details |
|--------|---------|
| **Production Model** | XGBoost (Gradient Boosting) |
| **Training Data** | 8,345 properties (80% split) |
| **Test Data** | 2,087 properties (20% split) |
| **Features** | 166 engineered features |
| **Target Variable** | Price (log-transformed) |
| **MAPE** | 18.01% |
| **R² Score** | 0.8663 |
| **Confidence Interval** | ±2.67B VND @ 95% (±1 MAE × 1.96) |

---

## 📊 Model Comparison Results

### Performance Metrics (Test Set)

| Model | MAPE | R² | RMSE | MAE |
|-------|------|-----|------|-----|
| **XGBoost ⭐** | **18.01%** | **0.8663** | 4.37B | 2.67B |
| LightGBM | 18.76% | 0.8642 | 4.52B | 2.89B |
| CatBoost | 19.52% | 0.8529 | 4.89B | 3.12B |

### Why XGBoost Wins

1. **Best MAPE** - Most accurate percentage error (lowest is better)
2. **Best R²** - Explains most variance
3. **Stable training** - Fewer tuning parameters
4. **Fast inference** - Real-time predictions

**Alternatives:**
- **LightGBM:** Only 0.75% worse MAPE, faster training
- **CatBoost:** Better with categorical features, but slower to train

---

## 🏗️ Model Architecture

### XGBoost Hyperparameters

```python
{
    'n_estimators': 500,        # Number of boosting rounds
    'max_depth': 8,             # Tree depth (deeper = more complex)
    'learning_rate': 0.03,      # Step size (lower = slower but better fit)
    'subsample': 0.8,           # Row sampling (prevent overfitting)
    'colsample_bytree': 0.8,    # Column sampling per tree
    'min_child_weight': 1,      # Min child samples per leaf
    'gamma': 0,                 # Min loss reduction to split
    'objective': 'reg:squarederror',  # Regression task
}
```

### What Each Parameter Does

| Parameter | Effect |
|-----------|--------|
| `n_estimators=500` | 500 weak learners combined = strong model |
| `max_depth=8` | Max tree depth (8 = not too simple, not too complex) |
| `learning_rate=0.03` | Small steps = better generalization |
| `subsample=0.8` | Use 80% rows per round = reduce variance |
| `colsample_bytree=0.8` | Use 80% features per tree = reduce overfitting |

---

## 📈 Training Process

### Step 1: Data Preparation

```python
# Load cleaned features
df = fetch_csv_from_supabase()  # 10,432 properties
X = df.drop('price', axis=1)    # 166 features
y = df['price']                 # Target: price in VND

# Log transform for better fit
y = np.log1p(y)  # log(1 + price)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# X_train: 8,345 samples
# X_test:  2,087 samples
```

**Why log-transform?**
- Prices are right-skewed (few ultra-expensive, many normal)
- Log transform makes distribution more normal
- Better gradient descent convergence

### Step 2: Model Training

```python
model = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1  # Parallel processing
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],  # Monitor test performance
    early_stopping_rounds=50,       # Stop if no improvement
    verbose=10
)
```

**Training takes:** ~2-5 minutes (depending on hardware)

### Step 3: Model Evaluation

```python
# Predictions
y_pred = model.predict(X_test)

# Inverse log transform to get back to original prices
y_pred_price = np.expm1(y_pred)  # exp(pred) - 1
y_test_price = np.expm1(y_test)

# Calculate metrics
mape = mean_absolute_percentage_error(y_test_price, y_pred_price)
r2 = r2_score(y_test_price, y_pred_price)
mae = mean_absolute_error(y_test_price, y_pred_price)
rmse = np.sqrt(mean_squared_error(y_test_price, y_pred_price))
```

### Step 4: Save Model

```python
import joblib

joblib.dump(model, 'models/production/production_model.pkl')
# File size: ~8 MB
```

---

## 📊 Feature Importance

### Top 10 Most Important Features

```
Feature                 Importance (%)
1. area_m2             24.5%  ← Area is CRITICAL
2. width_m             19.8%
3. road_width_m        16.7%
4. num_bedrooms        14.3%
5. supermarket_count    8.9%
6. school_count         7.8%
7. log_distance         6.2%
8. location_score       5.4%
9. district_encoded     4.1%
10. hospital_count      3.7%
```

### Key Insight

**Physical dimensions (area, width, depth) account for ~61% of model's decision-making.**

Location & amenities matter, but dimension is king.

---

## 🎯 Model Performance Analysis

### Error Distribution

```
MAPE (Mean Absolute Percentage Error):
- 18.01% means predictions are off by ~18% on average
- Example: Predict 10B, actual is 8.18B - 12.18B

Test Set Distribution:
- 50% of predictions: within ±2.5% of actual
- 75% of predictions: within ±8.2% of actual
- 90% of predictions: within ±15% of actual
- 95% of predictions: within ±20% of actual
```

### Confidence Intervals

```
For a prediction of 15B VND:
- Point estimate: 15.00B
- 95% CI: ±5.24B (15.00 ± 2.67×1.96)
- Range: 9.76B - 20.24B
```

---

## 🚀 Using the Model

### Method 1: Python (Direct)

```python
import joblib
import numpy as np

# Load model
model = joblib.load('models/production/production_model.pkl')

# Prepare features (166 features)
features = np.zeros((1, 166))
features[0, 5] = 100   # area_m2
features[0, 0] = 3     # num_floors
features[0, 1] = 2     # num_bedrooms
# ... set all 166 features

# Predict
price_log = model.predict(features)[0]
price_vnd = np.expm1(price_log)  # Convert from log space
print(f"Predicted price: {price_vnd / 1e9:.1f}B VND")
```

### Method 2: Via Python Wrapper

```python
from app.inference_simple import load_model, predict_price

model = load_model()
predictions = predict_price(features)  # Returns price in VND
```

### Method 3: Web App

```bash
streamlit run app/app_simple.py
# Visit http://localhost:8501
# Input: area, floors, bedrooms → Output: predicted price
```

### Method 4: REST API

```bash
# Start server
python app/api_simple.py

# Make request
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"area_m2": 100, "num_floors": 3, ...}'
```

---

## 🔄 Model Updates & Retraining

### When to Retrain

**Retrain if:**
1. New data collected (>500 new properties)
2. Model performance drops (MAPE > 20%)
3. Market conditions change significantly
4. Feature engineering pipeline updated

**Don't retrain if:**
- Just using for predictions
- Testing on same data distribution

### How to Retrain

**Option 1: Full Retraining**
```bash
jupyter notebook notebooks/02_model_training/02_model_training.ipynb
# Trains LightGBM, XGBoost, CatBoost
# Compares metrics
# Saves best to models/production/
```

**Option 2: Script-based**
```bash
python run_training.py
# Trains & saves to models/save_models/
```

### Checklist Before Retraining

- [ ] Data cleaned & features engineered
- [ ] 166 features in correct order
- [ ] Train/test split is 80/20
- [ ] Log-transformed target variable
- [ ] Backup old model: `cp models/production/production_model.pkl models/archive/`

---

## ⚠️ Model Limitations

### What It Can Predict Well
- Standard residential properties
- Size range: 15-500 m²
- Price range: 2-50B VND
- Ho Chi Minh City locations

### What It Can't Predict Well
- **Ultra-luxury properties** (>50B) - too few in training data
- **Micro apartments** (<15 m²) - out of distribution
- **Special use** (industrial, commercial) - not trained on these
- **New developments** - market conditions may differ

### Known Issues

1. **Tends to underpredict luxury** - Mean reversion bias
2. **Struggles with unusual dimensions** - e.g., 2m wide, 100m deep
3. **Location-biased** - Better for central districts (1,3,4), worse for outlying

---

## 🧪 Model Evaluation Metrics Explained

### MAPE (Mean Absolute Percentage Error)
```
MAPE = mean(|actual - predicted| / |actual|) × 100%

Example: Actual 10B, Predicted 9B
  Error = |10 - 9| / 10 = 10%

Why use it?
- Scale-independent (percent error)
- Easier to interpret than absolute errors
- Standard in real estate valuation
```

### R² (Coefficient of Determination)
```
R² = 1 - (SS_residual / SS_total)

Meaning:
- R² = 0.8663 means model explains 86.63% of price variance
- Remaining 13.37% due to unmeasured factors (market sentiment, timing, etc.)

Scale:
- 0.0 = model predicts mean (useless)
- 0.5 = explains half the variance (okay)
- 0.8+ = explains most variance (good)
- 1.0 = perfect predictions (unrealistic)
```

### RMSE (Root Mean Squared Error)
```
RMSE = sqrt(mean((actual - predicted)²))

Units: Billion VND
- Our model: 4.37B VND RMSE
- Useful for averaging errors, but heavily weighted to outliers
```

### MAE (Mean Absolute Error)
```
MAE = mean(|actual - predicted|)

Units: Billion VND
- Our model: 2.67B VND MAE
- Less sensitive to outliers than RMSE
- Used for confidence intervals
```

---

## 🎓 Hyperparameter Tuning History

### Baseline (Current Production)
```
n_estimators=500, max_depth=8, learning_rate=0.03
Result: MAPE 18.01% ✅
```

### Tuning Attempt 1 (More Aggressive)
```
n_estimators=1000, max_depth=12, learning_rate=0.05
Result: MAPE 20.68% ❌ (Worse - overfitting)
```

### Learning
- Deeper trees + higher learning rate = overfitting
- Current baseline is well-balanced
- **Lesson:** Don't over-tune; simpler is better

---

## 🔮 Future Improvements

### Quick Wins
1. **Feature selection** - Drop low-importance features (noise reduction)
2. **Ensemble** - Combine XGBoost + LightGBM (averaging predictions)
3. **Market segmentation** - Separate models for low/mid/high-end properties

### Medium-term
1. **More data** - Collect 50k+ properties (current: 10.4k)
2. **Feature engineering** - Add text NLP, satellite imagery
3. **Temporal modeling** - Track price trends over time

### Long-term
1. **Deep learning** - Neural networks (need >100k data)
2. **Real-time updates** - Continuous retraining
3. **Production deployment** - Cloud serving (AWS/GCP)

---

## 📁 Model Files Structure

```
models/
├── production/
│   ├── production_model.pkl       # XGBoost (8 MB)
│   ├── model_results.csv          # Metrics & predictions
│   └── README.md
│
├── archive/
│   ├── lightgbm_model.pkl         # Previous model (18.76% MAPE)
│   ├── catboost_model.pkl         # Previous model (19.52% MAPE)
│   └── xgboost_tuned_model.pkl    # Tuning experiment
│
├── data/
│   ├── model_ready_data.csv       # Training data
│   └── predictions.csv            # Saved predictions
│
└── README.md
```

---

## 🐛 Troubleshooting

### "Model won't load"
```python
import joblib
try:
    model = joblib.load('models/production/production_model.pkl')
except Exception as e:
    print(f"Error: {e}")
    # Check: file exists, not corrupted, correct path
```

### "Feature mismatch error"
```
Error: Expected 166 features, got 150

Fix:
1. Check feature_pipeline.py generated correct features
2. Re-run: python pipeline/transformation/feature_pipeline.py
3. Verify feature order matches training data
```

### "Predictions are all the same"
```
Cause: Model loaded incorrectly or features not scaled

Fix:
1. Reload model: model = joblib.load(...)
2. Check features: no NaN, correct dtypes (float)
3. Feature ranges similar to training data
```

---

**Last Updated:** 2026-07-18  
**Model Version:** v1.0 (XGBoost)  
**Training Date:** 2026-07-17
