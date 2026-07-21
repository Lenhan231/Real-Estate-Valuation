# Model Improvement Plan

## Current Baseline
- **Model:** Price-Only Ensemble (LGBM + XGB + CatBoost)
- **MAPE:** 13.38%
- **R²:** 0.9164
- **Target:** <10% MAPE

## Three Improvement Paths

### 1️⃣ **Hyperparameter Tuning** (Quick)
**Effort:** Medium | **Impact:** 0.5-1.5% improvement | **Time:** 2-3 hours

**What to tune:**
- Learning rate: 0.03 → try 0.01, 0.05, 0.1
- Max depth: 8 → try 6, 10, 12
- Number of estimators: 1000 → try 500, 1500, 2000
- Subsample: 0.8 → try 0.6, 0.9

**Expected gain:** 13.38% → ~12.5-13% MAPE

**Strategy:**
- Grid search or random search
- Use validation set to pick best params
- Test on each model independently (LGBM, XGB, CatBoost)

---

### 3️⃣ **Ensemble Stacking** (Advanced)
**Effort:** High | **Impact:** 1-2% improvement | **Time:** 4-6 hours

**What it is:**
```
Level 0 (Base models):
├─ LGBM predictions (on validation set)
├─ XGB predictions (on validation set)
└─ CatBoost predictions (on validation set)
        ↓
Level 1 (Meta-learner):
└─ Ridge/Lasso/Neural Network learns optimal weights
        ↓
Final prediction: Much better ensemble
```

**Expected gain:** 13.38% → ~11.5-12.5% MAPE

**Why better than current weighted average:**
- Current: Fixed weights (inverse RMSE)
- Stacking: Learned weights (adaptive to different input patterns)
- Captures when LGBM is best vs XGB vs CatBoost

---

### 4️⃣ **Data Quality Improvements** (Research)
**Effort:** Very High | **Impact:** 2-5% improvement | **Time:** 6-12 hours

**Investigation areas:**

A. **Outlier detection**
   - Identify prices that don't match features (data entry errors)
   - Example: 2-bedroom 50m² apartment for 200B VND (unrealistic)
   - Strategy: Use Isolation Forest or statistical methods

B. **Missing value patterns**
   - Some features missing systematically (not random)
   - Example: Luxury properties missing "num_bedrooms" (data collection issue)
   - Strategy: Analyze missing patterns, consider separate models for sparse data

C. **Data consistency checks**
   - Validate relationships (price_per_sqm vs price vs area)
   - Check geographic coordinates (within valid range)
   - Verify temporal data (posting dates make sense)

D. **Feature quality**
   - Some engineered features might have errors
   - Example: distance calculation error for outliers
   - Strategy: Recalculate key features from raw data

E. **New data collection**
   - If available: Get more recent data
   - More training samples = better model
   - Target: 15k-20k rows (currently 8.3k train)

**Expected gain:** 13.38% → ~11-12% MAPE (if data issues found & fixed)

---

## Recommended Sequence

### Phase 1: Quick Wins (Hyperparameter Tuning)
1. Grid search on LGBM, XGB, CatBoost separately
2. Find best params per model
3. Test combined → Should get 12.5-13% MAPE
4. Time: 2-3 hours

### Phase 2: Advanced (Ensemble Stacking)
1. Implement stacking with Ridge meta-learner
2. Validate improvements
3. Compare to current weighted ensemble
4. Time: 4-6 hours
5. Expected: 11.5-12.5% MAPE

### Phase 3: Deep Dive (Data Quality)
1. Outlier detection and removal
2. Missing value pattern analysis
3. Feature validation
4. Potentially improve to 11-12% MAPE
5. Time: 6-12 hours

---

## Success Criteria

| MAPE | Status |
| --- | --- |
| 13.38% | ✅ Current |
| 12.5-13% | 🎯 Phase 1 (Tuning) |
| 11.5-12.5% | 🎯 Phase 2 (Stacking) |
| 11-12% | 🎯 Phase 3 (Data Quality) |
| <10% | 🚀 Target (might need all 3 + more) |

---

## Next Steps

Choose approach:
- [ ] **Start with Phase 1 (Quick tuning)**
- [ ] **Go straight to Phase 2 (Stacking)**
- [ ] **Investigate Phase 3 (Data quality)**
- [ ] **Do all three in sequence**
