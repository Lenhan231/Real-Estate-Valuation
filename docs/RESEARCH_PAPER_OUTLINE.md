# Research Paper Outline: Automated Real Estate Valuation Using Ensemble Learning

## Title
**"A Price-Stratified Ensemble Approach to Automated Property Valuation: Balancing Heteroscedasticity and Model Complexity"**

---

## 1. Abstract (150-200 words)

Real estate valuation traditionally relies on manual appraisals, creating bottlenecks for portfolio assessment and market transparency. We propose a **price-stratified 3-model ensemble** (LightGBM, XGBoost, CatBoost) trained on 10,421 Vietnamese urban properties across three price tiers. Our approach achieves **13.10% MAPE and 0.9200 R²**, addressing inherent heteroscedasticity in high-price regions through ensemble diversity rather than loss-function manipulation. Analysis of 78 engineered features reveals that weak-importance features retain value through non-linear interactions, suggesting that traditional feature importance may be insufficient for ensemble optimization. The model successfully deployed on a production web application validates the feasibility of automated valuation at scale.

---

## 2. Introduction (1-2 pages)

### 2.1 Motivation & Problem Statement
- Vietnamese real estate market context: rapid urbanization, price volatility (+26% in Hanoi condos recently)
- Limitation of manual appraisals: subjective, slow, unscalable
- Market need: instantaneous, objective price estimates for buyers, sellers, lenders
- Research gap: Most prior work focuses on single models; ensemble approaches with price stratification underexplored in emerging-market real estate

### 2.2 Research Questions
1. Can price-stratified ensembles outperform single-model approaches on heteroscedastic real estate data?
2. How do feature importance metrics relate to ensemble performance in tabular data?
3. What is the practical trade-off between prediction accuracy (13.10% MAPE) and model simplicity for deployment?

### 2.3 Contributions
- Novel finding: Tree-based feature importance does not predict ensemble performance; weak-importance features provide non-linear signal
- Empirical validation: Price stratification + ensemble diversity > weighted loss functions for heteroscedasticity
- Practical deployment: End-to-end system (data pipeline → model → API → web dashboard)

---

## 3. Literature Review (1.5-2 pages)

### 3.1 Property Valuation Methods
- Hedonic pricing models (traditional)
- Automated Valuation Models (AVMs) — e.g., Zillow Zestimate
- Machine learning approaches: KNN, linear regression, tree-based methods
- **Gap:** Limited work on ensemble methods in emerging markets; heteroscedasticity often ignored

### 3.2 Handling Heteroscedasticity in Regression
- Classic approaches: weighted least squares, robust regression
- Modern: quantile regression, gradient boosting with custom loss (Huber loss, etc.)
- Issue: Explicit weighting often degrades tree-model performance due to overfitting

### 3.3 Feature Importance in Ensemble Learning
- SHAP, TreeExplainer, permutation importance
- Finding: Importance rankings don't always correlate with downstream performance
- Reason: Non-linear interactions, multi-collinearity, ensemble voting masks weak-feature contributions

### 3.4 Real Estate AI in Vietnam
- Limited academic literature; most work in US/UK markets
- Alonhadat, Batdongsan platforms use rule-based or simple ML models
- Opportunity: Rigorous ML ensemble approach for Vietnamese market

---

## 4. Methodology (2-2.5 pages)

### 4.1 Dataset
- **Source:** Alonhadat.com.vn, manually collected & enriched
- **Sample Size:** 12,814 property listings → 10,421 after cleaning
- **Geographic Scope:** Primarily Ho Chi Minh City, secondary Hanoi/Da Nang
- **Target Variable:** Transaction price (VND), log-transformed for training
- **Feature Engineering:** 78 features (64 base + 14 polynomial/interaction)

### 4.2 Price Stratification Strategy
- **Motivation:** Heteroscedastic errors increase with price; different price tiers reflect different market segments
- **Bins:**
  - Low: 0-5B VND (affordable, location-centric)
  - Mid: 5-20B VND (middle-market, structure-sensitive)
  - High: 20B+ VND (luxury, amenity-sensitive, sparse)
- **Data Split:**
  - Low: 924 train / 232 test
  - Mid: 5,069 train / 1,250 test
  - High: 2,343 train / 603 test

### 4.3 Ensemble Architecture
- **3 Models per Tier:**
  - LightGBM: 1000 estimators, depth 8, lr=0.05 (fast, good for mid-tier)
  - XGBoost: 1500 estimators, depth 8, lr=0.03 (stable, good for high-tier)
  - CatBoost: 1500 iterations, depth 8, lr=0.05 (categorical handling, good for low-tier)
  
- **Ensemble Method:** Inverse-RMSE weighted average
  - Weight = 1/RMSE_val / Σ(1/RMSE_val)
  - Rationale: Better-performing models get higher weight

### 4.4 Feature Engineering
- **Base Features (64):** area, floors, bedrooms, age, legal_status, distance_to_landmarks, neighborhood_encoded

- **Interaction Features (12):**
  1. area_x_floors (structural density)
  2. area_x_bedrooms (unit density)
  3. area_x_distance (size-location tradeoff)
  4. area_x_road_width (accessibility × size)
  5. area_per_distance (size efficiency per km)
  6. area_per_bedroom (area per room)
  7. bedrooms_x_distance (rooms × distance)
  8. floors_x_distance (height × distance)
  9. width_x_length (plot shape)
  10. density_x_area (population × size)
  11. locality_sq_x_area (prestige × size)
  12. distance_vs_area (distance relative to size)

- **Polynomial Features (6):**
  1. area_m2_squared (non-linear area scaling)
  2. area_m2_sqrt (concave scaling)
  3. distance_squared (non-linear distance)
  4. road_width_squared (street width effect)
  5. bedrooms_squared (unit density scaling)
  6. floors_squared (building height effect)

- **Locality Features (added post-split):** district_median_price, nearby_schools_density, proximity_to_metro

### 4.5 Optimization & Hyperparameter Selection
- Grid search: depth {6,8,10}, lr {0.01-0.1}, subsample {0.7-0.9}
- Early stopping on validation set (50-round patience)
- **Tested but Rejected:** Per-tier tuning, weighted loss functions (all degraded performance)

---

## 5. Results (1.5-2 pages)

### 5.1 Main Performance Metrics

```
Global Ensemble (All Tiers):
- MAPE: 13.10%
- R²: 0.9200
- MAE: 2.15 Billion VND (~US$85K at current rates)
- RMSE: 3.41 Billion VND (~US$135K)
```

### 5.2 Per-Tier Breakdown
| Tier | MAPE | R² | Samples | Notes |
|------|------|-----|---------|-------|
| Low | 11.8% | 0.938 | 232 | Tight predictions, location-driven |
| Mid | 12.5% | 0.925 | 1250 | Balanced errors, structure-important |
| High | 14.9% | 0.895 | 603 | Sparse, high-price noise |

### 5.3 Feature Importance Analysis

**Top 10 Features (by TreeExplainer):**
1. area_x_road_width (6.79%)
2. area_per_distance_to_center (5.45%)
3. area_x_floors (3.64%)
4. proximity_to_hcm_center (3.21%)
5. floor_area (3.10%)
6. distance_to_nearest_school (2.88%)
7. bedrooms (2.65%)
8. age (2.34%)
9. area_x_age (2.15%)
10. legal_status_encoded (1.98%)

**Key Finding:** Features ranked 11-78 individually <1% importance, but collectively impact model significantly. Removing 24 low-importance features → 13.20% MAPE (worse).

### 5.4 Optimization Experiments
- **Heteroscedasticity Weighting:** 13.11% MAPE (+0.01%) — **Failed**
- **Per-Tier Hyperparameter Tuning:** 13.12% MAPE (+0.02%) — **Failed**
- **Feature Pruning (v2.7):** 13.20% MAPE (+0.10%) — **Failed**

**Conclusion:** v2.6 represents a local optimum; marginal improvements require architectural changes or data expansion.

---

## 6. Discussion (1.5-2 pages)

### 6.1 Why Ensemble Outperforms Single Models
- **Bias-Variance Reduction:** LightGBM excels at mid-tier, XGBoost at high-tier, CatBoost at low-tier
- **Diversity:** 3 models find different feature subsets; voting reduces overfitting
- **Robustness:** Single model MAPE ranges 13.3-13.5%; ensemble achieves 13.10%

### 6.2 Heteroscedasticity & Why Weighting Failed
- **Observation:** High-tier RMSE higher (0.0280-0.0292 on log-scale) due to sparse, diverse luxury properties
- **Hypothesis Tested:** Sqrt-price weighting → focus on high-tier errors
- **Result:** 13.11% MAPE (worse). Why?
  - Tree-based models already downweight outliers via leaf-sample averaging
  - Explicit weighting causes overfitting to training-set high-price outliers
  - Ensemble averaging naturally handles heteroscedasticity better than single-model tricks
- **Implication:** For complex data, ensemble diversity > statistical fine-tuning

### 6.3 Feature Importance ≠ Ensemble Contribution
- **Finding:** v2.7 feature pruning (removing 24 weak features) degraded v2.6
- **Interpretation:** Tree-based importance is **local** (importance in a single tree), not **global** (contribution to ensemble)
  - Weak feature A might be crucial for XGBoost's split at node 45
  - Same feature unimportant in LightGBM's structure
  - Ensemble votes; weak feature still valuable
- **Implication:** Cannot optimize ensembles by importance ranking alone; must test empirically

### 6.4 Gap to Target (13.10% vs 10% MAPE)
- **Structural Limits:** Current dataset/architecture reaches 92% R²; diminishing returns beyond
- **Paths to <10%:**
  - **Short-term:** Collect more high-tier properties (data-heavy)
  - **Medium-term:** Add temporal features (price trends, market momentum)
  - **Long-term:** Quantile regression, satellite imagery, macro indicators
- **Practical Trade-off:** 13.10% MAPE is deployable; <10% would require 6-12 months effort

### 6.5 Deployment & Real-World Considerations
- **Latency:** Model inference <50ms; acceptable for web app
- **Interpretability:** SHAP values computed per prediction; explainable to end-users
- **Monitoring:** Deployed on production; W&B tracks MAPE drift over time
- **Retraining:** Weekly scheduled; handles market shifts

---

## 7. Conclusion (0.5-1 page)

We have demonstrated that **price-stratified ensemble learning** is effective for automated real estate valuation in emerging markets. Our v2.6 model achieves 13.10% MAPE on Vietnamese property data, outperforming single-model baselines by 0.2-0.4%.

Key insights:
1. **Ensemble diversity** handles heteroscedasticity better than loss-function manipulation
2. **Feature importance** metrics are insufficient for ensemble optimization; weak features provide non-linear value
3. **Price stratification** is necessary but not sufficient for <10% MAPE; requires data/feature expansion

The system is production-ready and deployed on a web application. Future work should focus on data collection (high-tier properties) and temporal feature engineering to reach <10% MAPE targets.

---

## 8. References (To Be Compiled)

### Key References to Include:
- Muth, R. F. (1969). "Cities and Housing." University of Chicago Press.
- Bourassa, S. C., et al. (2003). "Predicting housing prices." Journal of Real Estate Research.
- Zillow Research (2018). "Automated Valuation Models." Zillow Insights.
- Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." SIGKDD.
- Ke, G., et al. (2017). "LightGBM: A fast, distributed gradient boosting framework." NIPS.
- Prokhorenkova, L., et al. (2018). "CatBoost: gradient boosting for categorical features." NIPS.
- Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." NIPS.

---

## Appendix: Paper Status & Next Steps

### To Complete Paper:
- [ ] Compile full reference list (APA format)
- [ ] Add real citations (Zillow, academic papers on AVM)
- [ ] Write expanded sections (Results, Discussion)
- [ ] Add tables/figures (ROC curves, residual plots, SHAP summary)
- [ ] Polish for conference submission (IEEE, KDD, or regional ML conference)

### Target Publication Venues (Optional):
- IEEE Access (rapid, good for applied ML)
- Journal of Real Estate Research (domain-specific)
- ACM SIGKDD Regional Workshops (practice-oriented)

### Estimated Writing Time:
- Complete draft: 2-3 weeks
- Peer review + revision: 1-2 weeks
- Submission: ~4 weeks

---

**Document prepared:** 2026-07-22  
**Status:** Outline Complete — Ready for Full Draft
