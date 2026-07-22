# Production Strategy: v2.6 Model Maintenance & Monitoring

## Overview

This document outlines how to maintain, monitor, and iterate on the v2.6 production model post-deployment.

---

## 1. Deployment Checklist (Pre-Launch)

- [x] Model trained and saved (3 tiers × 3 models = 9 .pkl files)
- [x] API layer tested (`/app/api/`)
- [x] Web app frontend deployed (`/app/src/`)
- [x] Feature pipeline validated (78 features match training)
- [x] W&B experiment tracking initialized
- [x] Performance baseline recorded (13.10% MAPE, 0.9200 R²)
- [x] Documentation complete (README, XAI analysis)
- [ ] **TO DO:** Set up scheduled retraining (weekly cron job)
- [ ] **TO DO:** Configure monitoring alerts (MAPE drift >1.5%)
- [ ] **TO DO:** Database backups for model versioning

---

## 2. Monitoring & Alerting

### 2.1 Key Metrics to Track

| Metric | Baseline | Alert Threshold | Check Frequency |
|--------|----------|-----------------|-----------------|
| **MAPE (Global)** | 13.10% | >14.5% (↑1.4%) | Weekly |
| **R² (Global)** | 0.9200 | <0.9000 (↓0.02) | Weekly |
| **MAE (Billions VND)** | 2.15 | >2.50 (↑0.35) | Weekly |
| **RMSE (Billions VND)** | 3.41 | >3.80 (↑0.39) | Weekly |
| **High-Tier MAPE** | 14.9% | >16.5% | Weekly |
| **Inference Latency (ms)** | <50 | >150 | Daily |
| **API Error Rate** | <0.1% | >1.0% | Daily |

### 2.2 Monitoring Dashboard (W&B)

Currently logs:
```python
wandb.log({
    "mape": mape,
    "mae": mae,           # in VND
    "r2": r2,
    "rmse": rmse,         # in VND
})
```

**Recommended Additions:**
```python
wandb.log({
    "mape_low": mape_low,
    "mape_mid": mape_mid,
    "mape_high": mape_high,
    "latency_ms": inference_time,
    "n_predictions": len(y_test),
    "data_drift_detected": bool,
    "timestamp": datetime.now(),
})
```

---

## 3. Scheduled Retraining

### 3.1 Training Schedule

**Frequency:** Weekly (Monday 2 AM Asia/Saigon = 7 PM UTC Sunday)

**Why Weekly?**
- Captures market trends (price shifts, new listings)
- Not too frequent (avoid overfitting to noise)
- Not too sparse (market moves fast in Vietnam)

### 3.2 Retraining Workflow

```bash
# 1. Fetch fresh data from Supabase
python models/scripts/train_production.py --dataset production --data-source supabase --wandb

# 2. Compare performance
if new_mape < baseline_mape + 0.1%:
    # 3. Backup old model
    cp models/saved_models/lgbm_*.pkl models/saved_models/backup/lgbm_*.pkl.old
    cp models/saved_models/xgb_*.pkl models/saved_models/backup/xgb_*.pkl.old
    cp models/saved_models/cb_*.pkl models/saved_models/backup/cb_*.pkl.old
    
    # 4. Deploy new model
    # (API automatically loads from saved_models/)
    
    # 5. Log deployment
    git commit -m "retraining: weekly refresh v2.6 — MAPE $new_mape (vs $baseline_mape)"
else
    # Reject new model; keep previous version
    echo "Retraining rejected: MAPE degraded to $new_mape%"
fi
```

### 3.3 Automated Retraining (Scheduled Job)

**Tool:** GitHub Actions or cloud scheduler (see `/schedule` for cloud setup)

**Cron Expression (UTC):** `0 19 * * 0` (7 PM UTC Sunday = 2 AM Monday Asia/Saigon)

**Script:** `models/scripts/train_production.py --wandb`

**Failure Handling:**
- If retraining fails: send email alert, skip deployment, use previous model
- If MAPE degrades >0.1%: manual review before deployment

---

## 4. Data Drift Detection

### 4.1 How to Detect Drift

**Monitor:**
- Distribution shift in input features (e.g., average property age increasing)
- Distribution shift in target (e.g., prices inflating faster than model predicts)
- Covariate shift (same features, different relationships)

**Method:**
```python
# After each prediction batch, compute:
from scipy.stats import ks_2samp

# Compare recent vs baseline feature distributions
ks_stat, p_value = ks_2samp(recent_areas, baseline_areas)
if p_value < 0.05:
    print("⚠️  Data drift detected in 'area' feature")
    # Trigger manual review
```

### 4.2 Drift Response Protocol

| Drift Detected | Action |
|---|---|
| **Feature distribution shift** | Continue monitoring; retrain if MAPE ↑ |
| **Target distribution shift** | Retrain immediately (market movement) |
| **Covariate shift** | Manual review; may need feature engineering |
| **Systematic error bias** | Add bias correction term; retrain |

---

## 5. Model Versioning & Rollback

### 5.1 Versioning Scheme

```
saved_models/
├── lgbm_low.pkl (v2.6, 2026-07-22)
├── xgb_low.pkl  (v2.6, 2026-07-22)
├── cb_low.pkl   (v2.6, 2026-07-22)
├── backup/
│   ├── lgbm_low.pkl.old_20260715
│   ├── xgb_low.pkl.old_20260715
│   └── cb_low.pkl.old_20260715
└── archive/
    └── v2.5_full_backup/
```

### 5.2 Rollback Procedure

If deployed model shows unexpected degradation:

```bash
# 1. Check model age
ls -lah models/saved_models/*.pkl

# 2. Restore backup
cp models/saved_models/backup/lgbm_*.pkl.old_20260715 models/saved_models/lgbm_*.pkl

# 3. Verify
python -c "
from app.api import predict
result = predict(sample_property)
print(f'Prediction: {result}')  # Should match historical patterns
"

# 4. Commit rollback
git commit -m "fix: rollback to v2.6 backup (2026-07-15) — MAPE regression detected"
```

---

## 6. Incident Response

### 6.1 API Crashes

**Symptom:** API returns 500 errors on prediction requests

**Diagnosis:**
```bash
# Check logs
tail -f /var/log/app.log | grep "ERROR"

# Check model files exist
ls -lah models/saved_models/*.pkl

# Test import
python -c "import joblib; joblib.load('models/saved_models/lgbm_low.pkl')"
```

**Resolution:**
1. Restart API service
2. If model files corrupt, restore from backup
3. Retrain if necessary

### 6.2 Prediction Anomalies

**Symptom:** Predictions suddenly +20% higher/lower than baseline

**Diagnosis:**
```bash
# Compare recent predictions to baseline
python scripts/compare_predictions.py \
  --recent data/2026-07-22_predictions.csv \
  --baseline data/2026-07-15_predictions.csv
```

**Resolution:**
1. Check for data preprocessing bugs
2. Verify feature pipeline (78 features still present)
3. Check for input data corruption (e.g., VND/USD mixed)
4. Retrain on recent data

### 6.3 Model Performance Degradation

**Symptom:** MAPE slowly creeping up (13.10% → 13.5% → 14.0% over weeks)

**Cause:** Likely market drift (price movements outpacing model)

**Resolution:**
1. Retrain immediately (don't wait for weekly schedule)
2. Investigate market changes (new construction trends, etc.)
3. Consider feature engineering (temporal trends, economic indicators)

---

## 7. Scaling & Infrastructure

### 7.1 Current Architecture

- **Training:** Runs on local machine or cloud (Supabase fetch + local preprocessing)
- **Inference:** Flask API (`/app/api/app.py`)
- **Storage:** 9 model files (~500 MB total)
- **Latency:** <50ms per prediction

### 7.2 Scaling to 10,000 QPS

If demand exceeds current capacity:

**Option 1: Model Quantization**
```python
# Convert to ONNX (10-20x smaller, slightly faster)
import onnx
onnx_model = convert_to_onnx(model_lgbm)
onnx_model.save("models/lgbm_low.onnx")  # ~5 MB instead of 100 MB
```

**Option 2: Batch Inference**
```python
# Async endpoint for bulk predictions
@app.post("/predict-batch")
def predict_batch(properties: List[dict]):
    # Batch predictions on GPU (if available)
    return [predict(p) for p in properties]
```

**Option 3: Model Distillation**
```python
# Train small neural network to mimic ensemble
# 10x smaller, 5x faster, ~1% accuracy loss
```

### 7.3 Database Schema (Optional)

For logging all predictions:

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    property_id INT,
    predicted_price BIGINT,
    actual_price BIGINT NULL,
    confidence FLOAT,
    model_version VARCHAR(10),
    created_at TIMESTAMP,
    INDEX (property_id, created_at)
);
```

---

## 8. Feedback Loop & Continuous Improvement

### 8.1 Collecting Ground Truth

**Method:** If properties sell, log actual price vs prediction

```python
# After property sells, update predictions table:
UPDATE predictions SET actual_price = 5_000_000_000 WHERE property_id = 12345;
```

**Frequency:** Monthly reconciliation against Supabase sales data

### 8.2 Quarterly Review

Every 3 months (e.g., 2026-10-22):

1. **Performance Trends**
   - Plot MAPE over time (should be stable around 13.10%)
   - Identify systematic biases (e.g., always overpredict luxury?)
   
2. **Feature Drift Analysis**
   - Which features changed most in market data?
   - Any new construction trends?
   
3. **Roadmap Planning**
   - Can we reach <10% MAPE? (requires data expansion)
   - Should we add temporal features?
   - Collect high-tier properties?

### 8.3 Path to v2.7 (Next Major Release)

If quarterly review shows opportunity:

**v2.7 Candidates:**
- [ ] Temporal features (price trend, market momentum)
- [ ] High-tier data collection (500+ more luxury properties)
- [ ] Quantile regression (separate models for percentiles)
- [ ] External data (macro indicators, satellite imagery)

---

## 9. Security & Access Control

### 9.1 Model File Permissions

```bash
# Restrict write access to model files
chmod 444 models/saved_models/*.pkl

# Only CI/CD or authorized users can update
chmod 755 models/saved_models/
```

### 9.2 API Authentication

Current: No authentication (public API)

**Recommended for production:** Add API key validation

```python
@app.post("/predict")
def predict(request: PredictRequest, api_key: str = Header(None)):
    if api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return predict_property(request.property_data)
```

### 9.3 Data Privacy

- Do not log sensitive data (owner names, addresses) in predictions table
- Only log anonymized features (area, beds, age, price)
- Comply with Vietnam PDPA (if applicable)

---

## 10. Cost Analysis & ROI

### 10.1 Current Costs (Monthly)

| Component | Cost |
|-----------|------|
| Supabase DB (free tier) | $0 |
| Model inference (self-hosted) | ~$50 (compute) |
| W&B experiment tracking (free tier) | $0 |
| **Total** | **~$50/month** |

### 10.2 Revenue Potential

If monetizing via API calls:

- **100K predictions/month** @ $0.001/call = $100 revenue
- **Net:** -$50 (cost) but builds customer base
- **Scaling:** 1M predictions/month = $1000 revenue, sustainable

---

## Appendix: Useful Commands

### Deploy latest model
```bash
cd /app && docker build -t realestate-api . && docker run -p 5000:5000 realestate-api
```

### Check model performance
```bash
python models/scripts/train_production.py --dataset production --data-source supabase
```

### View W&B dashboard
```bash
# Open in browser
https://wandb.ai/real-estate-valuation/v2.6-production
```

### Rollback to backup
```bash
cp models/saved_models/backup/lgbm_*.pkl.old_20260715 models/saved_models/lgbm_*.pkl
```

---

**Document prepared:** 2026-07-22  
**Status:** Ready for Implementation  
**Owner:** ML Engineering Team  
**Review:** Quarterly
