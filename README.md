# 🏠 House Price Prediction - MAPE 3.36%

**Target:** MAPE < 10% ✅  
**Achieved:** MAPE 3.36% (89% improvement from 31.54%)

## 🚀 Quick Start

```bash
# Option 1: Interactive Notebook (Recommended)
cd models && jupyter notebook train.ipynb

# Option 2: Python Scripts
cd models
python train.py      # Train (2 min)
python predict.py    # Predict (all 8,636 properties)

# Option 3: Web UI
streamlit run app/app.py          # Property valuation
streamlit run app/dashboard.py    # BI Dashboard
```

## 📁 Structure

```
HousePricePrediction/
├── README.md                    ← You are here
│
├── models/                      ← ALL MODEL CODE HERE
│   ├── train.ipynb             # 📓 Main: Complete pipeline (7 parts)
│   ├── train.py                # Script: Training
│   ├── predict.py              # Script: Predictions  
│   ├── README.md               # Model documentation
│   ├── saved_models/           # Trained models (ready to use)
│   └── data/                   # Datasets & predictions
│
├── app/                         ← Web interfaces
│   ├── app.py                  # Interactive web UI
│   ├── dashboard.py            # BI Dashboard
│   ├── api_simple.py           # REST API
│   └── README.md               # App documentation
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| **MAPE** | 3.36% |
| **MAE** | 0.66B VND |
| **R²** | 0.9939 |

By segment:
- **0-5B:** MAPE 6.6% (budget)
- **5-20B:** MAPE 2.8% (BEST - mass market)
- **20-100B:** MAPE 3.2% (luxury)

## 🛠️ What to Do

### For Model Work
See `models/README.md`
- Train: `cd models && jupyter notebook train.ipynb`
- Or: `python train.py && python predict.py`

### For Web/API
See `app/README.md`
- Web UI: `streamlit run app/app.py`
- Dashboard: `streamlit run app/dashboard.py`
- API: `python app/api_simple.py`

---

**Status:** ✅ Production Ready  
**Training Time:** ~2 minutes  
**Prediction Time:** ~30ms per property
