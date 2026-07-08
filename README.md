# 🏠 House Price Prediction - MAPE 3.36%

**Target:** MAPE < 10% ✅  
**Achieved:** MAPE 3.36% (89% improvement from 31.54%)

## 🚀 Quick Start

```bash
# 1. Train (2 min)
cd models && python train.py

# 2. Predict
python predict.py

# 3. Choose interface
streamlit run ../app/app.py          # Web UI
streamlit run ../app/dashboard.py    # BI Dashboard
python ../app/api_simple.py          # REST API
```

## 📁 Structure

```
├── models/
│   ├── train.py          ← Training script
│   ├── predict.py        ← Inference script
│   ├── README.md         ← Model documentation
│   ├── saved_models/     ← Trained models
│   └── data/             ← Datasets & predictions
│
├── app/
│   ├── app.py            ← Web UI
│   ├── dashboard.py      ← BI Dashboard
│   ├── api_simple.py     ← REST API
│   └── README.md         ← App documentation
│
└── README.md             ← This file
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| **MAPE** | 3.36% |
| **MAE** | 0.66B VND |
| **R²** | 0.9939 |

By segment:
- **0-5B:** MAPE 6.6%
- **5-20B:** MAPE 2.8% (BEST)
- **20-100B:** MAPE 3.2%

## 🛠️ What to Do

**For Model Training:**  
See `models/README.md`

**For Web/API:**  
See `app/README.md`

---

**Status:** ✅ Production Ready  
**Training Time:** ~2 minutes  
**Prediction Time:** ~30ms per property
