# 📓 Notebooks - Analysis & Model Training

## Structure

```
notebooks/
├── 01_eda/                         # Exploratory Data Analysis
│   ├── 01_eda_complete.ipynb      # Full EDA: distributions, correlations, heatmaps
│   └── output/                     # Generated visualizations & summaries
│
├── 02_model_training/              # Model Training & Evaluation
│   ├── 02_model_training.ipynb    # LightGBM, XGBoost, CatBoost training
│   └── results/                    # Model evaluation results
│
└── 03_experiments/                 # (Optional) Ad-hoc experiments
    └── README.md                   # Document any experimental notebooks
```

## Quick Start

### 1. Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_eda/01_eda_complete.ipynb
```
**What it covers:**
- 12,814 properties loaded from Supabase
- Price distribution analysis (2-50B VND range)
- Feature correlations with price
- Geographic heatmap visualization
- POI features analysis
- Data quality assessment

**Key findings:**
- Median price: 14.5B VND
- Top predictors: area (0.48), width (0.43), road width (0.34)
- Missing data mainly in metro distances & dimensions

### 2. Model Training
```bash
jupyter notebook notebooks/02_model_training/02_model_training.ipynb
```
**What it covers:**
- Data preprocessing (10.4k clean records)
- Feature engineering (166 features)
- Train 3 algorithms: LightGBM, XGBoost, CatBoost
- Model evaluation & comparison
- Feature importance analysis

**Results:**
- Best: XGBoost (18.01% MAPE)
- R²: 0.8663
- Production model saved

## Notes

- All notebooks use Supabase for data loading
- Features are cached locally for faster retraining
- Models saved to `models/production/` and `models/archive/`
- Visualizations stored in `01_eda/output/`

## Dependencies

```
pandas, numpy, matplotlib, seaborn, scikit-learn
lightgbm, xgboost, catboost
jupyter, plotly
```
