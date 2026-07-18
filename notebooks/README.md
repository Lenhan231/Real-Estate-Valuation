# 📓 Notebooks - Analysis & Model Training

## Structure

```
notebooks/
├── 01_eda/                         # Exploratory Data Analysis
│   ├── 01_eda_complete.ipynb       # EDA on Supabase Raw_Features + cleaned data comparison
│   └── output/                     # Generated visualizations & summaries

└── README.md
```

## Quick Start

### 1. Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_eda/01_eda_complete.ipynb
```
**What it covers:**
- Raw_Features loaded from Supabase
- Price distribution analysis and outlier checks
- Missing values, duplicates, and schema sanity checks
- Geographic heatmap visualization
- Data quality comparison before and after cleaning

**Key findings:**
- Median price: 14.5B VND
- Top predictors: area (0.48), width (0.43), road width (0.34)
- Missing data mainly in metro distances & dimensions

## Notes

- EDA notebook should read Supabase `Raw_Features` first
- Training lives in `models/scripts/`
- Models saved to `models/production/` and `models/archive/`
- Visualizations stored in `01_eda/output/`

## Dependencies

```
pandas, numpy, matplotlib, seaborn, scikit-learn
lightgbm, xgboost, catboost
jupyter, plotly
```
