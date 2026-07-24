# 📓 Analysis & Experiments Notebooks

Organized Jupyter notebooks for exploratory data analysis, feature engineering, model training, and explainability.

---

## 🗂️ Directory Structure

```
notebooks/
├── README.md                                    ← This file
├── 01_exploratory_data_analysis/                ← Initial data exploration
│   ├── README.md
│   └── 01_dataset_overview.ipynb
│
├── 02_feature_engineering_analysis/             ← Feature exploration & impact
│   ├── README.md
│   └── 01_feature_analysis.ipynb
│
├── 03_model_training_experiments/               ← Model training & tuning
│   ├── README.md
│   └── 01_ensemble_training.ipynb
│
├── 04_explainability_and_xai/                  ← Model interpretability
│   ├── README.md
│   └── 01_feature_importance.ipynb
│
└── 05_model_validation_and_maintenance/         ← Drift detection & monitoring
    ├── README.md
    └── 01_drift_detection.ipynb
```

---

## 📚 Notebook Descriptions

| # | Name | Purpose | Time | Key Output |
|---|------|---------|------|-----------|
| **01** | Exploratory Data Analysis | Understand data distribution, quality, patterns | 15-20min | Data quality report, visualizations |
| **02** | Feature Engineering Analysis | Validate engineered features, correlations | 20-30min | Feature importance rankings |
| **03** | Model Training Experiments | Develop & compare ML models | 30-60min | Best model selection rationale |
| **04** | Explainability & XAI | Understand model decisions | 20-30min | SHAP plots, feature impact |
| **05** | Validation & Maintenance | Monitor model drift & performance | 15-20min | Drift alerts, retraining triggers |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install jupyter jupyterlab pandas numpy scikit-learn matplotlib seaborn plotly shap
```

### 2. Start Jupyter

```bash
jupyter lab
# or
jupyter notebook
```

### 3. Run in Order

Start with **01** and proceed sequentially. Each builds on the previous:

```
01 (understand data)
  ↓
02 (validate features)
  ↓
03 (build models)
  ↓
04 (interpret models)
  ↓
05 (monitor production)
```

---

## 🔄 Data Flow

```
Supabase Raw_Features (10,432 properties)
    ↓
01_exploratory_data_analysis
    ↓ Data quality assessment
02_feature_engineering_analysis
    ↓ Feature validation
03_model_training_experiments
    ↓ Model development
04_explainability_and_xai
    ↓ Model interpretability
05_model_validation_and_maintenance
    ↓ Production monitoring
Ready for Deployment ✅
```

---

## 📊 Key Metrics by Notebook

| Notebook | Metric | Target | Status |
|----------|--------|--------|--------|
| 01 | Data completeness | >99% | ✅ |
| 01 | Duplicates | 0 | ✅ |
| 02 | Feature count | 78 | ✅ |
| 02 | Correlation |  <0.9 | ✅ |
| 03 | MAPE | <13.5% | ✅ |
| 03 | R² | >0.92 | ✅ |
| 04 | Top features | Identified | ✅ |
| 05 | Model drift | <2% | ✅ |

---

## 📁 Data Sources & Outputs

### Inputs
- **Supabase:** `Raw_Features` table
- **Local cache:** `data/cache/localities.csv`
- **Reference data:** `data/external/density_data.csv`

### Outputs
- **Processed data:** `data/processed/` (CSVs)
- **Plots:** `notebooks/outputs/` (PNGs, HTMLs)
- **Reports:** `notebooks/reports/` (markdown summaries)
- **Models:** `models/archive/` (experimental versions)

---

## 💡 Tips & Best Practices

### Setup Notebook

```python
# Add at top of each notebook
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
%matplotlib inline

# Load data
from app.core.models import get_models
models, meta, _, geo = get_models()
```

### Plotting

```python
# Use consistent styling
plt.figure(figsize=(12, 6))
plt.title("Analysis Title", fontsize=14, fontweight='bold')
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.tight_layout()
plt.savefig('outputs/plot_name.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Save Results

```python
# Export important findings
results_df.to_csv('data/processed/analysis_results.csv', index=False)

# Add markdown summary
summary = f"""
# Analysis Summary
- Finding 1: {finding1}
- Finding 2: {finding2}
"""
print(summary)
```

---

## 🐛 Troubleshooting

### "Module not found"
```python
import sys; sys.path.insert(0, '..')
```

### "Out of memory"
```python
# Sample data
df = pd.read_csv('data.csv').sample(frac=0.1)
```

### "Plots not showing"
```python
%matplotlib inline  # Add to cell
plt.show()
```

### "Supabase connection failed"
- Check `.env` has correct credentials
- Run: `python -c "from pipeline.supabase_handler import test_connection; test_connection()"`

---

## 📖 Standard Notebook Structure

All notebooks follow this format:

1. **Title & Description** (markdown)
2. **Setup & Imports** (code)
3. **Load Data** (code)
4. **Exploratory Analysis** (code + markdown)
5. **Visualizations** (code + output)
6. **Statistical Tests** (code + results)
7. **Conclusions** (markdown summary)
8. **Save Outputs** (code)

---

## 🔗 Related Documentation

- [Main README](../README.md) - Project overview
- [DATA.md](../DATA.md) - Data pipeline
- [MODELS.md](../MODELS.md) - Model architecture
- [pipeline/README.md](../pipeline/README.md) - ETL process

---

## 👥 Contributing

When adding notebooks:

1. **Create directory** with number prefix: `XX_descriptive_name/`
2. **Add notebook:** `01_main_analysis.ipynb`
3. **Add README:** `XX_descriptive_name/README.md`
4. **Update this file** with description
5. **Save outputs:** `notebooks/outputs/` or `data/processed/`

---

**Last Updated:** 2026-07-23
**Team:** DSP391m Capstone Project
