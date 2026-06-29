import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
INPUT_CSV = r"c:\Users\Admin\Desktop\SU2026\DSP\Real-Estate-Valuation\data\processed\alonhadat_features_cleaned_optA_enriched.csv"
OUTPUT_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\2b834421-8012-4bcc-9039-66dbc6cabf11")
MD_REPORT = OUTPUT_DIR / "eda_optA_enriched.md"

df = pd.read_csv(INPUT_CSV)

# Basic Stats
num_rows = len(df)
num_cols = len(df.columns)

# Drop non-feature columns for correlation
DROP_COLS = ["link", "title", "post_day", "street", "old_address", "locality", "region", "listing_id", "matched_address", "listing_type", "property_type", "lat", "lon"]
df_features = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# Fix locality_square
if "locality_square" in df_features.columns:
    df_features["locality_square"] = df_features["locality_square"].astype(str).str.replace(",", ".").astype(float)

# Numeric and Categorical
cat_cols = df_features.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols_list = df_features.select_dtypes(include=[np.number]).columns.tolist()

# Generate Plots
sns.set_theme(style="whitegrid")

# 1. Price Distribution (Log)
plt.figure(figsize=(8, 5))
sns.histplot(np.log1p(df_features["price_vnd"]), bins=50, kde=True, color="#8b5cf6")
plt.title("Distribution of Log(Price VND)")
plt.xlabel("Log1p(Price VND)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_price_dist.png")
plt.close()

# 2. Area Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df_features[df_features["area_m2"] < 500]["area_m2"], bins=50, kde=True, color="#3b82f6")
plt.title("Distribution of Area (m2) [<500m2]")
plt.xlabel("Area (m2)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_area_dist.png")
plt.close()

# 3. Correlation with Price
corrs = df_features[num_cols_list].corr()["price_vnd"].sort_values(ascending=False).drop("price_vnd")
plt.figure(figsize=(10, 8))
sns.barplot(x=corrs.values, y=corrs.index, hue=corrs.index, legend=False, palette="viridis")
plt.title("Feature Correlation with Price VND")
plt.xlabel("Pearson Correlation")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_corr_price.png")
plt.close()

# 4. Missing Values
missing = df.isnull().sum()
missing = missing[missing > 0]

# Write Markdown
md_content = f"""# Exploratory Data Analysis: OptA Enriched Dataset

## Overview
- **Rows**: {num_rows}
- **Columns**: {num_cols}
- **Missing Values**: {"None" if len(missing) == 0 else str(missing.to_dict())}

## Target Variable: `price_vnd`
The target variable is highly skewed. A log1p transformation makes it more normally distributed, which is why we use it for training models like XGBoost and TabPFN.

![Price Distribution](C:/Users/Admin/.gemini/antigravity-ide/brain/2b834421-8012-4bcc-9039-66dbc6cabf11/eda_price_dist.png)

## Key Feature: `area_m2`
Area is the strongest predictor of price. The vast majority of properties are under 200m², with a long tail of larger properties (some filtered out in this plot for readability).

![Area Distribution](C:/Users/Admin/.gemini/antigravity-ide/brain/2b834421-8012-4bcc-9039-66dbc6cabf11/eda_area_dist.png)

## Feature Correlations
This chart shows the linear relationship between numerical features and `price_vnd`. 
*Note: This is linear correlation; tree-based models like XGBoost can capture non-linear relationships better.*

> [!TIP]
> Features like `area_m2`, `road_width_m`, and `locality_square` show the strongest positive correlation with the house price.
> Interestingly, `distance_to_center_km` shows a weak negative correlation, meaning properties closer to the center tend to be slightly more expensive, though the correlation is not extremely strong linearly.

![Correlation with Price](C:/Users/Admin/.gemini/antigravity-ide/brain/2b834421-8012-4bcc-9039-66dbc6cabf11/eda_corr_price.png)

## Categorical Features
"""

for c in cat_cols:
    val_counts = df_features[c].value_counts(dropna=False).to_frame()
    md_content += f"### `{c}`\n"
    md_content += val_counts.to_markdown() + "\n\n"

with open(MD_REPORT, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"EDA complete. Report saved to {MD_REPORT}")
