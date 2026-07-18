"""
Exploratory Data Analysis (EDA)
Real Estate Valuation Dataset

Load data from Supabase and perform comprehensive analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

# Add the repo root (the folder that contains pipeline/)
for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    if (parent / 'pipeline').exists():
        sys.path.insert(0, str(parent))
        break

from pipeline.supabase_handler import fetch_csv_from_supabase

# Setup plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70 + "\n")

# ============================================================================
# 1. LOAD DATA FROM SUPABASE
# ============================================================================

print("[1/6] Loading data from Supabase...\n")
df = fetch_csv_from_supabase()

if len(df) == 0:
    print("❌ No data found in Supabase!")
    sys.exit(1)

print(f"✅ Loaded {len(df)} records\n")

# ============================================================================
# 2. DATA OVERVIEW
# ============================================================================

print("[2/6] Data Overview\n")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}\n")
print(f"Data Types:\n{df.dtypes}\n")
print(f"Missing Values:\n{df.isnull().sum()}\n")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

# ============================================================================
# 3. PRICE ANALYSIS
# ============================================================================

print("[3/6] Price Analysis\n")

if 'price' in df.columns:
    price_stats = df['price'].describe()
    print(price_stats)
    print(f"\nPrice Range: {df['price'].min():,.0f} - {df['price'].max():,.0f} VND")
    print(f"Median: {df['price'].median():,.0f} VND")

    # Create price distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram
    axes[0, 0].hist(df['price'], bins=50, edgecolor='black')
    axes[0, 0].set_title('Price Distribution (Histogram)')
    axes[0, 0].set_xlabel('Price (VND)')
    axes[0, 0].set_ylabel('Frequency')

    # Log scale
    axes[0, 1].hist(np.log10(df['price']), bins=50, edgecolor='black', color='orange')
    axes[0, 1].set_title('Price Distribution (Log Scale)')
    axes[0, 1].set_xlabel('Log10(Price)')
    axes[0, 1].set_ylabel('Frequency')

    # Box plot
    axes[1, 0].boxplot(df['price'], vert=True)
    axes[1, 0].set_title('Price Box Plot')
    axes[1, 0].set_ylabel('Price (VND)')

    # CDF
    sorted_price = np.sort(df['price'])
    cdf = np.arange(1, len(sorted_price) + 1) / len(sorted_price)
    axes[1, 1].plot(sorted_price, cdf, linewidth=2)
    axes[1, 1].set_title('Cumulative Distribution Function')
    axes[1, 1].set_xlabel('Price (VND)')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('price_distribution.png', dpi=100, bbox_inches='tight')
    print("\n✅ Saved: price_distribution.png\n")
    plt.show()

# ============================================================================
# 4. FEATURE ANALYSIS
# ============================================================================

print("[4/6] Feature Analysis\n")

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric Columns: {numeric_cols}\n")

# Feature statistics
print("Feature Statistics:")
print(df[numeric_cols].describe().round(2))

# Correlation with price (if exists)
if 'price' in df.columns:
    print("\n\nCorrelation with Price:")
    correlations = df[numeric_cols].corr()['price'].sort_values(ascending=False)
    print(correlations)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Select top features
    top_features = correlations.abs().nlargest(15).index.tolist()
    sns.heatmap(df[top_features].corr(), annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix (Top 15)')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=100, bbox_inches='tight')
    print("\n✅ Saved: correlation_heatmap.png\n")
    plt.show()

# ============================================================================
# 5. GEOGRAPHIC ANALYSIS
# ============================================================================

print("[5/6] Geographic Analysis\n")

if 'lat' in df.columns and 'lon' in df.columns and 'price' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Price by latitude
    scatter1 = axes[0].scatter(df['lat'], df['price'], alpha=0.5, c=df['price'],
                               cmap='viridis', s=20)
    axes[0].set_xlabel('Latitude')
    axes[0].set_ylabel('Price (VND)')
    axes[0].set_title('Price by Latitude')
    plt.colorbar(scatter1, ax=axes[0])

    # Price by longitude
    scatter2 = axes[1].scatter(df['lon'], df['price'], alpha=0.5, c=df['price'],
                               cmap='viridis', s=20)
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Price (VND)')
    axes[1].set_title('Price by Longitude')
    plt.colorbar(scatter2, ax=axes[1])

    plt.tight_layout()
    plt.savefig('geographic_analysis.png', dpi=100, bbox_inches='tight')
    print("✅ Saved: geographic_analysis.png\n")
    plt.show()

    # 2D scatter (lat vs lon)
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(df['lon'], df['lat'], c=df['price'], cmap='viridis',
                         s=30, alpha=0.6)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Price Heatmap (Geographic Distribution)')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Price (VND)')
    plt.tight_layout()
    plt.savefig('price_heatmap.png', dpi=100, bbox_inches='tight')
    print("✅ Saved: price_heatmap.png\n")
    plt.show()

# ============================================================================
# 6. KEY INSIGHTS
# ============================================================================

print("[6/6] Key Insights\n")

insights = {
    "Total Records": len(df),
    "Price Range": f"{df['price'].min():,.0f} - {df['price'].max():,.0f} VND",
    "Mean Price": f"{df['price'].mean():,.0f} VND",
    "Median Price": f"{df['price'].median():,.0f} VND",
    "Std Dev": f"{df['price'].std():,.0f} VND",
    "Missing Values": df.isnull().sum().sum(),
}

for key, value in insights.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 70)
print("✅ EDA COMPLETE!")
print("=" * 70)
print("\nVisualization files saved:")
print("  - price_distribution.png")
print("  - correlation_heatmap.png")
print("  - geographic_analysis.png")
print("  - price_heatmap.png")
print("\nNext: Use these insights for model building and paper writing!")
