"""
Compare distributions: Raw Data (Supabase) vs Processed Data
Visualization: 2 large figures side-by-side for all key features
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
VIZ_DIR = DATA_DIR / 'visualizations'
VIZ_DIR.mkdir(exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['font.size'] = 10

# Load data
print("[Loading Data]")
try:
    from pipeline.supabase_handler import fetch_csv_from_supabase
    df_raw = fetch_csv_from_supabase("Raw_Features")
    if len(df_raw) == 0:
        raise ValueError("No records from Supabase")
    print(f"  ✓ Raw (Supabase): {df_raw.shape}")
except Exception as e:
    print(f"  ⚠ {e}")
    df_raw = None

df_processed = pd.read_csv(DATA_DIR / 'processed/model_training_data.csv')
print(f"  ✓ Processed: {df_processed.shape}")

# Features to analyze
key_features = ['price_vnd', 'area_m2', 'width_m', 'length_m', 'num_floors', 'num_bedrooms', 'road_width_m']
colors = ['#9B59B6', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

# ============================================================================
# FIGURE 1: RAW DATA (from Supabase)
# ============================================================================
if df_raw is not None:
    fig1, axes1 = plt.subplots(3, 3, figsize=(18, 14))
    axes1 = axes1.flatten()

    fig1.suptitle('RAW DATA (Supabase) - Feature Distributions',
                  fontsize=16, fontweight='bold', y=0.995)

    for idx, (feat, color) in enumerate(zip(key_features, colors)):
        if feat in df_raw.columns:
            if feat == 'price_vnd':
                data = df_raw[feat].dropna() / 1e9  # Convert to billions
                label_suffix = 'B'
            else:
                data = df_raw[feat].dropna()
                label_suffix = ''

            axes1[idx].hist(data, bins=50, color=color, edgecolor='black',
                           alpha=0.7, label='Histogram')
            axes1[idx].axvline(data.median(), color='red', linestyle='--',
                              linewidth=2.5, label=f'Median: {data.median():.2f}{label_suffix}')
            axes1[idx].axvline(data.mean(), color='green', linestyle='--',
                              linewidth=2.5, label=f'Mean: {data.mean():.2f}{label_suffix}')

            xlabel = feat if feat != 'price_vnd' else 'Price (Billion VND)'
            axes1[idx].set_xlabel(xlabel, fontweight='bold', fontsize=11)
            axes1[idx].set_ylabel('Frequency', fontweight='bold')
            axes1[idx].set_title(f'{feat}\nn={len(data):,} | σ={data.std():.2f}',
                                fontsize=11, fontweight='bold')
            axes1[idx].legend(loc='upper right', fontsize=9)
            axes1[idx].grid(alpha=0.3)
        else:
            axes1[idx].text(0.5, 0.5, f'{feat}\nNOT AVAILABLE',
                           ha='center', va='center', fontsize=12,
                           transform=axes1[idx].transAxes, color='red')
            axes1[idx].set_title(f'{feat}', fontsize=11, fontweight='bold')

    # Add price_per_sqm to 8th subplot
    if 'price_vnd' in df_raw.columns and 'area_m2' in df_raw.columns:
        price_per_sqm = (df_raw['price_vnd'] / (df_raw['area_m2'] + 1) / 1e6).dropna()
        axes1[7].hist(price_per_sqm, bins=50, color='#E74C3C', edgecolor='black', alpha=0.7)
        axes1[7].axvline(price_per_sqm.median(), color='red', linestyle='--', linewidth=2.5,
                        label=f'Median: {price_per_sqm.median():.2f}M')
        axes1[7].axvline(price_per_sqm.mean(), color='green', linestyle='--', linewidth=2.5,
                        label=f'Mean: {price_per_sqm.mean():.2f}M')
        axes1[7].set_xlabel('Price per m² (Million VND)', fontweight='bold', fontsize=11)
        axes1[7].set_ylabel('Frequency', fontweight='bold')
        axes1[7].set_title(f'price_per_sqm\nn={len(price_per_sqm):,} | σ={price_per_sqm.std():.2f}',
                          fontsize=11, fontweight='bold')
        axes1[7].legend(loc='upper right', fontsize=9)
        axes1[7].grid(alpha=0.3)

    # Hide the 9th subplot (index 8)
    axes1[8].axis('off')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / '01_raw_distributions.png',
                dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 01_raw_distributions.png")
    plt.close()

# ============================================================================
# FIGURE 2: PROCESSED DATA
# ============================================================================
fig2, axes2 = plt.subplots(3, 3, figsize=(18, 14))
axes2 = axes2.flatten()

fig2.suptitle('PROCESSED DATA (model_training_data.csv) - Feature Distributions',
              fontsize=16, fontweight='bold', y=0.995)

for idx, (feat, color) in enumerate(zip(key_features, colors)):
    if feat in df_processed.columns:
        if feat == 'price_vnd':
            data = df_processed[feat].dropna() / 1e9  # Convert to billions
            label_suffix = 'B'
        else:
            data = df_processed[feat].dropna()
            label_suffix = ''

        axes2[idx].hist(data, bins=50, color=color, edgecolor='black',
                       alpha=0.7, label='Histogram')
        axes2[idx].axvline(data.median(), color='red', linestyle='--',
                          linewidth=2.5, label=f'Median: {data.median():.2f}{label_suffix}')
        axes2[idx].axvline(data.mean(), color='green', linestyle='--',
                          linewidth=2.5, label=f'Mean: {data.mean():.2f}{label_suffix}')

        xlabel = feat if feat != 'price_vnd' else 'Price (Billion VND)'
        axes2[idx].set_xlabel(xlabel, fontweight='bold', fontsize=11)
        axes2[idx].set_ylabel('Frequency', fontweight='bold')
        axes2[idx].set_title(f'{feat}\nn={len(data):,} | σ={data.std():.2f}',
                            fontsize=11, fontweight='bold')
        axes2[idx].legend(loc='upper right', fontsize=9)
        axes2[idx].grid(alpha=0.3)
    else:
        axes2[idx].text(0.5, 0.5, f'{feat}\nNOT AVAILABLE',
                       ha='center', va='center', fontsize=12,
                       transform=axes2[idx].transAxes, color='red')
        axes2[idx].set_title(f'{feat}', fontsize=11, fontweight='bold')

# Add price_per_sqm to 8th subplot
if 'price_vnd' in df_processed.columns and 'area_m2' in df_processed.columns:
    price_per_sqm = (df_processed['price_vnd'] / (df_processed['area_m2'] + 1) / 1e6).dropna()
    axes2[7].hist(price_per_sqm, bins=50, color='#E74C3C', edgecolor='black', alpha=0.7)
    axes2[7].axvline(price_per_sqm.median(), color='red', linestyle='--', linewidth=2.5,
                    label=f'Median: {price_per_sqm.median():.2f}M')
    axes2[7].axvline(price_per_sqm.mean(), color='green', linestyle='--', linewidth=2.5,
                    label=f'Mean: {price_per_sqm.mean():.2f}M')
    axes2[7].set_xlabel('Price per m² (Million VND)', fontweight='bold', fontsize=11)
    axes2[7].set_ylabel('Frequency', fontweight='bold')
    axes2[7].set_title(f'price_per_sqm\nn={len(price_per_sqm):,} | σ={price_per_sqm.std():.2f}',
                      fontsize=11, fontweight='bold')
    axes2[7].legend(loc='upper right', fontsize=9)
    axes2[7].grid(alpha=0.3)

# Hide the 9th subplot (index 8)
axes2[8].axis('off')

plt.tight_layout()
plt.savefig(VIZ_DIR / '02_processed_distributions.png',
            dpi=300, bbox_inches='tight')
print("✓ Saved: 02_processed_distributions.png")
plt.close()

# ============================================================================
# COMPARISON TABLE
# ============================================================================
if df_raw is not None:
    print("\n" + "="*100)
    print("COMPARISON: RAW vs PROCESSED DATA")
    print("="*100)

    comparison_data = []
    for feat in key_features:
        if feat in df_raw.columns and feat in df_processed.columns:
            raw_data = df_raw[feat].dropna()
            proc_data = df_processed[feat].dropna()

            # Special handling for price_vnd (convert to billions)
            if feat == 'price_vnd':
                raw_mean = f"{raw_data.mean()/1e9:.2f}B"
                raw_median = f"{raw_data.median()/1e9:.2f}B"
                proc_mean = f"{proc_data.mean()/1e9:.2f}B"
                proc_median = f"{proc_data.median()/1e9:.2f}B"
            else:
                raw_mean = f"{raw_data.mean():.2f}"
                raw_median = f"{raw_data.median():.2f}"
                proc_mean = f"{proc_data.mean():.2f}"
                proc_median = f"{proc_data.median():.2f}"

            comparison_data.append({
                'Feature': feat,
                'Raw_Count': len(raw_data),
                'Raw_Mean': raw_mean,
                'Raw_Median': raw_median,
                'Proc_Count': len(proc_data),
                'Proc_Mean': proc_mean,
                'Proc_Median': proc_median,
            })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print("="*100)
