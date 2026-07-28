"""
Compare distributions: Raw Data (Supabase) vs Processed Data
Visualization: 2 large figures side-by-side for all key features
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from models.scripts.shared.preprocessing import setup_matplotlib, get_data_paths, load_raw_data, load_processed_data

plt = setup_matplotlib()

# Get correct project root (parent of notebooks folder)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
VIZ_DIR = DATA_DIR / 'visualizations'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[DEBUG] VIZ_DIR: {VIZ_DIR}")
print("[Loading Data]")
df_raw = load_raw_data()
df_processed = load_processed_data()

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

            null_count = df_raw[feat].isna().sum()
            null_pct = (null_count / len(df_raw)) * 100
            xlabel = feat if feat != 'price_vnd' else 'Price (Billion VND)'
            axes1[idx].set_xlabel(xlabel, fontweight='bold', fontsize=11)
            axes1[idx].set_ylabel('Frequency', fontweight='bold')
            axes1[idx].set_title(f'{feat}\nn={len(data):,} | σ={data.std():.2f} | null={null_count:,} ({null_pct:.1f}%)',
                                fontsize=10, fontweight='bold')
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
        null_count_sqm = (df_raw['price_vnd'].isna().sum() + df_raw['area_m2'].isna().sum())
        null_pct_sqm = (null_count_sqm / len(df_raw)) * 100
        axes1[7].set_xlabel('Price per m² (Million VND)', fontweight='bold', fontsize=11)
        axes1[7].set_ylabel('Frequency', fontweight='bold')
        axes1[7].set_title(f'price_per_sqm\nn={len(price_per_sqm):,} | σ={price_per_sqm.std():.2f} | null={null_count_sqm:,} ({null_pct_sqm:.1f}%)',
                          fontsize=10, fontweight='bold')
        axes1[7].legend(loc='upper right', fontsize=9)
        axes1[7].grid(alpha=0.3)

    # Add summary in 9th subplot
    axes1[8].axis('off')
    summary_text = "DATA QUALITY SUMMARY (Raw)\n\n"
    for idx, feat in enumerate(key_features):
        if feat in df_raw.columns:
            null_count = df_raw[feat].isna().sum()
            null_pct = (null_count / len(df_raw)) * 100
            summary_text += f"{feat}: {null_count:,} nulls ({null_pct:.1f}%)\n"

    axes1[8].text(0.1, 0.9, summary_text, transform=axes1[8].transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

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

        null_count = df_processed[feat].isna().sum()
        null_pct = (null_count / len(df_processed)) * 100
        xlabel = feat if feat != 'price_vnd' else 'Price (Billion VND)'
        axes2[idx].set_xlabel(xlabel, fontweight='bold', fontsize=11)
        axes2[idx].set_ylabel('Frequency', fontweight='bold')
        axes2[idx].set_title(f'{feat}\nn={len(data):,} | σ={data.std():.2f} | null={null_count:,} ({null_pct:.1f}%)',
                            fontsize=10, fontweight='bold')
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
    null_count_sqm = (df_processed['price_vnd'].isna().sum() + df_processed['area_m2'].isna().sum())
    null_pct_sqm = (null_count_sqm / len(df_processed)) * 100
    axes2[7].set_xlabel('Price per m² (Million VND)', fontweight='bold', fontsize=11)
    axes2[7].set_ylabel('Frequency', fontweight='bold')
    axes2[7].set_title(f'price_per_sqm\nn={len(price_per_sqm):,} | σ={price_per_sqm.std():.2f} | null={null_count_sqm:,} ({null_pct_sqm:.1f}%)',
                      fontsize=10, fontweight='bold')
    axes2[7].legend(loc='upper right', fontsize=9)
    axes2[7].grid(alpha=0.3)

# Add summary in 9th subplot
axes2[8].axis('off')
summary_text = "DATA QUALITY SUMMARY (Processed)\n\n"
for idx, feat in enumerate(key_features):
    if feat in df_processed.columns:
        null_count = df_processed[feat].isna().sum()
        null_pct = (null_count / len(df_processed)) * 100
        summary_text += f"{feat}: {null_count:,} nulls ({null_pct:.1f}%)\n"

axes2[8].text(0.1, 0.9, summary_text, transform=axes2[8].transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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
