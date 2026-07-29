"""
Figure 3: Data Distribution Across Price Segments (Low/Mid/High Tiers)
Analyze how features vary across different price segments
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from models.scripts.shared.preprocessing import setup_matplotlib, load_processed_data

plt = setup_matplotlib()

# Save outputs to folder next to this script
SCRIPT_DIR = Path(__file__).parent
VIZ_DIR = SCRIPT_DIR / 'outputs'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print(f"[DEBUG] VIZ_DIR: {VIZ_DIR}")
print(f"[DEBUG] VIZ_DIR exists: {VIZ_DIR.exists()}")
print("[Loading Data]")
df = load_processed_data()
print(f"  {df.shape[0]:,} records")

# Define price segments using custom bins
# Low: 0-5B, Mid: 5-20B, High: >20B
df['price_tier'] = pd.cut(df['price_vnd'],
                          bins=[0, 5e9, 20e9, np.inf],
                          labels=['Low', 'Mid', 'High'],
                          right=False)  # [0-5B), [5-20B), [20B+)

# Count by tier
tier_counts = df['price_tier'].value_counts().sort_index()
print(f"\nPrice Segments:")
print(f"  Low (0-5B VND): {tier_counts['Low']:,} ({tier_counts['Low']/len(df)*100:.1f}%)")
print(f"  Mid (5-20B VND): {tier_counts['Mid']:,} ({tier_counts['Mid']/len(df)*100:.1f}%)")
print(f"  High (>20B VND): {tier_counts['High']:,} ({tier_counts['High']/len(df)*100:.1f}%)")

# Key features to analyze
key_features = ['area_m2', 'width_m', 'length_m', 'num_floors', 'num_bedrooms', 'road_width_m']
tier_colors = {'Low': '#3498DB', 'Mid': '#E74C3C', 'High': '#2ECC71'}

# ============================================================================
# FIGURE 2.5: PRICE DISTRIBUTION BY PRICE TIER
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

fig.suptitle('Price Distribution by Tier (VND)', fontsize=16, fontweight='bold')

# Subplot 1: Overlapping histograms
ax1 = axes[0]
for tier in ['Low', 'Mid', 'High']:
    data = df[df['price_tier'] == tier]['price_vnd'].dropna() / 1e9
    ax1.hist(data, bins=50, alpha=0.5, label=f'{tier} (n={len(data):,})',
            color=tier_colors[tier], edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Price (Billion VND)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Frequency', fontweight='bold', fontsize=12)
ax1.set_title('Overlapping Price Distributions', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(alpha=0.3, axis='y')

# Subplot 2: Box plots
ax2 = axes[1]
data_to_plot = [df[df['price_tier'] == tier]['price_vnd'].dropna() / 1e9
                 for tier in ['Low', 'Mid', 'High']]

bp = ax2.boxplot(data_to_plot, patch_artist=True,
                 widths=0.6, showmeans=True,
                 meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
ax2.set_xticklabels(['Low', 'Mid', 'High'])

for patch, tier in zip(bp['boxes'], ['Low', 'Mid', 'High']):
    patch.set_facecolor(tier_colors[tier])
    patch.set_alpha(0.7)

ax2.set_ylabel('Price (Billion VND)', fontweight='bold', fontsize=12)
ax2.set_title('Price Statistics by Tier', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')

# Add value labels on box plot
for i, tier in enumerate(['Low', 'Mid', 'High'], 1):
    tier_prices = df[df['price_tier'] == tier]['price_vnd'].dropna() / 1e9
    q1 = tier_prices.quantile(0.25)
    median = tier_prices.median()
    q3 = tier_prices.quantile(0.75)
    mean = tier_prices.mean()

    print(f"\n{tier.upper()} TIER Price Stats:")
    print(f"  Q1: {q1:.2f}B | Median: {median:.2f}B | Q3: {q3:.2f}B | Mean: {mean:.2f}B")

plt.tight_layout()
plt.savefig(VIZ_DIR / '02_price_tier_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 02_price_tier_distribution.png")
plt.close()

# ============================================================================
# FIGURE 3: OVERLAPPING DISTRIBUTIONS BY PRICE TIER (6x1 layout)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

fig.suptitle('Data Distribution Across Price Segments (Low/Mid/High Tiers)',
             fontsize=16, fontweight='bold', y=0.995)

for idx, feat in enumerate(key_features):
    ax = axes[idx]

    # Plot distributions for each tier
    for tier in ['Low', 'Mid', 'High']:
        data = df[df['price_tier'] == tier][feat].dropna()
        ax.hist(data, bins=40, alpha=0.5, label=f'{tier} (n={len(data):,})',
                color=tier_colors[tier], edgecolor='black', linewidth=0.5)

    ax.set_xlabel(feat, fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'{feat} by Price Tier', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(VIZ_DIR / '03_price_segments_overlapping.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 03_price_segments_overlapping.png")
plt.close()

# ============================================================================
# FIGURE 4: BOX PLOTS BY PRICE TIER (6x1 layout)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

fig.suptitle('Feature Statistics by Price Tier (Box Plots)',
             fontsize=16, fontweight='bold', y=0.995)

for idx, feat in enumerate(key_features):
    ax = axes[idx]

    # Prepare data for box plot
    data_to_plot = [df[df['price_tier'] == tier][feat].dropna() for tier in ['Low', 'Mid', 'High']]

    bp = ax.boxplot(data_to_plot, patch_artist=True,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    ax.set_xticklabels(['Low', 'Mid', 'High'])

    # Color boxes
    for patch, tier in zip(bp['boxes'], ['Low', 'Mid', 'High']):
        patch.set_facecolor(tier_colors[tier])
        patch.set_alpha(0.7)

    ax.set_ylabel(feat, fontweight='bold', fontsize=11)
    ax.set_title(f'{feat} by Price Tier', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(VIZ_DIR / '04_price_segments_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_price_segments_boxplots.png")
plt.close()

# ============================================================================
# STATISTICS TABLE
# ============================================================================
print("\n" + "="*120)
print("FEATURE STATISTICS BY PRICE TIER")
print("="*120)

stats_data = []
for feat in key_features:
    for tier in ['Low', 'Mid', 'High']:
        data = df[df['price_tier'] == tier][feat].dropna()
        stats_data.append({
            'Feature': feat,
            'Tier': tier,
            'Count': len(data),
            'Mean': f"{data.mean():.2f}",
            'Median': f"{data.median():.2f}",
            'Std': f"{data.std():.2f}",
            'Min': f"{data.min():.2f}",
            'Max': f"{data.max():.2f}"
        })

stats_df = pd.DataFrame(stats_data)

# Print by feature
for feat in key_features:
    print(f"\n{feat.upper()}:")
    feature_stats = stats_df[stats_df['Feature'] == feat].drop('Feature', axis=1)
    print(feature_stats.to_string(index=False))

print("\n" + "="*120)

# ============================================================================
# PRICE TIER SUMMARY
# ============================================================================
print("\nPRICE TIER SUMMARY:")
print("="*120)

for tier in ['Low', 'Mid', 'High']:
    tier_data = df[df['price_tier'] == tier]['price_vnd']
    print(f"\n{tier.upper()} TIER:")
    print(f"  Count: {len(tier_data):,} properties")
    print(f"  Price Range: {tier_data.min()/1e9:.2f}B - {tier_data.max()/1e9:.2f}B VND")
    print(f"  Mean: {tier_data.mean()/1e9:.2f}B | Median: {tier_data.median()/1e9:.2f}B")
    print(f"  Typical Features:")

    # Get median feature values for this tier
    area = df[df['price_tier'] == tier]['area_m2'].median()
    width = df[df['price_tier'] == tier]['width_m'].median()
    beds = df[df['price_tier'] == tier]['num_bedrooms'].median()
    floors = df[df['price_tier'] == tier]['num_floors'].median()

    print(f"    Area: {area:.0f}m² | Width: {width:.1f}m | Bedrooms: {beds:.0f} | Floors: {floors:.0f}")

print("="*120)
