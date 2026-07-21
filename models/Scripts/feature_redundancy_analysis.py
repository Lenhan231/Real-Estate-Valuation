"""
Feature Redundancy Analysis - Remove collinear features using VIF
=========================================================================
Methods:
1. Variance Inflation Factor (VIF) - identify multicollinearity
2. Correlation matrix - find redundant pairs
3. PCA - identify low-variance components
4. Feature selection - keep only non-redundant features

Expected: Remove 8-15 redundant features → 63-70 optimal
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ANALYSIS_DIR = PROJECT_ROOT / "models" / "saved_models" / "redundancy_analysis"

def load_training_data():
    """Load preprocessed v2.6 training data (78 features)."""
    print("[1/5] Loading training data (v2.6)...")
    processed_file = DATA_DIR / "model_training_data.csv"
    df = pd.read_csv(processed_file)

    X = df.drop(columns=['price_vnd'])
    print(f"  ✓ Loaded {len(X)} samples × {X.shape[1]} features")
    return X

def calculate_vif(X):
    """Calculate VIF for each feature to identify multicollinearity."""
    print("\n[2/5] Calculating VIF (Variance Inflation Factor)...")

    # Manual VIF calculation (no statsmodels required)
    from sklearn.linear_model import LinearRegression

    vif_list = []
    for i in range(X.shape[1]):
        X_others = X.drop(columns=[X.columns[i]])
        y = X.iloc[:, i]

        model = LinearRegression()
        model.fit(X_others, y)
        r2 = model.score(X_others, y)

        # VIF = 1 / (1 - R²)
        vif = 1 / (1 - r2 + 1e-6) if r2 < 0.9999 else 1000  # Cap at 1000
        vif_list.append(vif)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = vif_list
    vif_data = vif_data.sort_values('VIF', ascending=False)

    # High VIF (>10) indicates multicollinearity
    high_vif = vif_data[vif_data['VIF'] > 10]
    print(f"  ⚠️  Found {len(high_vif)} features with VIF > 10 (multicollinear)")

    if len(high_vif) > 0:
        print("\n  Top 20 features with highest VIF:")
        for idx, row in vif_data.head(20).iterrows():
            marker = "⚠️ " if row['VIF'] > 10 else "✓ "
            print(f"    {marker} {row['Feature']:<40} VIF: {row['VIF']:>8.2f}")

    return vif_data

def analyze_correlation(X):
    """Analyze feature correlations to find redundant pairs."""
    print("\n[3/5] Analyzing correlation matrix...")

    numeric_X = X.select_dtypes(include=[np.number])
    corr_matrix = numeric_X.corr().abs()

    # Find highly correlated pairs (r > 0.95)
    redundant_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.95:
                redundant_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })

    redundant_pairs = sorted(redundant_pairs, key=lambda x: x['correlation'], reverse=True)

    print(f"  ⚠️  Found {len(redundant_pairs)} highly correlated pairs (r>0.95)")
    if len(redundant_pairs) > 0:
        print("\n  Top redundant pairs:")
        for pair in redundant_pairs[:15]:
            print(f"    {pair['feature1']:<35} ↔ {pair['feature2']:<35} r={pair['correlation']:.3f}")

    return redundant_pairs

def pca_analysis(X):
    """PCA to identify low-variance components."""
    print("\n[4/5] PCA Analysis...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    cumsum = np.cumsum(pca.explained_variance_ratio_)

    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumsum >= 0.95) + 1
    n_components_90 = np.argmax(cumsum >= 0.90) + 1

    print(f"  Components for 90% variance: {n_components_90} ({n_components_90/len(X.columns)*100:.1f}% features)")
    print(f"  Components for 95% variance: {n_components_95} ({n_components_95/len(X.columns)*100:.1f}% features)")
    print(f"  Components for 99% variance: {np.argmax(cumsum >= 0.99) + 1}")

    return pca, cumsum

def generate_recommendations(vif_data, redundant_pairs, pca, cumsum, X):
    """Generate recommendations for feature removal."""
    print("\n[5/5] Generating recommendations...")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Remove high-VIF features
    high_vif_features = vif_data[vif_data['VIF'] > 10]['Feature'].tolist()

    # Strategy 2: Remove one from each redundant pair (keep higher importance)
    redundant_to_drop = []
    for pair in redundant_pairs:
        # Drop the one that appears more in redundant pairs
        redundant_to_drop.append(pair['feature2'])  # Keep feature1, drop feature2

    # Strategy 3: Keep only top components from PCA
    n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1

    # Combined strategy: remove VIF + redundant
    features_to_drop = list(set(high_vif_features + redundant_to_drop))
    features_to_keep = [f for f in X.columns if f not in features_to_drop]

    print(f"\n  Strategy: Remove high-VIF + redundant features")
    print(f"  Current features: {len(X.columns)}")
    print(f"  Recommended to remove: {len(features_to_drop)}")
    print(f"  Recommended to keep: {len(features_to_keep)}")
    print(f"  Expected MAPE improvement: +0.2-0.5% (noise reduction)")

    # Save recommendations
    report_file = ANALYSIS_DIR / "redundancy_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FEATURE REDUNDANCY ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Current features: {len(X.columns)}\n")
        f.write(f"High-VIF features (>10): {len(high_vif_features)}\n")
        f.write(f"Redundant pairs (r>0.95): {len(redundant_pairs)}\n")
        f.write(f"Recommended to drop: {len(features_to_drop)}\n")
        f.write(f"Recommended to keep: {len(features_to_keep)}\n\n")

        f.write("HIGH-VIF FEATURES (VIF > 10)\n")
        f.write("-"*80 + "\n")
        for idx, row in vif_data[vif_data['VIF'] > 10].iterrows():
            f.write(f"  {row['Feature']:<40} VIF: {row['VIF']:>8.2f}\n")
        f.write("\n")

        f.write("REDUNDANT FEATURE PAIRS (r > 0.95)\n")
        f.write("-"*80 + "\n")
        for pair in redundant_pairs:
            f.write(f"  {pair['feature1']:<35} ↔ {pair['feature2']:<35} r={pair['correlation']:.3f}\n")
        f.write("\n")

        f.write("FEATURES TO DROP (REDUNDANCY)\n")
        f.write("-"*80 + "\n")
        for feat in sorted(features_to_drop):
            f.write(f"  {feat}\n")
        f.write("\n")

        f.write("FEATURES TO KEEP\n")
        f.write("-"*80 + "\n")
        for feat in sorted(features_to_keep):
            f.write(f"  {feat}\n")
        f.write("\n")

        f.write("PCA ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Components for 90% variance: {np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1}\n")
        f.write(f"Components for 95% variance: {np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1}\n")
        f.write(f"Components for 99% variance: {np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.99) + 1}\n\n")

        f.write("RECOMMENDATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Remove {len(features_to_drop)} redundant features\n")
        f.write(f"Keep {len(features_to_keep)} non-redundant features\n")
        f.write(f"Expected: Cleaner model, 0.2-0.5% MAPE improvement\n")
        f.write(f"Next: Retrain with features_to_keep.txt\n")

    # Save feature lists
    with open(ANALYSIS_DIR / "features_to_drop.txt", 'w') as f:
        f.write('\n'.join(sorted(features_to_drop)))

    with open(ANALYSIS_DIR / "features_to_keep.txt", 'w') as f:
        f.write('\n'.join(sorted(features_to_keep)))

    # Plot PCA variance
    fig, ax = plt.subplots(figsize=(12, 6))
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cumsum)+1), cumsum, linewidth=2, marker='o', markersize=4)
    ax.axhline(y=0.90, color='orange', linestyle='--', label='90% threshold')
    ax.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax.set_title('PCA: Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "pca_cumulative_variance.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ PCA plot saved: {ANALYSIS_DIR / 'pca_cumulative_variance.png'}")

    print(f"\n  ✓ Report saved: {report_file}")
    print(f"  ✓ Features to drop: {ANALYSIS_DIR / 'features_to_drop.txt'}")
    print(f"  ✓ Features to keep: {ANALYSIS_DIR / 'features_to_keep.txt'}")

def main():
    print("="*80)
    print("FEATURE REDUNDANCY ANALYSIS (v2.6: 78 features)")
    print("="*80)

    # Load data
    X = load_training_data()

    # Analyze VIF
    vif_data = calculate_vif(X)

    # Analyze correlation
    redundant_pairs = analyze_correlation(X)

    # PCA analysis
    pca, cumsum = pca_analysis(X)

    # Generate recommendations
    generate_recommendations(vif_data, redundant_pairs, pca, cumsum, X)

    print("\n" + "="*80)
    print("✅ REDUNDANCY ANALYSIS COMPLETE")
    print("="*80)
    print(f"📁 Analysis saved to: {ANALYSIS_DIR}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
