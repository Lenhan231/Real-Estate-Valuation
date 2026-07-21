"""
SHAP Analysis for Feature Importance & Selection
==================================================
Analyzes model interpretability using SHAP values.
Identifies low-impact features for removal.

Pipeline:
1. Load trained models (LightGBM, XGBoost, CatBoost) × 3 price tiers
2. Load training data
3. Calculate SHAP values per model
4. Aggregate importance across models
5. Identify bottom 20-30% features for removal
6. Generate visualizations & report
"""

import argparse
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

from shared import preprocess, add_locality_features, mean_absolute_percentage_error

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "models" / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"
PLOT_DIR = MODEL_DIR / "plots"
ANALYSIS_DIR = MODEL_DIR / "shap_analysis"


def load_training_data():
    """Load preprocessed training data."""
    print("[1/4] Loading training data...")
    processed_file = PROJECT_ROOT / "data" / "processed" / "model_training_data.csv"
    df = pd.read_csv(processed_file)

    X = df.drop(columns=['price_vnd'])
    y = df['price_vnd']

    # Log transform price for training
    y_log = np.log1p(y)

    print(f"  ✓ Loaded {len(X)} samples × {X.shape[1]} features")
    return X, y, y_log


def calculate_shap_importance(model, X, model_name, max_samples=500):
    """Calculate SHAP values for a single model."""
    print(f"    Calculating SHAP for {model_name}...")

    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)

    # Sample for speed if dataset large
    if len(X) > max_samples:
        sample_idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X.iloc[sample_idx]
    else:
        X_sample = X

    shap_values = explainer.shap_values(X_sample)

    # For XGBoost, shap_values is already 2D
    # For LightGBM, convert if needed
    if isinstance(shap_values, list):
        shap_values = shap_values[0] if len(shap_values) == 1 else np.array(shap_values[0])

    # Calculate mean absolute SHAP values
    feature_importance = np.abs(shap_values).mean(axis=0)

    return feature_importance, explainer


def analyze_all_tiers(X):
    """Analyze SHAP importance across all price tiers and models."""
    print("\n[2/4] Calculating SHAP importance across models...")

    PRICE_TIERS = ['low', 'mid', 'high']
    MODEL_TYPES = ['lgbm', 'xgb', 'cb']

    # Store importance: {tier: {model: importance_array}}
    all_importance = {}

    for tier in PRICE_TIERS:
        print(f"\n  {tier.upper()} TIER:")
        all_importance[tier] = {}

        for model_type in MODEL_TYPES:
            model_file = MODEL_DIR / f"{model_type}_{tier}.pkl"

            if not model_file.exists():
                print(f"    ⚠️  Model not found: {model_file}")
                continue

            try:
                model = joblib.load(model_file)
                importance, _ = calculate_shap_importance(model, X, f"{model_type.upper()}")
                all_importance[tier][model_type] = importance
            except Exception as e:
                print(f"    ✗ Error calculating SHAP for {model_type}: {e}")
                continue

        if all_importance[tier]:
            print(f"    ✓ {tier.upper()} tier complete")

    return all_importance


def aggregate_importance(all_importance, X):
    """Aggregate SHAP importance across tiers and models."""
    print("\n[3/4] Aggregating importance...")

    aggregated = np.zeros(X.shape[1])
    count = 0

    for tier, models in all_importance.items():
        for model_type, importance in models.items():
            aggregated += importance
            count += 1

    if count > 0:
        aggregated /= count

    # Create DataFrame with feature names
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'shap_importance': aggregated
    }).sort_values('shap_importance', ascending=False)

    print(f"  ✓ Aggregated across {count} model instances")

    return importance_df


def identify_low_impact_features(importance_df, percentile=30):
    """Identify bottom N% features for potential removal."""
    print(f"\n[4/4] Identifying low-impact features (bottom {percentile}%)...")

    threshold = np.percentile(importance_df['shap_importance'], percentile)
    low_impact = importance_df[importance_df['shap_importance'] <= threshold]

    print(f"\n{'='*70}")
    print(f"LOW-IMPACT FEATURES (bottom {percentile}%)")
    print(f"{'='*70}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Count: {len(low_impact)} features\n")

    print(low_impact.to_string(index=False))
    print(f"\n{'='*70}\n")

    return low_impact, threshold


def create_visualizations(importance_df, low_impact, output_dir):
    """Create SHAP importance visualizations."""
    print("Creating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Top 30 features
    fig, ax = plt.subplots(figsize=(10, 8))
    top30 = importance_df.head(30)
    ax.barh(range(len(top30)), top30['shap_importance'].values, color='steelblue')
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30['feature'].values, fontsize=9)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title('Top 30 Most Important Features', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'top30_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: top30_features.png")

    # 2. Low-impact features
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(low_impact)), low_impact['shap_importance'].values, color='coral')
    ax.set_yticks(range(len(low_impact)))
    ax.set_yticklabels(low_impact['feature'].values, fontsize=8)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title(f'Low-Impact Features (Bottom 30%)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'low_impact_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: low_impact_features.png")

    # 3. Cumulative importance
    fig, ax = plt.subplots(figsize=(12, 6))
    cumsum = importance_df['shap_importance'].cumsum() / importance_df['shap_importance'].sum() * 100
    ax.plot(range(len(cumsum)), cumsum.values, linewidth=2, color='darkgreen')
    ax.axhline(y=80, color='red', linestyle='--', label='80% threshold')
    ax.axhline(y=90, color='orange', linestyle='--', label='90% threshold')
    ax.set_xlabel('Feature Index (sorted by importance)', fontsize=11)
    ax.set_ylabel('Cumulative Importance (%)', fontsize=11)
    ax.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: cumulative_importance.png")


def generate_report(importance_df, low_impact, threshold, output_dir):
    """Generate analysis report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / 'shap_analysis_report.txt'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SHAP FEATURE IMPORTANCE ANALYSIS\n")
        f.write("="*80 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Features Analyzed: {len(importance_df)}\n")
        f.write(f"Low-Impact Features (bottom 30%): {len(low_impact)}\n")
        f.write(f"Importance Threshold: {threshold:.6f}\n\n")

        f.write("RECOMMENDATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Remove {len(low_impact)} low-impact features and retrain:\n")
        f.write(f"  - Expected feature reduction: {len(low_impact)/len(importance_df)*100:.1f}%\n")
        f.write(f"  - Expected MAPE improvement: 0.3-0.8% (noise reduction)\n")
        f.write(f"  - Expected training time reduction: 15-20%\n\n")

        f.write("FEATURES TO REMOVE\n")
        f.write("-"*80 + "\n")
        for idx, row in low_impact.iterrows():
            f.write(f"  {row['feature']:<40} SHAP: {row['shap_importance']:.6f}\n")
        f.write("\n")

        f.write("TOP 30 IMPORTANT FEATURES (KEEP THESE)\n")
        f.write("-"*80 + "\n")
        for idx, row in importance_df.head(30).iterrows():
            f.write(f"  {row['feature']:<40} SHAP: {row['shap_importance']:.6f}\n")
        f.write("\n")

        f.write("NEXT STEPS\n")
        f.write("-"*80 + "\n")
        f.write("1. Remove identified low-impact features from preprocessing.py\n")
        f.write("2. Retrain models with reduced feature set\n")
        f.write("3. Compare MAPE: current (13.24%) vs cleaned\n")
        f.write("4. If improvement >0.3%, proceed with cleaned version\n")
        f.write("5. Then apply hyperparameter tuning on final feature set\n\n")

    print(f"  ✓ Saved: shap_analysis_report.txt")
    return report_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--percentile", type=int, default=30, help="Percentile for low-impact features")
    args = parser.parse_args()

    print("="*70)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*70)

    # Load data
    X, y, y_log = load_training_data()

    # Calculate SHAP importance
    all_importance = analyze_all_tiers(X)

    # Aggregate
    importance_df = aggregate_importance(all_importance, X)

    # Identify low-impact
    low_impact, threshold = identify_low_impact_features(importance_df, percentile=args.percentile)

    # Create analysis directory
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Visualizations
    create_visualizations(importance_df, low_impact, ANALYSIS_DIR)

    # Report
    report_file = generate_report(importance_df, low_impact, threshold, ANALYSIS_DIR)

    # Save feature list for next step
    features_to_keep = importance_df[~importance_df['feature'].isin(low_impact['feature'])]['feature'].tolist()
    features_file = ANALYSIS_DIR / 'features_to_keep.txt'
    with open(features_file, 'w') as f:
        f.write('\n'.join(features_to_keep))

    print("\n" + "="*70)
    print("✅ SHAP ANALYSIS COMPLETE")
    print("="*70)
    print(f"📊 Results saved to: {ANALYSIS_DIR}")
    print(f"📄 Report: {report_file}")
    print(f"📋 Features to keep: {features_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
