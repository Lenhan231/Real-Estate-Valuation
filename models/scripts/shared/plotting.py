import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importance(model, feature_names, save_path):
    """Plot top 20 feature importances from model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 8))
    plt.title("Top 20 Feature Importances (LGBM)")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def plot_pred_vs_actual(y_true, y_pred, save_path):
    """Plot predicted vs actual values scatter plot."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, edgecolors='none', s=15)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price (VND)')
    plt.ylabel('Predicted Price (VND)')
    plt.title('Predicted vs Actual Prices')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path
