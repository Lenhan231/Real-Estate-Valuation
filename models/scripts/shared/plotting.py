import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importance(model, feature_names, save_path):
    """Plot the top 20 feature importances from the averaged ensemble."""
    if isinstance(model, dict):
        ensemble = []
        for value in model.values():
            if isinstance(value, dict):
                for nested in value.values():
                    if hasattr(nested, "feature_importances_"):
                        ensemble.append(np.asarray(nested.feature_importances_, dtype=float))
            elif hasattr(value, "feature_importances_"):
                ensemble.append(np.asarray(value.feature_importances_, dtype=float))

        if not ensemble:
            raise ValueError("No feature_importances_ found in provided models")

        importances = np.mean(ensemble, axis=0)
    else:
        importances = np.asarray(model.feature_importances_, dtype=float)

    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 8))
    plt.title("Global Top 20 Feature Importances (Ensemble)")
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
