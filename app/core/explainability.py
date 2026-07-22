"""Explainable AI (XAI) for model predictions."""
import numpy as np
import pandas as pd
from pathlib import Path


def get_feature_importance_from_model(models, meta):
    """Extract feature importance from trained models (LightGBM/XGBoost/CatBoost)."""
    importances = {}
    feature_names = meta.get("feature_names", [])

    for model_name, model in models.items():
        try:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                importances[model_name] = dict(zip(feature_names, imp))
        except Exception:
            pass

    if importances:
        all_features = set()
        for imp_dict in importances.values():
            all_features.update(imp_dict.keys())

        avg_importance = {}
        for feat in all_features:
            values = [imp.get(feat, 0) for imp in importances.values()]
            avg_importance[feat] = np.mean(values)

        total = sum(avg_importance.values())
        if total > 0:
            avg_importance = {k: (v / total) * 100 for k, v in avg_importance.items()}

        return avg_importance
    return {}


def get_model_predictions(models, meta, row_dict, bucket):
    """Get predictions from all 3 models in the bucket ensemble.

    Args:
        models: Dict of trained models
        meta: Metadata with feature names
        row_dict: Feature dict from build_row (single sample)
        bucket: Budget range ('low', 'mid', 'high')
    """
    predictions = {}
    feature_names = meta.get("feature_names", [])

    model_keys = [f"lgbm_{bucket}", f"xgb_{bucket}", f"cb_{bucket}"]

    for key in model_keys:
        if key in models:
            try:
                model = models[key]

                # Create DataFrame with the row
                X = pd.DataFrame([row_dict])

                # Ensure all features exist (fill missing with 0)
                for feat in feature_names:
                    if feat not in X.columns:
                        X[feat] = 0.0

                # Select only features model expects
                if hasattr(model, "feature_names_in_"):
                    model_features = list(model.feature_names_in_)
                    X_pred = X[model_features]
                else:
                    X_pred = X[feature_names]

                pred_log = float(model.predict(X_pred)[0])
                predictions[key] = pred_log
            except Exception as e:
                print(f"Error predicting with {key}: {e}")

    return predictions


def calculate_confidence_score(predictions):
    """Calculate confidence score based on model agreement.

    Range interpretation:
    - 85-100%: Very high agreement (reliable)
    - 70-85%: Good agreement (confident)
    - 50-70%: Moderate agreement (caution)
    - <50%: Low agreement (uncertain)
    """
    if not predictions or len(predictions) == 0:
        return 0.0

    values = list(predictions.values())
    if len(values) < 2:
        return 75.0

    # Calculate model agreement
    std = np.std(values)
    mean = np.mean(values)

    # Coefficient of variation (normalized std)
    cv = std / (abs(mean) + 1e-6)

    # Scale: cv < 0.01 (1%) = very high confidence
    #        cv > 0.05 (5%) = low confidence
    # Map to 50-100% range (never below 50% with multiple models)
    confidence = max(50, min(100, 100 - (cv * 500)))  # Scale factor adjusted

    return float(confidence)


def get_feature_contributions(row_dict, importance_dict, base_value=0):
    """Calculate how much each feature contributed to the prediction (THIS property)."""
    if not importance_dict:
        return []

    top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]

    contributions = []
    for feat, imp_score in top_features:
        value = row_dict.get(feat, 0)

        # Estimate contribution: importance * normalized value
        if isinstance(value, (int, float)):
            contrib = (imp_score / 100) * abs(value) * 0.01
            contributions.append({
                "feature": feat,
                "importance": imp_score,
                "value": value,
                "contribution": contrib,
            })

    return contributions
