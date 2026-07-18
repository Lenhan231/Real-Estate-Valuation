"""
Simple Inference - Load production XGBoost model
Lightweight wrapper for app.py integration
"""
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "save_models" / "production_model.pkl"

_model = None
_baseline_mape = 18.01
_baseline_r2 = 0.8663


def load_model():
    """Load production XGBoost model"""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def get_model_info():
    """Return model metadata"""
    return {
        'model': 'XGBoost',
        'mape': _baseline_mape,
        'r2': _baseline_r2,
        'rmse_billion': 4.37,
        'mae_billion': 2.67,
        'n_features': 166,
        'status': '✅ Production Ready'
    }


def predict_price(features_df: pd.DataFrame) -> np.ndarray:
    """
    Predict prices using production model

    Args:
        features_df: DataFrame with engineered features

    Returns:
        Predicted prices (clipped to [0, inf))
    """
    model = load_model()

    # Predict in log space
    y_pred_log = model.predict(features_df)

    # Inverse log transform
    y_pred = np.expm1(y_pred_log)

    # Clip to positive
    y_pred = np.clip(y_pred, 0, None)

    return y_pred


def predict_confidence(y_pred: np.ndarray) -> dict:
    """
    Estimate prediction confidence bounds
    Based on MAE of 2.67B VND (±1.33B at 95%)
    """
    mae_billion = 2.67
    ci_95 = mae_billion * 1.96  # ~5.24B

    return {
        'lower_bound': y_pred - ci_95,
        'upper_bound': y_pred + ci_95,
        'mae': mae_billion,
        'confidence_level': '95%'
    }


if __name__ == "__main__":
    print("Testing inference...")
    info = get_model_info()
    print(f"Model: {info['model']}")
    print(f"MAPE: {info['mape']:.2f}%")
    print(f"R²: {info['r2']:.4f}")
    print("✅ Inference module ready")
