"""Model retraining pipeline using feedback data."""
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


MODEL_DIR = Path(__file__).parent / "saved_models"


def get_retraining_data(min_samples=5):
    """Extract feedback data for model retraining.

    Args:
        min_samples: Minimum feedback samples required to retrain

    Returns:
        DataFrame with features and actual prices, or None if insufficient data
    """
    try:
        from supabase import create_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")

        if not url or not key:
            print("❌ Supabase credentials not set")
            return None

        client = create_client(url, key)

        # Fetch feedback with features and actual prices
        response = client.table("feedback").select("*").neq("features_json", "{}").neq("actual_price_vnd", None).execute()

        if not response.data or len(response.data) < min_samples:
            print(f"❌ Insufficient feedback data: {len(response.data) if response.data else 0} samples (need {min_samples})")
            return None

        # Extract records
        records = []
        for row in response.data:
            if row.get("features_json") and row.get("actual_price_vnd"):
                feature_row = row["features_json"]
                feature_row["price_vnd"] = row["actual_price_vnd"]
                records.append(feature_row)

        if not records:
            print("❌ No valid feedback records for retraining")
            return None

        df = pd.DataFrame(records)
        print(f"✅ Loaded {len(df)} feedback samples for retraining")

        return df

    except Exception as e:
        print(f"❌ Error getting retraining data: {e}")
        return None


def detect_model_drift(feedback_df, drift_threshold=5.0):
    """Detect model drift by comparing performance on new feedback.

    Args:
        feedback_df: DataFrame with feedback data
        drift_threshold: MAPE difference threshold (%) to flag drift

    Returns:
        Dict with drift detection results
    """
    try:
        if feedback_df is None or feedback_df.empty:
            return None

        # Load current models
        lgb_model = joblib.load(MODEL_DIR / "lightgbm_model.pkl")
        xgb_model = joblib.load(MODEL_DIR / "xgboost_model.pkl")
        cat_model = joblib.load(MODEL_DIR / "catboost_model.pkl")

        # Prepare features (same as training)
        X = feedback_df.drop(columns=["price_vnd"])
        y = feedback_df["price_vnd"]

        # Make predictions with current models
        lgb_pred = np.expm1(lgb_model.predict(X))
        xgb_pred = np.expm1(xgb_model.predict(X))
        cat_pred = np.expm1(cat_model.predict(X))

        # Ensemble prediction
        ensemble_pred = (lgb_pred + xgb_pred + cat_pred) / 3

        # Calculate MAPE
        mape = np.mean(np.abs((y - ensemble_pred) / y)) * 100

        # Calculate by bucket (if available)
        drift_by_bucket = {}
        if "bucket" in feedback_df.columns:
            for bucket in feedback_df["bucket"].unique():
                mask = feedback_df["bucket"] == bucket
                bucket_y = y[mask]
                bucket_pred = ensemble_pred[mask]
                bucket_mape = np.mean(np.abs((bucket_y - bucket_pred) / bucket_y)) * 100
                drift_by_bucket[bucket] = float(bucket_mape)

        result = {
            "overall_mape": float(mape),
            "drift_by_bucket": drift_by_bucket,
            "is_drift_detected": mape > 15.0,  # Alert if MAPE > 15%
            "samples_evaluated": len(feedback_df),
        }

        print(f"📊 Model Drift Analysis: MAPE={mape:.2f}% | Drift Detected: {result['is_drift_detected']}")

        return result

    except Exception as e:
        print(f"❌ Error detecting drift: {e}")
        return None


def retrain_models(feedback_df, test_size=0.2, random_state=42):
    """Retrain all models using feedback data.

    Args:
        feedback_df: DataFrame with feedback data
        test_size: Fraction for validation set
        random_state: Random seed for reproducibility

    Returns:
        Dict with training results and model paths
    """
    try:
        if feedback_df is None or feedback_df.empty:
            print("❌ No feedback data for retraining")
            return None

        print(f"\n🔄 Starting model retraining with {len(feedback_df)} samples...")

        # Prepare data
        X = feedback_df.drop(columns=["price_vnd"])
        y = np.log1p(feedback_df["price_vnd"])  # Log transform for stability

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"  Train: {len(X_train)} | Val: {len(X_val)}")

        # Retrain LightGBM
        print("  Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            random_state=random_state,
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        lgb_pred = lgb_model.predict(X_val)
        lgb_mape = np.mean(np.abs((np.expm1(y_val) - np.expm1(lgb_pred)) / np.expm1(y_val))) * 100
        print(f"    LightGBM MAPE: {lgb_mape:.2f}%")

        # Retrain XGBoost
        print("  Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            random_state=random_state,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_pred = xgb_model.predict(X_val)
        xgb_mape = np.mean(np.abs((np.expm1(y_val) - np.expm1(xgb_pred)) / np.expm1(y_val))) * 100
        print(f"    XGBoost MAPE: {xgb_mape:.2f}%")

        # Retrain CatBoost
        print("  Training CatBoost...")
        cat_model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            depth=7,
            random_state=random_state,
            verbose=False,
        )
        cat_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], use_best_model=True)
        cat_pred = cat_model.predict(X_val)
        cat_mape = np.mean(np.abs((np.expm1(y_val) - np.expm1(cat_pred)) / np.expm1(y_val))) * 100
        print(f"    CatBoost MAPE: {cat_mape:.2f}%")

        # Ensemble
        ensemble_pred = (np.expm1(lgb_pred) + np.expm1(xgb_pred) + np.expm1(cat_pred)) / 3
        ensemble_mape = np.mean(np.abs((np.expm1(y_val) - ensemble_pred) / np.expm1(y_val))) * 100
        print(f"    Ensemble MAPE: {ensemble_mape:.2f}%")

        # Save new models
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(lgb_model, MODEL_DIR / "lightgbm_model.pkl")
        joblib.dump(xgb_model, MODEL_DIR / "xgboost_model.pkl")
        joblib.dump(cat_model, MODEL_DIR / "catboost_model.pkl")

        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "samples_used": len(X_train),
            "val_samples": len(X_val),
            "models": {
                "lightgbm_mape": float(lgb_mape),
                "xgboost_mape": float(xgb_mape),
                "catboost_mape": float(cat_mape),
                "ensemble_mape": float(ensemble_mape),
            },
        }

        print(f"\n✅ Retraining complete! Models saved to {MODEL_DIR}")

        return result

    except Exception as e:
        print(f"❌ Error retraining models: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_models(feedback_df):
    """Compare old vs new model performance on feedback data.

    Args:
        feedback_df: Feedback data for comparison

    Returns:
        Dict with comparison results
    """
    try:
        if feedback_df is None or feedback_df.empty:
            return None

        # Prepare data
        X = feedback_df.drop(columns=["price_vnd"])
        y = feedback_df["price_vnd"]

        # Load models
        lgb_model = joblib.load(MODEL_DIR / "lightgbm_model.pkl")
        xgb_model = joblib.load(MODEL_DIR / "xgboost_model.pkl")
        cat_model = joblib.load(MODEL_DIR / "catboost_model.pkl")

        # Predictions
        lgb_pred = np.expm1(lgb_model.predict(X))
        xgb_pred = np.expm1(xgb_model.predict(X))
        cat_pred = np.expm1(cat_model.predict(X))
        ensemble_pred = (lgb_pred + xgb_pred + cat_pred) / 3

        # Calculate metrics
        metrics = {}
        for name, pred in [("LightGBM", lgb_pred), ("XGBoost", xgb_pred),
                           ("CatBoost", cat_pred), ("Ensemble", ensemble_pred)]:
            mape = np.mean(np.abs((y - pred) / y)) * 100
            mae = np.mean(np.abs(y - pred))
            metrics[name] = {
                "mape": float(mape),
                "mae": float(mae / 1e9),  # Convert to billions
            }

        return metrics

    except Exception as e:
        print(f"❌ Error comparing models: {e}")
        return None
