"""Admin endpoints for model management and retraining."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from models.retraining import (
    get_retraining_data,
    detect_model_drift,
    retrain_models,
    compare_models,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


class RetrainingResponse(BaseModel):
    """Retraining result response."""
    status: str
    message: str
    data: Dict[str, Any] = {}


@router.post("/retrain", response_model=RetrainingResponse)
async def trigger_retrain():
    """Trigger model retraining using feedback data."""
    try:
        print("\n" + "="*80)
        print("🔄 [RETRAIN] Starting model retraining pipeline")
        print("="*80)

        # Step 1: Get retraining data
        feedback_df = get_retraining_data(min_samples=3)
        if feedback_df is None:
            raise HTTPException(
                status_code=400,
                detail="Insufficient feedback data for retraining (need ≥3 samples)"
            )

        # Step 2: Detect drift on current data
        print("\n📊 Detecting model drift...")
        drift_result = detect_model_drift(feedback_df)

        # Step 3: Retrain models
        print("\n🔄 Retraining models...")
        retrain_result = retrain_models(feedback_df)

        if retrain_result is None:
            raise HTTPException(status_code=500, detail="Model retraining failed")

        # Step 4: Compare performance
        print("\n📈 Comparing model performance...")
        comparison = compare_models(feedback_df)

        result = {
            "timestamp": retrain_result["timestamp"],
            "samples_used": retrain_result["samples_used"],
            "val_samples": retrain_result["val_samples"],
            "old_vs_new": comparison,
            "drift_analysis": drift_result,
            "model_performance": retrain_result["models"],
        }

        print("\n" + "="*80)
        print("✅ [RETRAIN] Retraining completed successfully!")
        print("="*80 + "\n")

        return RetrainingResponse(
            status="success",
            message="✅ Models retrained successfully",
            data=result,
        )

    except HTTPException as e:
        print(f"❌ [RETRAIN] HTTP Error: {e.detail}")
        raise
    except Exception as e:
        print(f"❌ [RETRAIN] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")


@router.get("/drift-status")
async def get_drift_status():
    """Get current model drift status."""
    try:
        feedback_df = get_retraining_data(min_samples=1)
        if feedback_df is None:
            return {"status": "no_data", "message": "No feedback data available"}

        drift_result = detect_model_drift(feedback_df)

        return {
            "status": "ok" if not drift_result["is_drift_detected"] else "warning",
            "drift_detected": drift_result["is_drift_detected"],
            "overall_mape": drift_result["overall_mape"],
            "by_bucket": drift_result["drift_by_bucket"],
            "samples": drift_result["samples_evaluated"],
        }

    except Exception as e:
        print(f"Error getting drift status: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/model-comparison")
async def get_model_comparison():
    """Get model performance comparison."""
    try:
        feedback_df = get_retraining_data(min_samples=1)
        if feedback_df is None:
            return {"status": "no_data", "message": "No feedback data available"}

        comparison = compare_models(feedback_df)

        return {
            "status": "ok",
            "metrics": comparison,
            "samples_evaluated": len(feedback_df),
        }

    except Exception as e:
        print(f"Error getting model comparison: {e}")
        return {"status": "error", "message": str(e)}
