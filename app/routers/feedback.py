"""Feedback endpoints."""
from fastapi import APIRouter, HTTPException

from app.schemas.feedback import FeedbackRequest, FeedbackResponse, FeedbackStats
from app.services.feedback import submit_feedback, get_feedback_analytics

router = APIRouter(prefix="/api", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_user_feedback(request: FeedbackRequest):
    """Submit user feedback for model improvement."""
    try:
        print(f"[FEEDBACK] Received feedback: rating={request.rating}, bucket={request.bucket}")

        success = submit_feedback(
            predicted_price_vnd=request.predicted_price_vnd,
            actual_price_vnd=request.actual_price_vnd,
            rating=request.rating,
            bucket=request.bucket,
            confidence=request.confidence,
            features_json=request.features_json,
            feature_version=request.feature_version,
            timestamp=request.timestamp,
        )

        print(f"[FEEDBACK] Save result: {success}")

        if success:
            return FeedbackResponse(
                success=True,
                message="✅ Feedback saved to Supabase",
                id=1,
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to save feedback to Supabase")

    except HTTPException as e:
        print(f"[FEEDBACK] HTTP Exception: {e.detail}")
        raise
    except Exception as e:
        print(f"[FEEDBACK] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/feedback/stats")
async def get_stats() -> dict:
    """Get feedback statistics."""
    try:
        stats = get_feedback_analytics()
        if stats is None:
            return {
                "total_feedback": 0,
                "feedback_with_prices": 0,
                "message": "No feedback data yet"
            }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
