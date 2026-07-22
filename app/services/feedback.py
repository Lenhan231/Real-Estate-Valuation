"""Feedback service for collecting and analyzing feedback."""
from typing import Dict, Any, Optional

from app.core.feedback import save_feedback_to_supabase, get_feedback_stats


def submit_feedback(
    predicted_price_vnd: float,
    actual_price_vnd: Optional[float],
    rating: str,
    bucket: str,
    confidence: float,
    features_json: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> bool:
    """Submit user feedback to Supabase."""
    feedback_data = {
        "predicted_price_vnd": predicted_price_vnd,
        "actual_price_vnd": actual_price_vnd,
        "rating": rating,
        "bucket": bucket,
        "confidence": confidence,
        "timestamp": timestamp,
    }
    return save_feedback_to_supabase(feedback_data, row_dict=features_json)


def get_feedback_analytics() -> Optional[Dict[str, Any]]:
    """Get feedback statistics and analytics."""
    return get_feedback_stats()
