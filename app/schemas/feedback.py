"""Schemas for feedback requests/responses."""
from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class FeedbackRequest(BaseModel):
    """Feedback submission request."""
    predicted_price_vnd: float
    actual_price_vnd: Optional[float] = None
    rating: str
    bucket: str
    confidence: float
    features_json: Optional[Dict[str, Any]] = None
    feature_version: int = 1  # Current feature version
    timestamp: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Feedback submission response."""
    success: bool
    message: str
    id: Optional[int] = None


class FeedbackStats(BaseModel):
    """Feedback statistics."""
    total_feedback: int
    feedback_with_prices: int
    mean_error_pct: float
    mae_pct: float
    mape_pct: float
    worst_predictions: List[Dict[str, Any]]
