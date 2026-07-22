"""Inference service for price predictions."""
from typing import Tuple, Dict, Any

from app.core.models import get_models, get_geo
from app.core.inference import build_row, predict_price
from app.core.parsers import parse_listing
from app.core.explainability import (
    get_feature_importance_from_model,
    get_model_predictions,
    calculate_confidence_score,
)


def predict_property_price(
    street: str,
    locality: str,
    property_type: str,
    legal_status: str,
    direction: str,
    area_m2: float,
    width_m: float,
    length_m: float,
    num_floors: float,
    num_bedrooms: float,
    road_width_m: float,
    bin_flags: Dict[str, int],
    text_flags: Dict[str, int],
) -> Tuple[float, Dict[str, Any], Dict[str, float], Dict[str, Any]]:
    """Predict property price and return XAI data.

    Returns:
        (price_vnd, info, row, xai_data)
    """
    models, meta, _, geo = get_models()

    if not models or not meta:
        raise RuntimeError("Models not loaded")

    # Build feature row
    row, info = build_row(
        meta, geo,
        street=street,
        locality=locality,
        property_type=property_type or "Nhà phố",
        legal_status=legal_status or "Sổ hồng",
        direction=direction or "Không",
        area_m2=area_m2,
        width_m=width_m or 0,
        length_m=length_m or 0,
        num_floors=num_floors,
        num_bedrooms=num_bedrooms,
        road_width_m=road_width_m,
        bin_flags=bin_flags or {},
        text_flags=text_flags or {},
    )

    if row is None:
        raise ValueError(f"Failed to build feature row: {info}")

    # Determine price tier
    bucket = "low"
    est_price = area_m2 * 50e6
    if est_price > 20e9:
        bucket = "high"
    elif est_price > 5e9:
        bucket = "mid"

    # Get predictions from ensemble
    price_vnd = predict_price(models, meta, row, bucket)

    # Get XAI data
    feature_importance = get_feature_importance_from_model(models, meta)
    model_preds = get_model_predictions(models, meta, row, bucket)
    confidence = calculate_confidence_score(model_preds)

    xai_data = {
        "models": model_preds,
        "confidence": confidence,
        "bucket": bucket,
        "feature_importance": dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]),
    }

    return price_vnd, info, row, xai_data


def parse_listing_text(text: str) -> Dict[str, Any]:
    """Parse listing description."""
    return parse_listing(text)


def get_localities() -> list:
    """Get all available localities."""
    geo = get_geo()
    return geo.localities()
