"""Schemas for prediction requests/responses."""
from pydantic import BaseModel
from typing import Optional, Dict, Any


class PredictRequest(BaseModel):
    """Price prediction request."""
    street: str
    locality: str
    property_type: Optional[str] = None
    legal_status: Optional[str] = None
    direction: Optional[str] = None
    area_m2: float
    width_m: Optional[float] = None
    length_m: Optional[float] = None
    num_floors: Optional[float] = None
    num_bedrooms: Optional[float] = None
    road_width_m: Optional[float] = None
    bin_flags: Optional[Dict[str, int]] = None
    text_flags: Optional[Dict[str, int]] = None


class PredictResponse(BaseModel):
    """Price prediction response."""
    price_vnd: float
    price_billion_vnd: float
    bucket: str
    xai: Dict[str, Any]
    row: Dict[str, float]
    info: Dict[str, Any]


class ParseListingRequest(BaseModel):
    """Listing text parsing request."""
    text: str


class ParseListingResponse(BaseModel):
    """Parsing response."""
    properties: Dict[str, Any]
