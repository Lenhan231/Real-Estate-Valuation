"""Prediction endpoints."""
from fastapi import APIRouter, HTTPException

from app.schemas.predict import PredictRequest, PredictResponse, ParseListingRequest, ParseListingResponse
from app.services.inference import predict_property_price, parse_listing_text, get_localities

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict property price with XAI."""
    try:
        price_vnd, info, row, xai_data = predict_property_price(
            street=request.street,
            locality=request.locality,
            property_type=request.property_type,
            legal_status=request.legal_status,
            direction=request.direction,
            price_tier=request.price_tier,
            area_m2=request.area_m2,
            width_m=request.width_m or 0,
            length_m=request.length_m or 0,
            num_floors=request.num_floors,
            num_bedrooms=request.num_bedrooms,
            road_width_m=request.road_width_m,
            bin_flags=request.bin_flags or {},
            text_flags=request.text_flags or {},
        )

        return PredictResponse(
            price_vnd=float(price_vnd),
            price_billion_vnd=float(price_vnd / 1e9),
            bucket=xai_data["bucket"],
            xai=xai_data,
            row=row,
            info=info,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[PREDICT] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/parse", response_model=ParseListingResponse)
async def parse_listing(request: ParseListingRequest):
    """Parse listing description."""
    try:
        props = parse_listing_text(request.text)
        return ParseListingResponse(properties=props)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse failed: {str(e)}")


@router.get("/localities")
async def localities():
    """Get all available localities."""
    try:
        locs = get_localities()
        return {"localities": locs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
