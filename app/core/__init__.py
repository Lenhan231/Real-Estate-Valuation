"""Core modules for real estate valuation app."""
from .geo import GeoLookup
from .parsers import parse_listing, extract_street_from_address

# Lazy imports for inference to avoid circular dependencies and import issues
def __getattr__(name):
    if name == "load_models":
        from .inference import load_models
        return load_models
    elif name == "build_row":
        from .inference import build_row
        return build_row
    elif name == "predict_price":
        from .inference import predict_price
        return predict_price
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GeoLookup",
    "load_models",
    "build_row",
    "predict_price",
    "parse_listing",
    "extract_street_from_address",
]
