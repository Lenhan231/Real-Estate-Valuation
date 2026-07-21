"""Listing parser - extract data from Vietnamese real estate listing text."""
import re
from .constants import TEXT_FEATURE_PATTERNS, NUMERIC_PATTERNS, LEGAL_PATTERNS


def parse_listing(text: str) -> dict:
    """Parse Vietnamese listing text → extract property data."""
    result = {
        "area_m2": 80.0,
        "num_floors": 3,
        "num_bedrooms": 3,
        "width_m": 4.0,
        "length_m": 20.0,
        "road_width_m": 6.0,
        "is_gap": False,
        "is_hem_xe_hoi": False,
        "has_noi_that": False,
        "is_kinh_doanh": False,
        "is_no_hau": False,
        "legal_status": "unknown",
    }

    text_lower = text.lower()

    # Numeric features using shared patterns
    for key, pattern in NUMERIC_PATTERNS.items():
        match = re.search(pattern, text_lower)
        if match:
            try:
                val = float(match.group(1).replace(",", "."))
                if key == "num_floors" or key == "num_bedrooms":
                    result[key] = int(val)
                else:
                    result[key] = val
            except (ValueError, IndexError):
                pass

    # Text-based flags using shared patterns
    for flag_name, patterns in TEXT_FEATURE_PATTERNS.items():
        if any(pattern in text_lower for pattern in patterns):
            result[flag_name] = True

    # Legal status using shared patterns
    for status, patterns in LEGAL_PATTERNS.items():
        if any(pattern in text_lower for pattern in patterns):
            result["legal_status"] = status
            break

    return result


def extract_street_from_address(text: str) -> str:
    """Try to extract street name from address text."""
    # Look for "Đường XXX" pattern
    match = re.search(r'(?:đường|street)\s+([^,]+)', text.lower())
    if match:
        return match.group(1).strip()
    return ""
