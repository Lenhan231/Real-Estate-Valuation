"""Listing parser - extract data from Vietnamese real estate listing text."""
import re
from .constants import TEXT_FEATURE_PATTERNS, NUMERIC_PATTERNS, LEGAL_PATTERNS


def parse_listing(text: str) -> dict:
    """Parse Vietnamese real estate listing DESCRIPTION → extract property data.

    Handles full listing text like:
    "Vlasta Premier Phú Thuận, Đường Đào Trí, Phường Phú Thuận, Quận 7, HCM
     80m2, 3 tầng, 3 phòng ngủ, rộng 5m, nội thất, sổ hồng"
    """
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
        "street": "",
        "locality": "",
    }

    text_lower = text.lower()

    # ===== NUMERIC FEATURES =====
    # Try all numeric patterns in shared constants
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

    # Additional area patterns (alternatives if standard fails)
    if result["area_m2"] == 80.0:
        # Try "diện tích XXm2" pattern
        area_match = re.search(r'diện tích\s*(\d+(?:[.,]\d+)?)\s*m[²2²]', text_lower)
        if area_match:
            result["area_m2"] = float(area_match.group(1).replace(",", "."))

    # Additional floor patterns
    if result["num_floors"] == 3:
        floor_match = re.search(r'(?:^|\s)(\d+)\s*lầu', text_lower)  # "4 lầu" (colloquial)
        if floor_match:
            result["num_floors"] = int(floor_match.group(1))

    # Additional bedroom patterns
    if result["num_bedrooms"] == 3:
        bed_match = re.search(r'(\d+)\s*(?:bedroom|phòng)', text_lower)
        if bed_match:
            result["num_bedrooms"] = int(bed_match.group(1))

    # ===== TEXT-BASED FLAGS =====
    # Check all text feature patterns from constants
    for flag_name, patterns in TEXT_FEATURE_PATTERNS.items():
        if any(pattern in text_lower for pattern in patterns):
            result[flag_name] = True

    # ===== LEGAL STATUS =====
    # Check legal status patterns
    for status, patterns in LEGAL_PATTERNS.items():
        if any(pattern in text_lower for pattern in patterns):
            result["legal_status"] = status
            break

    # ===== ADDRESS EXTRACTION (from description) =====
    # Extract street: look for "Đường XXX" or just the first street-like phrase
    street_match = re.search(r'(?:đường|street|đ\.?)\s+([^,\n]+?)(?:,|\s+phường|\s+xã|\s+quận|$)', text_lower)
    if street_match:
        street_raw = street_match.group(1).strip()
        # Clean up: remove "số" prefix if present
        street_clean = re.sub(r'^(?:số\s*\d+[a-z]?\s+)?', '', street_raw).strip()
        result["street"] = street_clean.title() if street_clean else ""

    # Extract locality (phường/xã): PRIORITIZE actual locality names over ward numbers!
    # Strategy: 1) Look for "Phường/Xã + real name" patterns (avoid ward numbers like "Phường 24")
    #          2) Skip "Phường/Xã + digits" (old format ward numbers)
    #          3) Fallback to known ward names

    # First, try to find "Phường XXX" where XXX is a real name (not just digits)
    # Look for pattern like "Phường Bình Thạnh" (not "Phường 24")
    locality_match = re.search(r'(?:phường|xã)\s+([a-zÀ-ÿ\s]+?)(?:,|\s+quận|\s+huyện|hồ chí minh|\s*mới|$)', text_lower)
    if locality_match:
        locality_raw = locality_match.group(1).strip()
        # Only accept if it's not just numbers (avoid "Phường 24")
        if not locality_raw.replace(" ", "").isdigit():
            result["locality"] = f"phường {locality_raw.title()}"

    # If we only found "Phường 24" type, look for named phường nearby in the text
    if not result["locality"] or re.match(r'phường\s+\d+$', result["locality"].lower()):
        # Look for locality mentioned in different part (e.g., after "mới")
        # Like "Phường Bình Thạnh, Hồ Chí Minh mới"
        alt_match = re.search(r'phường\s+([a-zÀ-ÿ]+)', text_lower)
        if alt_match:
            locality_name = alt_match.group(1).strip()
            if not locality_name.isdigit():
                result["locality"] = f"phường {locality_name.title()}"

    # Ultimate fallback: if still not found, try to infer from known names
    if not result["locality"]:
        known_wards = ["phú thuận", "bình thạnh", "tân bình", "tân phú", "bến thành",
                      "quận 1", "quận 3", "quận 4", "phú nhuận", "gò vấp"]
        for ward in known_wards:
            if ward in text_lower:
                result["locality"] = f"phường {ward.split()[-1].title()}"
                break

    return result


def extract_street_from_address(text: str) -> str:
    """Try to extract street name from address text."""
    # Look for "Đường XXX" pattern
    match = re.search(r'(?:đường|street)\s+([^,]+)', text.lower())
    if match:
        return match.group(1).strip()
    return ""
