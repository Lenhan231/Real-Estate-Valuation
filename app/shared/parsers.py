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
    # Find all "phường/xã XXX" patterns, prefer ones with real names over numbers

    all_phường_matches = re.findall(r'(?:phường|xã)\s+([^\n,]+?)(?:,|$)', text_lower)

    if all_phường_matches:
        # Go through matches and prioritize non-numeric ones
        for match in all_phường_matches:
            locality_raw = match.strip().split()[0] if match.strip().split() else ""
            # Reject if it's just numbers like "24" or "14"
            if locality_raw and not locality_raw.isdigit():
                result["locality"] = f"phường {locality_raw.title()}"
                break

        # If all matches were numeric, just use the first one (it'll be last resort)
        if not result["locality"] and all_phường_matches:
            result["locality"] = f"phường {all_phường_matches[0].split()[0].title()}"

    # Fallback: if still not found, try to infer from known names in text
    if not result["locality"]:
        known_wards = ["phú thuận", "bình thạnh", "tân bình", "tân phú", "bến thành",
                      "quận 1", "quận 3", "quận 4", "phú nhuận", "gò vấp", "bình tân",
                      "thủ đức", "gò vấp", "tân phú"]
        for ward in known_wards:
            if ward in text_lower:
                ward_name = ward.split()[-1].replace("quận", "").strip()
                result["locality"] = f"phường {ward_name.title()}"
                break

    return result


def extract_street_from_address(text: str) -> str:
    """Try to extract street name from address text."""
    # Look for "Đường XXX" pattern
    match = re.search(r'(?:đường|street)\s+([^,]+)', text.lower())
    if match:
        return match.group(1).strip()
    return ""
