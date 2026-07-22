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
        "area_m2": None,
        "num_floors": None,
        "num_bedrooms": None,
        "width_m": None,
        "length_m": None,
        "road_width_m": None,
        "is_gap": False,
        "is_hem_xe_hoi": False,
        "has_noi_that": False,
        "is_kinh_doanh": False,
        "is_no_hau": False,
        "legal_status": "unknown",
        "street": "",
        "locality": "",
        "property_type": None,
        "direction": "unknown",
        "price_range": None,
    }

    text_lower = text.lower()

    # ===== NUMERIC FEATURES =====
    # Area: "52m2" or "52 m²" or "diện tích 52m2"
    area_patterns = [
        r'(\d+(?:[.,]\d+)?)\s*m[²2]',  # 52m2, 52 m²
        r'diện tích\s*(\d+(?:[.,]\d+)?)\s*m[²2]',  # diện tích 52m2
    ]
    for pattern in area_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result["area_m2"] = float(match.group(1).replace(",", "."))
            break

    # Floors: "5 tầng" or "5 lầu"
    floor_patterns = [
        r'(\d+)\s*tầng',  # 5 tầng
        r'(\d+)\s*lầu',   # 5 lầu (colloquial)
    ]
    for pattern in floor_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result["num_floors"] = int(match.group(1))
            break

    # Bedrooms: "4 phòng ngủ" or "4 PN"
    bed_patterns = [
        r'(\d+)\s*phòng\s*ngủ',  # 4 phòng ngủ
        r'(\d+)\s*pn',            # 4 PN
    ]
    for pattern in bed_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result["num_bedrooms"] = int(match.group(1))
            break

    # Width (mặt tiền / chiều ngang): "4m" or "4 m" (after "mặt tiền" or "rộng")
    width_patterns = [
        r'mặt\s*tiền\s+(\d+(?:[.,]\d+)?)\s*m',  # mặt tiền 4m
        r'rộng\s+(\d+(?:[.,]\d+)?)\s*m',         # rộng 4m
        r'chiều\s*ngang\s+(\d+(?:[.,]\d+)?)\s*m',  # chiều ngang 4m
    ]
    for pattern in width_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result["width_m"] = float(match.group(1).replace(",", "."))
            break

    # Length (chiều dài): "4x13" or "13m" or "chiều dài 13m"
    length_patterns = [
        r'(?:\d+\s*x\s*)(\d+(?:[.,]\d+)?)\s*m',  # 4x13 or 4 x 13
        r'chiều\s*dài\s+(\d+(?:[.,]\d+)?)\s*m',  # chiều dài 13m
        r'\((\d+)\s*x\s*(\d+)\)',  # (4x13) - take second number
    ]
    for pattern in length_patterns:
        match = re.search(pattern, text_lower)
        if match:
            if '(' in pattern:  # Special case for (4x13)
                result["length_m"] = float(match.group(2))
            else:
                result["length_m"] = float(match.group(1).replace(",", "."))
            break

    # Road width: "đường vào 4,5m" or "hẻm rộng 4m"
    road_patterns = [
        r'đường\s*vào\s+(\d+(?:[.,]\d+)?)\s*m',  # đường vào 4.5m
        r'hẻm\s*rộng\s+(\d+(?:[.,]\d+)?)\s*m',   # hẻm rộng 4m
    ]
    for pattern in road_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result["road_width_m"] = float(match.group(1).replace(",", "."))
            break

    # ===== TEXT-BASED FLAGS =====
    for flag_name, patterns in TEXT_FEATURE_PATTERNS.items():
        if any(pattern in text_lower for pattern in patterns):
            result[flag_name] = True

    # ===== LEGAL STATUS =====
    for status, patterns in LEGAL_PATTERNS.items():
        if any(pattern in text_lower for pattern in patterns):
            result["legal_status"] = status
            break

    # ===== PROPERTY TYPE =====
    # nha_mat_tien: "mặt tiền", "mặt phố", "nhà mặt phố"
    # nha_trong_hem: "hẻm", "trong hẻm", "nhà trong hẻm"
    if re.search(r'(?:nhà\s*)?(?:mặt\s*tiền|mặt\s*phố)', text_lower):
        result["property_type"] = "nha_mat_tien"
    elif re.search(r'(?:nhà\s*)?(?:trong\s*hẻm|hẻm)', text_lower):
        result["property_type"] = "nha_trong_hem"

    # ===== DIRECTION (hướng nhà) =====
    directions = {
        "dong": ["hướng đông", "h đông", "h. đông", "hướng đ", "phía đông", "đông"],
        "tay": ["hướng tây", "h tây", "h. tây", "hướng t", "phía tây", "tây nam", "tây"],
        "nam": ["hướng nam", "h nam", "h. nam", "hướng n", "phía nam", "nam"],
        "bac": ["hướng bắc", "h bắc", "h. bắc", "hướng b", "phía bắc", "bắc"],
        "dong_nam": ["hướng đông nam", "đông nam"],
        "dong_bac": ["hướng đông bắc", "đông bắc"],
        "tay_nam": ["hướng tây nam", "tây nam"],
        "tay_bac": ["hướng tây bắc", "tây bắc"],
    }
    for dir_key, patterns in directions.items():
        if any(pattern in text_lower for pattern in patterns):
            result["direction"] = dir_key
            break

    # ===== PRICE TIER (from listing) =====
    # Extract price in tỷ (billion) and categorize
    price_match = re.search(r'(\d+(?:[.,]\d+)?)\s*tỷ', text_lower)
    if price_match:
        price_b = float(price_match.group(1).replace(",", "."))
        if price_b < 5:
            result["price_range"] = "low"
        elif price_b < 20:
            result["price_range"] = "mid"
        else:
            result["price_range"] = "high"

    # ===== ADDRESS EXTRACTION (from description) =====
    # Extract street: look for "Đường XXX" or just the first street-like phrase
    street_match = re.search(r'(?:đường|street|đ\.?)\s+([^,\n]+?)(?:,|\s+phường|\s+xã|\s+quận|$)', text_lower)
    if street_match:
        street_raw = street_match.group(1).strip()
        # Clean up: remove "số" prefix if present
        street_clean = re.sub(r'^(?:số\s*\d+[a-z]?\s+)?', '', street_raw).strip()
        result["street"] = street_clean.title() if street_clean else ""
        print(f"✅ [PARSER] Street found: {result['street']}")
    else:
        print(f"❌ [PARSER] No street match found")

    # Extract locality (phường/xã): PRIORITIZE actual locality names over ward numbers!
    # Find all "phường/xã XXX" patterns, prefer ones with real names over numbers

    all_phường_matches = re.findall(r'(?:phường|xã)\s+([^\n,]+?)(?:,|$)', text_lower)
    print(f"🔍 [PARSER] Phường matches found: {all_phường_matches}")

    if all_phường_matches:
        # Go through matches and prioritize non-numeric ones
        for match in all_phường_matches:
            locality_raw = match.strip()
            # Reject if it's just numbers like "24" or "14"
            # Accept anything with at least one letter (e.g., "bình thạnh", "phú nhuận")
            if locality_raw and not locality_raw.replace(" ", "").isdigit():
                result["locality"] = f"phường {locality_raw.title()}"
                break

        # If all matches were numeric, just use the first one (it'll be last resort)
        if not result["locality"] and all_phường_matches:
            first_match = all_phường_matches[0].strip()
            result["locality"] = f"phường {first_match.title()}"

    # Fallback: if still not found, try to infer from known names in text
    if not result["locality"]:
        known_wards = ["phú thuận", "bình thạnh", "tân bình", "tân phú", "bến thành",
                      "quận 1", "quận 3", "quận 4", "phú nhuận", "gò vấp", "bình tân",
                      "thủ đức", "gò vấp", "tân phú"]
        for ward in known_wards:
            if ward in text_lower:
                ward_name = ward.split()[-1].replace("quận", "").strip()
                result["locality"] = f"phường {ward_name.title()}"
                print(f"✅ [PARSER] Locality found (fallback): {result['locality']}")
                break

    if result["locality"]:
        print(f"✅ [PARSER] Final locality: {result['locality']}")
    else:
        print(f"❌ [PARSER] No locality found!")

    return result


def extract_street_from_address(text: str) -> str:
    """Try to extract street name from address text."""
    # Look for "Đường XXX" pattern
    match = re.search(r'(?:đường|street)\s+([^,]+)', text.lower())
    if match:
        return match.group(1).strip()
    return ""
