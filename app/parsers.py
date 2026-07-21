"""Listing parser - extract data from Vietnamese real estate listing text."""
import re


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

    # Diện tích: "46m2", "46 m²", "diện tích 46"
    area_match = re.search(r'(\d+(?:[.,]\d+)?)\s*m[²2²]', text_lower)
    if area_match:
        result["area_m2"] = float(area_match.group(1).replace(",", "."))

    # Số tầng: "3 tầng", "lầu"
    floor_match = re.search(r'(\d+)\s*tầng', text_lower)
    if floor_match:
        result["num_floors"] = int(floor_match.group(1))

    # Phòng ngủ: "4 phòng ngủ", "4 PN"
    bed_match = re.search(r'(\d+)\s*(?:phòng ngủ|PN)', text_lower)
    if bed_match:
        result["num_bedrooms"] = int(bed_match.group(1))

    # Đường trước nhà: "5m", "hẻm rộng 5m", "hẻm trước nhà rộng 5m"
    road_match = re.search(
        r'(?:hẻm|đường)(?:\s+\w+)*?\s*(?:trước|rộng)?\s*(?:\w+)?\s*(\d+)\s*m',
        text_lower
    )
    if road_match:
        result["road_width_m"] = float(road_match.group(1))

    # Cần bán gấp
    if "cần bán gấp" in text_lower or "bán gấp" in text_lower:
        result["is_gap"] = True

    # Hẻm xe hơi
    if "hẻm xe" in text_lower or "ô tô vào" in text_lower:
        result["is_hem_xe_hoi"] = True

    # Nở hậu
    if "nở hậu" in text_lower:
        result["is_no_hau"] = True

    # Có nội thất / cho thuê
    if "nội thất" in text_lower or "cho thuê" in text_lower:
        result["has_noi_that"] = True

    # Tiện kinh doanh
    if "kinh doanh" in text_lower or "thích hợp" in text_lower:
        result["is_kinh_doanh"] = True

    # Pháp lý: Sổ hồng / Sổ đỏ
    if "sổ hồng" in text_lower or "sổ đỏ" in text_lower:
        result["legal_status"] = "so_hong_so_do"
    elif "giấy tờ" in text_lower:
        result["legal_status"] = "giay_to_hop_le"

    return result


def extract_street_from_address(text: str) -> str:
    """Try to extract street name from address text."""
    # Look for "Đường XXX" pattern
    match = re.search(r'(?:đường|street)\s+([^,]+)', text.lower())
    if match:
        return match.group(1).strip()
    return ""
