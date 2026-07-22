"""Shared constants for listing parsing and preprocessing."""

# Text-based feature patterns (used by both preprocessing.py and parsers.py)
TEXT_FEATURE_PATTERNS = {
    "is_hem_xe_hoi": ["hẻm xe hơi", "hxh", "ô tô vào", "xe hơi ngủ", "mặt ngõ", "hẻm"],
    "is_mat_tien": ["mặt tiền", "mặt phố"],
    "is_no_hau": ["nở hậu"],
    "has_noi_that": ["nội thất", "full", "đầy đủ", "trang trí", "hiện đại"],
    "is_gap": ["gấp", "giảm giá", "cần bán", "bớt lộc"],
    "is_kinh_doanh": ["kinh doanh", "cho thuê", "thu nhập", "tiếp khách"],
}

# Numeric feature patterns
NUMERIC_PATTERNS = {
    "area_m2": r'(\d+(?:[.,]\d+)?)\s*m[²2²]',  # 46m2, 46 m²
    "num_floors": r'(\d+)\s*tầng',  # 3 tầng
    "num_bedrooms": r'(\d+)\s*(?:phòng ngủ|PN)',  # 4 phòng ngủ, 4 PN
    "road_width_m": r'(?:hẻm|đường)(?:\s+\w+)*?\s*(?:trước|rộng)?\s*(?:\w+)?\s*(\d+)\s*m',  # hẻm rộng 5m
}

# Legal status patterns
LEGAL_PATTERNS = {
    "so_hong_so_do": ["sổ hồng", "sổ đỏ"],
    "giay_to_hop_le": ["giấy tờ"],
}
