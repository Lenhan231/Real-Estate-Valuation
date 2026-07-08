import re
import numpy as np
import pandas as pd
from pathlib import Path

BINARY_COLS_VI = [
    "Phòng ăn",
    "Nhà bếp",
    "Sân thượng",
    "Chổ để xe hơi",
    "Chính chủ"
]

MISSING_MARKERS = ["---", "_", "", "null", "None", "nan"]

VI_TO_EN_COLS = {
    "Mã tin": "listing_id",
    "Hướng": "direction",
    "Phòng ăn": "dining_room",
    "Loại tin": "listing_type",
    "Đường trước nhà": "road_width_m",
    "Nhà bếp": "kitchen",
    "Loại BDS": "property_type",
    "Pháp lý": "legal_status",
    "Sân thượng": "terrace",
    "Chiều ngang": "width_m",
    "Số lầu": "num_floors",
    "Chổ để xe hơi": "car_parking",
    "Chiều dài": "length_m",
    "Số phòng ngủ": "num_bedrooms",
    "Chính chủ": "owner_listing",
    "Thông tin chi tiết": "description",
    "Giá": "price_vnd",
    "area": "area_m2"
}

DIRECTION_MAP = {
    "Đông Bắc": "Dong_Bac",
    "Đông Nam": "Dong_Nam",
    "Tây Bắc": "Tay_Bac",
    "Tây Nam": "Tay_Nam",
    "Đông": "Dong",
    "Tây": "Tay",
    "Nam": "Nam",
    "Bắc": "Bac",
}

LEGAL_MAP = {
    "Sổ hồng/ Sổ đỏ": "so_hong_so_do",
    "Giấy tờ hợp lệ": "giay_to_hop_le",
}

TYPE_MAP = {
    "Cần bán": "can_ban",
    "Nhà mặt tiền": "nha_mat_tien",
    "Nhà trong hẻm": "nha_trong_hem",
}

KEEP_COLS = [
    "link", "title", "post_day","street", "old_address", "locality", "region", "listing_id", "direction",
    "listing_type", "property_type", "legal_status", "num_floors",
    "num_bedrooms", "road_width_m", "width_m", "length_m",
    "price_vnd", "area_m2", "description"
]

BINARY_COLS_EN = ["dining_room", "kitchen", "terrace", "car_parking", "owner_listing"]


def to_float_m(value):
    if pd.isna(value):
        return np.nan
    s = str(value).lower().strip().replace(",", ".")
    s = re.sub(r"(\d+)m(\d{1,2})\b", r"\1.\2", s)
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan


def parse_price_to_vnd(value, square=None):
    if pd.isna(value):
        return np.nan
    s = str(value).lower().strip().replace("giá:", "").strip().replace(",", ".")
    s = re.sub(r"\s+", " ", s)

    if any(x in s for x in ["thỏa thuận", "thoả thuận", "liên hệ", "lien he", "đang cập nhật"]):
        return np.nan

    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return np.nan

    num = float(m.group(1))

    if any(x in s for x in ["triệu / m²", "triệu/m²", "triệu / m2", "triệu/m2"]):
        if square is None or pd.isna(square):
            return np.nan
        return num * 1_000_000 * float(square)

    if "tỷ" in s or "ty" in s:
        return num * 1_000_000_000
    if "triệu" in s or "trieu" in s:
        return num * 1_000_000
    if "nghìn" in s or "nghin" in s:
        return num * 1_000

    return num


def to_int(value):
    if pd.isna(value):
        return np.nan
    s = str(value).lower().strip()
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan


def normalize_text(value):
    if pd.isna(value):
        return np.nan
    return str(value).strip()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    if "được đánh giá" in df.columns:
        df["được đánh giá"] = (
                df["được đánh giá"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )
        df = df[df["được đánh giá"] > 2].copy()

    df = df.drop(columns=["Số Điện Thoại", "được đánh giá", "Tên liên hệ"], errors="ignore")
    df = df.drop_duplicates(subset=["link"]).copy()

    if "Mã tin" in df.columns:
        df["Mã tin"] = pd.to_numeric(df["Mã tin"], errors="coerce").astype("Int64")

    for col in BINARY_COLS_VI:
        if col in df.columns:
            df[col] = (df[col]
                      .replace(["---", "_"], "0")
                      .fillna("1")
                      .astype("Int64"))

    cols_to_clean = [col for col in df.columns if col not in BINARY_COLS_VI]
    df[cols_to_clean] = df[cols_to_clean].replace(MISSING_MARKERS, np.nan)

    for col in ["Đường trước nhà", "Chiều ngang", "Chiều dài", "area"]:
        if col in df.columns:
            df[col] = df[col].apply(to_float_m)
    
    if "Giá" in df.columns:
        df["Giá"] = df.apply(
            lambda row: parse_price_to_vnd(row.get("Giá"), row.get("area")),
            axis=1
        )

    for col in ["Số lầu", "Số phòng ngủ"]:
        if col in df.columns:
            df[col] = df[col].apply(to_int).astype("Int64")

    categorical_cols = ["street", "locality", "region", "Hướng", "Loại tin", "Loại BDS", "Pháp lý"]
    for col in categorical_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(normalize_text)
        if col == "Hướng":
            df[col] = df[col].replace(DIRECTION_MAP)
        elif col == "Pháp lý":
            df[col] = df[col].replace(LEGAL_MAP)
        elif col in ["Loại tin", "Loại BDS"]:
            df[col] = df[col].replace(TYPE_MAP)
        df[col] = df[col].fillna("unknown").str.lower().str.strip()

    for col in ["Số lầu", "Số phòng ngủ"]:
        if col in df.columns:
            df = df[(df[col].isna()) | (df[col] >= 0)]

    for col in ["Chiều ngang", "Chiều dài", "Đường trước nhà"]:
        if col in df.columns:
            df = df[(df[col].isna()) | (df[col] > 0)]

    if "Giá" in df.columns:
        df = df[(df["Giá"].isna()) | (df["Giá"] > 0)]

    df = df.rename(columns={**VI_TO_EN_COLS})
    df = df.loc[:, ~df.columns.duplicated()].copy()

    for col in BINARY_COLS_EN:
        if col in df.columns:
            df[col + "_bin"] = df[col]

    cols_to_keep = KEEP_COLS + [col + "_bin" for col in BINARY_COLS_EN]
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]

    cleaned_df = df[cols_to_keep].copy()
    output_file = Path("data/processed/alonhadat_cleaned.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_file, index=False)
    return cleaned_df

def final_clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()

if __name__ == "__main__":
    pass