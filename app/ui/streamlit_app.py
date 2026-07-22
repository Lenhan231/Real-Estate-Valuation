"""Định giá nhà TP.HCM — Valuation + Market Analysis
Chạy: streamlit run app/app.py
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup path from PROJECT_ROOT only (before any module imports)
# __file__ = app/ui/streamlit_app.py, so go up 3 levels to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
try:
    load_dotenv(PROJECT_ROOT / ".env")
    print("[STREAMLIT-STARTUP] ✓ .env loaded successfully")
except Exception as e:
    print(f"[STREAMLIT-STARTUP] ✗ Error loading .env: {e}")

import pandas as pd
import streamlit as st
import requests
from requests.exceptions import RequestException

# Verify API connection
print("[STREAMLIT-STARTUP] Verifying API connection...")

try:
    import pydeck as pdk
except Exception:
    pdk = None

# Import from consolidated app.core module
from app.core.geo import GeoLookup

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30

# Feature version (same as app/core/constants.py)
FEATURE_VERSION = 1

ROOT = PROJECT_ROOT
BI_DATA_FILE = ROOT / "data" / "processed" / "model_training_data.csv"

st.set_page_config(
    page_title="Định giá & Phân tích BĐS TP.HCM",
    page_icon="🏠",
    layout="wide",
)

# Initialize session state for valuation & feedback
if "feedback_status" not in st.session_state:
    st.session_state.feedback_status = None
if "feedback_message" not in st.session_state:
    st.session_state.feedback_message = None
if "valuation_result" not in st.session_state:
    st.session_state.valuation_result = None
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "last_price" not in st.session_state:
    st.session_state.last_price = None
if "last_info" not in st.session_state:
    st.session_state.last_info = None
if "last_row" not in st.session_state:
    st.session_state.last_row = None
if "last_xai_data" not in st.session_state:
    st.session_state.last_xai_data = None

# ---------------------------------------------------------------------------
# API Helper Functions
# ---------------------------------------------------------------------------
def api_predict(street, locality, property_type, legal_status, direction,
                 area_m2, width_m, length_m, num_floors, num_bedrooms, road_width_m,
                 bin_flags, text_flags):
    """Call /api/predict endpoint."""
    try:
        payload = {
            "street": street,
            "locality": locality,
            "property_type": property_type,
            "legal_status": legal_status,
            "direction": direction,
            "area_m2": area_m2,
            "width_m": width_m,
            "length_m": length_m,
            "num_floors": num_floors,
            "num_bedrooms": num_bedrooms,
            "road_width_m": road_width_m,
            "bin_flags": bin_flags,
            "text_flags": text_flags,
        }
        response = requests.post(
            f"{API_BASE_URL}/api/predict",
            json=payload,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        return {"error": f"API error: {str(e)}"}


def api_parse(text):
    """Call /api/parse endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/parse",
            json={"text": text},
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json()["properties"]
    except RequestException as e:
        return {"error": str(e)}


def api_localities():
    """Call /api/localities endpoint."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/localities",
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json()["localities"]
    except RequestException as e:
        st.error(f"Failed to load localities: {e}")
        return []


# ---------------------------------------------------------------------------
# Load assets (Geo only, models are on API)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_geo():
    return GeoLookup()

@st.cache_data(show_spinner=False)
def load_bi_data():
    """Load BI data from Supabase Raw_Features table."""
    try:
        from supabase import create_client
        import os

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            return pd.DataFrame()

        client = create_client(url, key)

        # Fetch all data with pagination
        all_data = []
        page_size = 1000
        offset = 0

        while True:
            response = client.table("Raw_Features").select("*").range(offset, offset + page_size - 1).execute()
            if not response.data:
                break
            all_data.extend(response.data)
            if len(response.data) < page_size:
                break
            offset += page_size

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        required = ["lat", "lon", "price_vnd", "area_m2", "post_day", "locality", "property_type"]
        if any(c not in df.columns for c in required):
            return pd.DataFrame()

        df = df.dropna(subset=["lat", "lon", "price_vnd", "area_m2", "post_day"]).copy()
        df["post_day"] = pd.to_datetime(df["post_day"], errors="coerce")
        df = df.dropna(subset=["post_day"]).copy()
        df["price_billion_vnd"] = df["price_vnd"] / 1e9
        df["price_per_m2_million"] = (df["price_vnd"] / df["area_m2"]) / 1e6
        df["month"] = df["post_day"].dt.to_period("M").dt.to_timestamp()
        return df
    except Exception as e:
        print(f"Error loading BI data: {e}")
        return pd.DataFrame()

geo = load_geo()

# ---------------------------------------------------------------------------
# Helper function for valuation
# ---------------------------------------------------------------------------
def do_valuation(street, locality, property_type, legal_status, direction,
                 area_m2, width_m, length_m, num_floors, num_bedrooms, road_width_m,
                 bin_flags, text_flags):
    """Run valuation via API."""
    result = api_predict(
        street=street,
        locality=locality,
        property_type=property_type,
        legal_status=legal_status,
        direction=direction,
        area_m2=area_m2,
        width_m=width_m,
        length_m=length_m,
        num_floors=num_floors,
        num_bedrooms=num_bedrooms,
        road_width_m=road_width_m,
        bin_flags=bin_flags,
        text_flags=text_flags,
    )

    if "error" in result:
        return None, result["error"], None, None

    # Extract from API response
    price_vnd = result["price_vnd"]
    bucket = result["bucket"]
    row = result["row"]
    info = result["info"]

    xai_data = {
        "importance": result["xai"].get("feature_importance", {}),
        "model_predictions": result["xai"].get("models", {}),
        "confidence": result["xai"].get("confidence", 0),
        "bucket": bucket,
    }

    return price_vnd, info, row, xai_data

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
PROPERTY_TYPES = {
    "nha_mat_tien": "Nhà mặt tiền",
    "nha_trong_hem": "Nhà trong hẻm",
}
BUDGET_RANGES = {
    "low": "Dưới 5 tỷ",
    "mid": "Từ 5 đến 20 tỷ",
    "high": "Trên 20 tỷ",
}
LEGAL_STATUS = {
    "so_hong_so_do": "Sổ hồng / Sổ đỏ",
    "giay_to_hop_le": "Giấy tờ hợp lệ",
    "unknown": "Không rõ",
}
DIRECTIONS = {
    "unknown": "Không rõ",
    "dong": "Đông", "tay": "Tây", "nam": "Nam", "bac": "Bắc",
    "dong_nam": "Đông Nam", "dong_bac": "Đông Bắc",
    "tay_nam": "Tây Nam", "tay_bac": "Tây Bắc",
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏠 Định giá & Phân tích Bất Động Sản TP.HCM")
col_title, col_status = st.columns([3, 1])
with col_title:
    st.caption(
        "Model: **v2.4 Ensemble (LightGBM + XGBoost + CatBoost)** · "
        "R² **0.9187** · MAPE **13.25%**"
    )
with col_status:
    status_color = "🟢" if geo.data_source == "Supabase" else "🟡"
    st.caption(f"{status_color} {geo.data_source}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_valuation, tab_analysis, tab_feedback = st.tabs(["💰 Định giá", "📊 Phân tích thị trường", "📈 Feedback Analytics"])

# =========================================================================
# TAB 1: VALUATION
# =========================================================================
with tab_valuation:
    mode = st.radio("📝 Cách nhập", ["📋 Paste mô tả", "✍️ Form chi tiết"], horizontal=True)

    if mode == "📋 Paste mô tả":
        st.subheader("📍 Dán Mô Tả Bài Đăng")
        st.caption("Paste toàn bộ mô tả bài đăng BĐS (địa chỉ + thông số)")
        address_text = st.text_area(
            "Ví dụ: Vlasta Premier Phú Thuận, Đường Đào Trí, Phường Phú Thuận, Quận 7, Hồ Chí Minh. 80m2, 3 tầng, 3 phòng ngủ, hẻm xe hơi, nội thất, sổ hồng.",
            placeholder="Paste mô tả bài đăng đầy đủ từ Alonhadat/Batdongsan...",
            height=100
        )

        if address_text.strip():
            address_lower = address_text.lower()
            matched_locality = None
            for locality in api_localities():
                if locality.replace("phường ", "").replace("xã ", "").replace("tp ", "") in address_lower:
                    matched_locality = locality
                    break

            if matched_locality:
                st.success(f"✅ Tìm thấy: **{matched_locality}**")
                parsed = api_parse(address_text)
                street_default = parsed.get("street", "") if "error" not in parsed else ""

                st.subheader("🗺️ Xác nhận địa chỉ")
                st.caption(f"💡 Tìm được đường: **{street_default}** (nếu trống = không có trong cache → dùng fallback)")

                # Show available streets for this locality (optional - collapsible)
                with st.expander("📍 Xem đường có sẵn trong cache"):
                    available_streets = geo.streets_of(matched_locality)
                    if available_streets:
                        st.write(f"**{len(available_streets)} đường trong cache** cho {matched_locality}:")
                        st.write(", ".join(available_streets[:15]))  # Show first 15
                    else:
                        st.write("(Không có đường nào trong cache)")

                street_input = st.text_input("Đường/Phố (chỉnh sửa nếu cần)", value=street_default or "")

                col_loc, col_house = st.columns(2)
                with col_loc:
                    st.subheader("🏷️ Phân loại")
                    # Property type default from parser
                    prop_type_default = parsed.get("property_type") or "nha_mat_tien"
                    prop_type_idx = list(PROPERTY_TYPES.keys()).index(prop_type_default) if prop_type_default in PROPERTY_TYPES else 0
                    property_type = st.radio("Loại nhà", list(PROPERTY_TYPES),
                                            format_func=PROPERTY_TYPES.get, horizontal=True, index=prop_type_idx)

                    # Budget range default from extracted price
                    price_range_default = parsed.get("price_range") or "low"
                    price_range_map = {"low": 0, "mid": 1, "high": 2}
                    budget_range = st.selectbox("Phân khúc giá", list(BUDGET_RANGES), format_func=BUDGET_RANGES.get,
                                               index=price_range_map.get(price_range_default, 0))

                    legal_status = st.selectbox("Pháp lý", list(LEGAL_STATUS), format_func=LEGAL_STATUS.get,
                                               index=list(LEGAL_STATUS.keys()).index(parsed.get("legal_status", "unknown")))

                    # Direction default from parser
                    direction_default = parsed.get("direction", "unknown")
                    direction_idx = list(DIRECTIONS.keys()).index(direction_default) if direction_default in DIRECTIONS else 0
                    direction = st.selectbox("Hướng nhà", list(DIRECTIONS), format_func=DIRECTIONS.get, index=direction_idx)

                with col_house:
                    st.subheader("📐 Thông số (tự động extract từ listing)")
                    c1, c2 = st.columns(2)
                    area_m2 = c1.number_input("Diện tích (m²)", 10.0, 1000.0, parsed.get("area_m2") or 80.0, step=5.0)
                    road_width_m = c2.number_input("Đường trước nhà (m)", 1.0, 60.0, parsed.get("road_width_m") or 6.0, step=0.5)
                    width_m = c1.number_input("Chiều ngang (m)", 1.0, 50.0, parsed.get("width_m") or 4.0, step=0.5)
                    length_m = c2.number_input("Chiều dài (m)", 1.0, 100.0, parsed.get("length_m") or 20.0, step=0.5)
                    num_floors = c1.number_input("Số tầng", 1, 15, parsed.get("num_floors") or 3)
                    num_bedrooms = c2.number_input("Số phòng ngủ", 1, 20, parsed.get("num_bedrooms") or 3)

                    st.subheader("✨ Tiện ích (tự detect)")
                    b1, b2, b3 = st.columns(3)
                    bin_flags = {
                        "kitchen_bin": b1.checkbox("Nhà bếp", True),
                        "dining_room_bin": b2.checkbox("Phòng ăn", True),
                        "terrace_bin": b3.checkbox("Sân thượng"),
                        "car_parking_bin": b1.checkbox("Chỗ để xe hơi"),
                    }
                    text_flags = {
                        "is_hem_xe_hoi": b2.checkbox("Hẻm xe hơi", parsed["is_hem_xe_hoi"]),
                        "is_no_hau": b3.checkbox("Nở hậu", parsed["is_no_hau"]),
                        "has_noi_that": b1.checkbox("Có nội thất", parsed["has_noi_that"]),
                        "is_gap": b2.checkbox("Cần bán gấp", parsed["is_gap"]),
                        "is_kinh_doanh": b3.checkbox("Tiện kinh doanh", parsed["is_kinh_doanh"]),
                    }

                st.divider()
                if st.button("💰 Định giá", type="primary", use_container_width=True):
                    # Use parsed street/locality if available, else fall back to form inputs
                    final_street = parsed.get("street", "") or street_input
                    final_locality = parsed.get("locality", "") or matched_locality

                    if not final_street.strip():
                        st.error("❌ Vui lòng nhập tên đường!")
                    else:
                        st.session_state.show_feedback = True
                        with st.spinner("Đang định giá..."):
                            # Debug: show what we're using
                            with st.expander("🔧 Debug Info"):
                                st.write(f"**Parsed street:** {parsed.get('street', '')}")
                                st.write(f"**Final street:** {final_street}")
                                st.write(f"**Parsed locality:** {parsed.get('locality', '')}")
                                st.write(f"**Matched locality:** {matched_locality}")
                                st.write(f"**Final locality:** {final_locality}")

                            try:
                                st.write("🔧 **Calling API for prediction...**")
                                price, info, row, xai_data = do_valuation(
                                    street=final_street,
                                    locality=final_locality,
                                    property_type=property_type, legal_status=legal_status, direction=direction,
                                    area_m2=area_m2, width_m=width_m, length_m=length_m,
                                    num_floors=num_floors, num_bedrooms=num_bedrooms, road_width_m=road_width_m,
                                    bin_flags=bin_flags, text_flags=text_flags,
                                )
                                # Store in session state for persistent feedback form
                                st.session_state.last_price = price
                                st.session_state.last_info = info
                                st.session_state.last_row = row
                                st.session_state.last_xai_data = xai_data
                            except Exception as e:
                                import traceback
                                error_details = traceback.format_exc()
                                st.error(f"❌ Exception in valuation:\n\n{error_details}")
                                st.stop()

                        if price is None:
                            error_msg = info if info else 'Unknown error - check logs'
                            st.error(f"❌ Lỗi xây dựng feature row:\n\n{error_msg}")
                            st.info("💡 Thử dùng **Form chi tiết** (tab ✍️) để nhập thủ công thay vì paste")
                        else:
                            mape_err = price * 0.1325

                            r1, r2, r3 = st.columns(3)
                            r1.metric("Giá dự đoán", f"{price / 1e9:,.2f} tỷ VND")
                            r2.metric("Khoảng ±MAPE", f"{max(price - mape_err, 0) / 1e9:,.1f} – {(price + mape_err) / 1e9:,.1f} tỷ")
                            r3.metric("Giá / m²", f"{price / area_m2 / 1e6:,.0f} triệu/m²")

                            m_col, f_col = st.columns([1, 1])

                            with m_col:
                                st.map(pd.DataFrame({"lat": [info["lat"]], "lon": [info["lon"]]}), zoom=14)
                                st.markdown("##### 📍 Thông tin tọa độ")
                                st.write(f"**Phường:** {matched_locality}")
                                st.write(f"**Đường:** {street_input}")
                                st.write(f"**Nguồn tọa độ:** {info['source']}")
                                st.write(f"**Lat/Lon:** {info['lat']:.6f}, {info['lon']:.6f}")
                                if info["poi_source"] == "overpass":
                                    st.info("Vị trí nằm ngoài vùng đã crawl - dùng Overpass API")

                            with f_col:
                                st.markdown("#### 🔍 64 Features")
                                feature_df = pd.DataFrame([
                                    {"Feature": k, "Value": f"{v:.4g}" if isinstance(v, float) else v}
                                    for k, v in sorted(row.items())
                                ])
                                st.dataframe(feature_df, hide_index=True, use_container_width=True, height=400)

                            st.divider()
                            st.markdown("### 🧠 Model Explainability (XAI)")

                            if xai_data:
                                # Model & Bucket info + Confidence
                                info_col1, info_col2, info_col3 = st.columns(3)

                                with info_col1:
                                    bucket = xai_data.get("bucket", "unknown")
                                    st.metric("📊 Bucket Used", f"{bucket.replace('_', ' ').title()}")

                                with info_col2:
                                    confidence = xai_data.get("confidence", 0)
                                    conf_color = "🟢" if confidence > 80 else "🟡" if confidence > 60 else "🔴"
                                    st.metric("🎯 Confidence", f"{conf_color} {confidence:.0f}%")

                                with info_col3:
                                    model_preds = xai_data.get("model_predictions", {})
                                    st.metric("📈 Models Used", f"{len(model_preds)}/3")

                                st.divider()

                                # Feature importance
                                importance = xai_data.get("importance", {})
                                if importance:
                                    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                                    xai_col1, xai_col2 = st.columns(2)

                                    with xai_col1:
                                        exp_df = pd.DataFrame([
                                            {"Feature": feat, "Importance %": f"{imp:.2f}"}
                                            for feat, imp in top_features
                                        ])
                                        st.markdown("**Top 10 Most Important Features:**")
                                        st.dataframe(exp_df, hide_index=True, use_container_width=True, height=300)

                                    with xai_col2:
                                        exp_chart_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
                                        st.bar_chart(exp_chart_df.set_index("Feature"), height=300, use_container_width=True)

                                    st.caption(
                                        "💡 Shows which property characteristics have the most impact on price predictions."
                                    )

                                # Model predictions breakdown
                                if model_preds:
                                    st.markdown("**Individual Model Predictions:**")
                                    import numpy as np
                                    model_df = pd.DataFrame([
                                        {
                                            "Model": k.split("_")[0].upper(),
                                            "Prediction (tỷ VND)": f"{np.expm1(v) / 1e9:.2f}",
                                            "Log-Value": f"{v:.3f}"
                                        }
                                        for k, v in sorted(model_preds.items())
                                    ])
                                    st.dataframe(model_df, hide_index=True, use_container_width=True)

                                    # Show prediction spread
                                    prices_vnd = [np.expm1(v) for v in model_preds.values()]
                                    min_price = min(prices_vnd)
                                    max_price = max(prices_vnd)
                                    avg_price = np.mean(prices_vnd)
                                    spread_pct = ((max_price - min_price) / avg_price) * 100 if avg_price > 0 else 0

                                    spread_col1, spread_col2, spread_col3 = st.columns(3)
                                    with spread_col1:
                                        st.metric("Min", f"{min_price / 1e9:.2f}T")
                                    with spread_col2:
                                        st.metric("Max", f"{max_price / 1e9:.2f}T")
                                    with spread_col3:
                                        st.metric("Spread", f"±{spread_pct/2:.1f}%")
                            else:
                                st.info("XAI data not available")
            else:
                st.warning("Không tìm thấy phường trong địa chỉ. Thử dùng Form chi tiết!")

    else:
        # Form chi tiết
        col_loc, col_house = st.columns(2)

        with col_loc:
            st.subheader("📍 Vị trí")
            localities = api_localities()
            default_idx = localities.index("phường bình thạnh") if "phường bình thạnh" in localities else 0
            locality = st.selectbox("Phường / Xã", localities, index=default_idx)

            streets = geo.streets_of(locality)
            street_choice = st.selectbox("Đường", streets + ["✏️ Khác (nhập tay)"])
            street = (st.text_input("Tên đường", placeholder="ví dụ: đường lê quang định")
                      if street_choice == "✏️ Khác (nhập tay)" else street_choice)

            st.subheader("🏷️ Phân loại")
            property_type = st.radio("Loại nhà", list(PROPERTY_TYPES),
                                      format_func=PROPERTY_TYPES.get, horizontal=True)
            budget_range  = st.selectbox("Phân khúc giá",
                                          list(BUDGET_RANGES), format_func=BUDGET_RANGES.get)
            legal_status  = st.selectbox("Pháp lý", list(LEGAL_STATUS), format_func=LEGAL_STATUS.get)
            direction     = st.selectbox("Hướng nhà", list(DIRECTIONS), format_func=DIRECTIONS.get)

        with col_house:
            st.subheader("📐 Thông số")
            c1, c2 = st.columns(2)
            area_m2      = c1.number_input("Diện tích (m²)", 10.0, 1000.0, 80.0, step=5.0)
            road_width_m = c2.number_input("Đường trước nhà (m)", 1.0, 60.0, 6.0, step=0.5)
            width_unknown = c1.checkbox("Không rõ chiều ngang")
            width_m  = c1.number_input("Chiều ngang (m)", 1.0, 50.0, 4.0, step=0.5, disabled=width_unknown)
            length_unknown = c2.checkbox("Không rõ chiều dài")
            length_m = c2.number_input("Chiều dài (m)", 1.0, 100.0, 20.0, step=0.5, disabled=length_unknown)
            if width_unknown:  width_m  = None
            if length_unknown: length_m = None
            num_floors    = c1.number_input("Số tầng", 1, 15, 3)
            num_bedrooms  = c2.number_input("Số phòng ngủ", 1, 20, 3)

            st.subheader("✨ Tiện ích & đặc điểm")
            b1, b2, b3 = st.columns(3)
            bin_flags = {
                "kitchen_bin":    b1.checkbox("Nhà bếp", True),
                "dining_room_bin": b2.checkbox("Phòng ăn", True),
                "terrace_bin":    b3.checkbox("Sân thượng"),
                "car_parking_bin": b1.checkbox("Chỗ để xe hơi"),
            }
            text_flags = {
                "is_hem_xe_hoi": b2.checkbox("Hẻm xe hơi / ô tô vào"),
                "is_no_hau":     b3.checkbox("Nở hậu"),
                "has_noi_that":  b1.checkbox("Có nội thất"),
                "is_gap":        b2.checkbox("Cần bán gấp"),
                "is_kinh_doanh": b3.checkbox("Tiện kinh doanh"),
            }

        st.divider()
        if st.button("💰 Định giá", type="primary", use_container_width=True):
            st.session_state.show_feedback = True
            with st.spinner("Đang gọi API để dự đoán giá..."):
                price, info, row, xai_data = do_valuation(
                    street=street, locality=locality,
                    property_type=property_type, legal_status=legal_status, direction=direction,
                    area_m2=area_m2, width_m=width_m, length_m=length_m,
                    num_floors=num_floors, num_bedrooms=num_bedrooms, road_width_m=road_width_m,
                    bin_flags=bin_flags, text_flags=text_flags,
                )
                # Store in session state so it persists after form submission
                st.session_state.last_price = price
                st.session_state.last_info = info
                st.session_state.last_row = row
                st.session_state.last_xai_data = xai_data

            if price is None:
                st.error("Không xác định được vị trí — kiểm tra lại tên đường / phường.")
            else:
                mape_err = price * 0.1325

                r1, r2, r3 = st.columns(3)
                r1.metric("Giá dự đoán", f"{price / 1e9:,.2f} tỷ VND")
                r2.metric("Khoảng tham khảo (±MAPE)", f"{max(price - mape_err, 0) / 1e9:,.1f} – {(price + mape_err) / 1e9:,.1f} tỷ")
                r3.metric("Giá / m²", f"{price / area_m2 / 1e6:,.0f} triệu/m²")

                m_col, d_col = st.columns([1, 1])
                with m_col:
                    st.map(pd.DataFrame({"lat": [info["lat"]], "lon": [info["lon"]]}), zoom=14)
                with d_col:
                    bucket_label = f"{BUDGET_RANGES[budget_range]} / {PROPERTY_TYPES[property_type]}"
                    st.write(f"**Bucket đã dùng:** {bucket_label}")
                    st.write(f"**Nguồn tọa độ:** {info['source']}")

                st.divider()
                st.markdown("### 🧠 Model Explainability (XAI)")

                if xai_data:
                    # Model & Bucket info + Confidence
                    info_col1, info_col2, info_col3 = st.columns(3)

                    with info_col1:
                        bucket = xai_data.get("bucket", "unknown")
                        st.metric("📊 Bucket Used", f"{bucket.replace('_', ' ').title()}")

                    with info_col2:
                        confidence = xai_data.get("confidence", 0)
                        conf_color = "🟢" if confidence > 80 else "🟡" if confidence > 60 else "🔴"
                        st.metric("🎯 Confidence", f"{conf_color} {confidence:.0f}%")

                    with info_col3:
                        model_preds = xai_data.get("model_predictions", {})
                        st.metric("📈 Models Used", f"{len(model_preds)}/3")

                    st.divider()

                    # Feature importance
                    importance = xai_data.get("importance", {})
                    if importance:
                        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                        xai_col1, xai_col2 = st.columns(2)

                        with xai_col1:
                            exp_df = pd.DataFrame([
                                {"Feature": feat, "Importance %": f"{imp:.2f}"}
                                for feat, imp in top_features
                            ])
                            st.markdown("**Top 10 Most Important Features:**")
                            st.dataframe(exp_df, hide_index=True, use_container_width=True, height=300)

                        with xai_col2:
                            exp_chart_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
                            st.bar_chart(exp_chart_df.set_index("Feature"), height=300, use_container_width=True)

                        st.caption(
                            "💡 Shows which property characteristics have the most impact on price predictions."
                        )

                    # Model predictions breakdown
                    if model_preds:
                        st.markdown("**Individual Model Predictions:**")
                        import numpy as np
                        model_df = pd.DataFrame([
                            {
                                "Model": k.split("_")[0].upper(),
                                "Prediction (tỷ VND)": f"{np.expm1(v) / 1e9:.2f}",
                                "Log-Value": f"{v:.3f}"
                            }
                            for k, v in sorted(model_preds.items())
                        ])
                        st.dataframe(model_df, hide_index=True, use_container_width=True)

                        # Show prediction spread
                        prices_vnd = [np.expm1(v) for v in model_preds.values()]
                        min_price = min(prices_vnd)
                        max_price = max(prices_vnd)
                        avg_price = np.mean(prices_vnd)
                        spread_pct = ((max_price - min_price) / avg_price) * 100 if avg_price > 0 else 0

                        spread_col1, spread_col2, spread_col3 = st.columns(3)
                        with spread_col1:
                            st.metric("Min", f"{min_price / 1e9:.2f}T")
                        with spread_col2:
                            st.metric("Max", f"{max_price / 1e9:.2f}T")
                        with spread_col3:
                            st.metric("Spread", f"±{spread_pct/2:.1f}%")
                else:
                    st.info("XAI data not available")

    # Show persistent feedback form for BOTH modes (after form submission rerun)
    if st.session_state.show_feedback and st.session_state.last_price is not None:
        price = st.session_state.last_price
        info = st.session_state.last_info
        row = st.session_state.last_row
        xai_data = st.session_state.last_xai_data

        st.divider()
        st.markdown("### 📝 Feedback & Learning (Persistent)")

        with st.form(key="feedback_form_persistent"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Đánh giá dự đoán:**")
                rating = st.radio("Dự đoán này chính xác không?",
                                ["👍 Accurate", "👎 Not accurate", "🤷 Not sure"],
                                horizontal=True, key="rating_persistent")
            with col2:
                st.markdown("**Giá thực tế (nếu biết):**")
                actual_price = st.number_input("Giá thực tế (tỷ VND)?", min_value=0.0, step=0.1, key="actual_persistent")

                submitted = st.form_submit_button("📤 Submit Feedback", use_container_width=True)

        if submitted:
            print("\n" + "="*80)
            print(f"✅ FEEDBACK PERSISTENT FORM SUBMITTED! Rating={rating}, ActualPrice={actual_price}B")
            print("="*80)
            st.info("⏳ Đang gửi feedback...")

            try:
                features_dict = {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in row.items()} if row else {}

                feedback_payload = {
                    "predicted_price_vnd": float(price),
                    "actual_price_vnd": float(actual_price * 1e9) if actual_price > 0 else None,
                    "rating": rating,
                    "bucket": xai_data.get("bucket", "mid") if xai_data else "mid",
                    "confidence": xai_data.get("confidence") if xai_data else 0,
                    "feature_version": FEATURE_VERSION,
                    "features_json": features_dict,
                    "timestamp": pd.Timestamp.now().isoformat(),
                }

                print(f"[PERSISTENT-FEEDBACK] Sending: {len(features_dict)} features")

                response = requests.post(
                    f"{API_BASE_URL}/api/feedback",
                    json=feedback_payload,
                    timeout=API_TIMEOUT
                )

                print(f"[PERSISTENT-FEEDBACK] Response status: {response.status_code}")
                print(f"[PERSISTENT-FEEDBACK] Response text: {response.text}")

                if response.status_code in [200, 201]:
                    st.success("✅ Feedback saved to Supabase!")
                    print("[PERSISTENT-FEEDBACK] Success!")
                    st.session_state.show_feedback = False
                    if actual_price > 0:
                        error_pct = abs((actual_price * 1e9 - price) / (actual_price * 1e9)) * 100
                        st.metric("Sai số", f"{error_pct:.1f}%")
                else:
                    try:
                        error_detail = response.json().get('detail', response.text)
                    except:
                        error_detail = response.text
                    print(f"[PERSISTENT-FEEDBACK] Error: {error_detail}")
                    st.error(f"❌ Failed ({response.status_code}): {error_detail}")
            except Exception as e:
                print(f"[PERSISTENT-FEEDBACK] Exception: {e}")
                import traceback
                traceback.print_exc()
                st.error(f"❌ Error: {str(e)}")

# =========================================================================
# TAB 2: MARKET ANALYSIS
# =========================================================================
with tab_analysis:
    st.subheader("📊 Phân tích Thị trường Bất Động Sản TP.HCM")
    st.caption("Heatmap giá, xu hướng thị trường từ dữ liệu crawl đã enrich.")

    df = load_bi_data()
    if df.empty:
        st.error("❌ Không tìm thấy dữ liệu BI hợp lệ từ Supabase Raw_Features table")
    else:
        # Filters
        fc = st.columns(3)
        property_options = ["Tất cả"] + sorted(df["property_type"].dropna().astype(str).unique())
        locality_options = ["Tất cả"] + sorted(df["locality"].dropna().astype(str).unique())
        date_min, date_max = df["post_day"].min().date(), df["post_day"].max().date()

        sel_prop  = fc[0].selectbox("Loại nhà", property_options, key="bi_prop")
        sel_loc   = fc[1].selectbox("Phường / Xã", locality_options, key="bi_loc")
        sel_dates = fc[2].date_input("Khoảng thời gian", (date_min, date_max),
                                      min_value=date_min, max_value=date_max, key="bi_dates")

        filt = df.copy()
        if sel_prop  != "Tất cả": filt = filt[filt["property_type"].astype(str) == sel_prop]
        if sel_loc   != "Tất cả": filt = filt[filt["locality"].astype(str) == sel_loc]
        if isinstance(sel_dates, tuple) and len(sel_dates) == 2:
            filt = filt[(filt["post_day"].dt.date >= sel_dates[0]) &
                        (filt["post_day"].dt.date <= sel_dates[1])]

        if filt.empty:
            st.warning("Bộ lọc hiện tại không có dữ liệu.")
        else:
            # Metrics
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("📍 Listings", f"{len(filt):,}")
            k2.metric("💰 Giá trung vị", f"{filt['price_billion_vnd'].median():,.2f} tỷ")
            k3.metric("📐 Giá/m² trung vị", f"{filt['price_per_m2_million'].median():,.0f} triệu/m²")
            k4.metric("🏘️ Phường/Xã", f"{filt['locality'].nunique():,}")

            # Heatmap + Trends
            st.divider()
            map_col, trend_col = st.columns([1.2, 1])

            with map_col:
                st.markdown("#### 🗺️ Heatmap Giá/m²")
                heat_df = filt[["lat", "lon", "price_per_m2_million"]].dropna().copy()
                if heat_df.empty:
                    st.info("Không đủ tọa độ.")
                elif pdk is None:
                    st.map(heat_df.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]])
                else:
                    st.pydeck_chart(pdk.Deck(
                        layers=[pdk.Layer("HeatmapLayer", data=heat_df,
                                          get_position="[lon, lat]",
                                          get_weight="price_per_m2_million",
                                          radius_pixels=55)],
                        initial_view_state=pdk.ViewState(latitude=float(heat_df["lat"].mean()),
                                                          longitude=float(heat_df["lon"].mean()),
                                                          zoom=10.2),
                        tooltip={"text": "Giá/m²: {price_per_m2_million} triệu"},
                    ), use_container_width=True)

            with trend_col:
                st.markdown("#### 📈 Xu hướng Thị trường")
                trend = (filt.groupby("month", as_index=False)
                         .agg(median_price_billion_vnd=("price_billion_vnd", "median"),
                              listing_count=("price_billion_vnd", "size"))
                         .sort_values("month"))
                if len(trend):
                    st.line_chart(trend.set_index("month")["median_price_billion_vnd"], use_container_width=True)
                    st.bar_chart(trend.set_index("month")["listing_count"], use_container_width=True)

            # Top localities
            st.divider()
            st.markdown("#### 🏆 Top 10 Phường/Xã theo Giá/m² Trung vị")
            rank = (filt.groupby("locality", as_index=False)
                    .agg(median_price_per_m2_million=("price_per_m2_million", "median"),
                         listings=("price_per_m2_million", "size"))
                    .sort_values("median_price_per_m2_million", ascending=False).head(10))
            st.dataframe(rank, hide_index=True, use_container_width=True)
            if len(rank):
                st.bar_chart(rank.set_index("locality")["median_price_per_m2_million"], use_container_width=True)

# =========================================================================
# TAB 3: FEEDBACK ANALYTICS
# =========================================================================
with tab_feedback:
    st.subheader("📈 Feedback Analytics Dashboard")
    st.caption("Track prediction accuracy and model performance from user feedback.")

    from app.core.feedback import (
        get_feedback_stats, get_feedback_trends, get_feedback_by_segment,
        get_feedback_distribution, get_best_predictions
    )

    # Load all feedback data
    stats = get_feedback_stats()

    if stats is None or stats.get("feedback_with_prices", 0) == 0:
        st.info("📭 No feedback data collected yet. Start making predictions and submit feedback!")
    else:
        # ===== SECTION 1: Summary Metrics =====
        st.markdown("### 📊 Summary Metrics")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Feedback", stats["total_feedback"])
        with col2:
            st.metric("With Prices", stats["feedback_with_prices"])
        with col3:
            mean_error = stats["mean_error_pct"]
            st.metric("Mean Error %", f"{mean_error:.2f}%",
                     delta=f"{'Under' if mean_error < 0 else 'Over'} predicted")
        with col4:
            st.metric("MAPE %", f"{stats['mape_pct']:.2f}%")
        with col5:
            bias = stats["model_bias_pct"]
            st.metric("Model Bias %", f"{bias:.1f}%",
                     delta=f"{'Over' if bias > 0 else 'Under'} predicting")

        st.divider()

        # ===== SECTION 2: Rating Distribution =====
        st.markdown("### 👍 Rating Distribution")
        rating_dist = stats.get("rating_distribution", {})
        if rating_dist:
            col1, col2, col3 = st.columns(3)
            with col1:
                accurate = rating_dist.get("Accurate (5)", 0)
                st.metric("👍 Accurate", accurate)
            with col2:
                not_sure = rating_dist.get("Not sure (3)", 0)
                st.metric("🤷 Not Sure", not_sure)
            with col3:
                not_accurate = rating_dist.get("Not accurate (2)", 0)
                st.metric("👎 Not Accurate", not_accurate)

            # Pie chart
            if sum(rating_dist.values()) > 0:
                st.bar_chart(pd.Series(rating_dist))

        st.divider()

        # ===== SECTION 3: Trends Over Time =====
        st.markdown("### 📈 Trends Over Time")
        trends = get_feedback_trends()
        if trends:
            trends_df = pd.DataFrame(trends)
            if not trends_df.empty:
                trend_col1, trend_col2 = st.columns(2)
                with trend_col1:
                    st.markdown("**Feedback Count per Day**")
                    st.bar_chart(trends_df.set_index("date")["feedback_count"])
                with trend_col2:
                    st.markdown("**Daily MAPE Trend**")
                    st.line_chart(trends_df.set_index("date")["daily_mape"])

        st.divider()

        # ===== SECTION 4: Segmentation Analysis =====
        st.markdown("### 🎯 Performance by Segment")
        segments = get_feedback_by_segment()

        if segments:
            seg_col1, seg_col2, seg_col3 = st.columns(3)

            # By bucket
            with seg_col1:
                st.markdown("**By Price Bucket**")
                by_bucket = pd.DataFrame(segments["by_bucket"])
                if not by_bucket.empty:
                    by_bucket_display = by_bucket[["bucket", "count", "mape"]].round(2)
                    st.dataframe(by_bucket_display, hide_index=True, use_container_width=True)
                    st.bar_chart(by_bucket.set_index("bucket")["mape"])

            # By property type
            with seg_col2:
                st.markdown("**By Property Type**")
                by_type = pd.DataFrame(segments["by_type"])
                if not by_type.empty:
                    by_type_display = by_type[["property_type", "count", "mape"]].round(2)
                    st.dataframe(by_type_display, hide_index=True, use_container_width=True)

            # Top localities
            with seg_col3:
                st.markdown("**Top 10 Localities**")
                by_locality = pd.DataFrame(segments["by_locality"])
                if not by_locality.empty:
                    by_locality_display = by_locality[["locality", "count", "mape"]].round(2)
                    st.dataframe(by_locality_display, hide_index=True, use_container_width=True)

        st.divider()

        # ===== SECTION 5: Best vs Worst Predictions =====
        col_best, col_worst = st.columns(2)

        # Best predictions
        with col_best:
            st.markdown("#### ✅ Best Predictions (Most Accurate)")
            best = get_best_predictions()
            if best:
                best_df = pd.DataFrame(best)
                best_df["predicted_billion_vnd"] = (best_df["predicted_price_vnd"] / 1e9).round(2)
                best_df["actual_billion_vnd"] = (best_df["actual_price_vnd"] / 1e9).round(2)
                best_df["error_%"] = best_df["abs_error_pct"].round(2)
                display_cols = ["street", "locality", "predicted_billion_vnd", "actual_billion_vnd", "error_%"]
                st.dataframe(best_df[display_cols], hide_index=True, use_container_width=True)
            else:
                st.info("No best predictions yet.")

        # Worst predictions
        with col_worst:
            st.markdown("#### ⚠️ Worst Predictions (Largest Errors)")
            worst = stats.get("worst_predictions", [])
            if worst:
                worst_df = pd.DataFrame(worst)
                worst_df["predicted_billion_vnd"] = (worst_df["predicted_price_vnd"] / 1e9).round(2)
                worst_df["actual_billion_vnd"] = (worst_df["actual_price_vnd"] / 1e9).round(2)
                worst_df["error_%"] = worst_df["abs_error_pct"].round(2)
                display_cols = ["street", "locality", "predicted_billion_vnd", "actual_billion_vnd", "error_%"]
                st.dataframe(worst_df[display_cols], hide_index=True, use_container_width=True)
            else:
                st.info("No worst predictions yet — all accurate!")
