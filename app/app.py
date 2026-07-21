"""Định giá nhà TP.HCM — Parser + Model Valuation
Chạy: streamlit run app/app.py
Để xem BI Dashboard: streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import GeoLookup
from inference import load_models, build_row, apply_locality_encoding, predict_price
from parsers import parse_listing, extract_street_from_address

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Định giá nhà TP.HCM",
    page_icon="🏠",
    layout="wide",
)

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
# Load assets
# ---------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    models, meta, medians = load_models()
    return models, meta, medians, GeoLookup()

models, meta, medians, geo = load_assets()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏠 Định giá nhà TP.HCM")
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
# Input mode selection
# ---------------------------------------------------------------------------
mode = st.radio("📝 Cách nhập", ["Paste địa chỉ nhanh", "Form chi tiết"], horizontal=True)

if mode == "Paste địa chỉ nhanh":
    st.subheader("📍 Dán địa chỉ đầy đủ")
    address_text = st.text_area(
        "Ví dụ: Vlasta Premier Phú Thuận, Đường Đào Trí, Phường Phú Thuận, Quận 7, Hồ Chí Minh",
        placeholder="Paste địa chỉ tại đây...",
        height=80
    )

    if address_text.strip():
        address_lower = address_text.lower()
        matched_locality = None
        for locality in geo.localities():
            if locality.replace("phường ", "").replace("xã ", "").replace("tp ", "") in address_lower:
                matched_locality = locality
                break

        if matched_locality:
            st.success(f"✅ Tìm thấy: **{matched_locality}**")

            # Auto-parse listing data
            parsed = parse_listing(address_text)
            street_default = extract_street_from_address(address_text)

            st.subheader("🗺️ Xác nhận địa chỉ")
            street_input = st.text_input("Đường/Phố (chỉnh sửa nếu cần)", value=street_default or "")

            col_loc, col_house = st.columns(2)
            with col_loc:
                st.subheader("🏷️ Phân loại")
                property_type = st.radio("Loại nhà", list(PROPERTY_TYPES),
                                        format_func=PROPERTY_TYPES.get, horizontal=True)
                budget_range = st.selectbox("Phân khúc giá", list(BUDGET_RANGES), format_func=BUDGET_RANGES.get)
                legal_status = st.selectbox("Pháp lý", list(LEGAL_STATUS), format_func=LEGAL_STATUS.get,
                                           index=list(LEGAL_STATUS.keys()).index(parsed.get("legal_status", "unknown")))
                direction = st.selectbox("Hướng nhà", list(DIRECTIONS), format_func=DIRECTIONS.get)

            with col_house:
                st.subheader("📐 Thông số (tự động extract từ listing)")
                c1, c2 = st.columns(2)
                area_m2 = c1.number_input("Diện tích (m²)", 10.0, 1000.0, parsed["area_m2"], step=5.0)
                road_width_m = c2.number_input("Đường trước nhà (m)", 1.0, 60.0, parsed["road_width_m"], step=0.5)
                width_m = c1.number_input("Chiều ngang (m)", 1.0, 50.0, parsed["width_m"], step=0.5)
                length_m = c2.number_input("Chiều dài (m)", 1.0, 100.0, parsed["length_m"], step=0.5)
                num_floors = c1.number_input("Số tầng", 1, 15, parsed["num_floors"])
                num_bedrooms = c2.number_input("Số phòng ngủ", 1, 20, parsed["num_bedrooms"])

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
                if not street_input.strip():
                    st.error("❌ Vui lòng nhập tên đường!")
                else:
                    with st.spinner("Đang định giá..."):
                        row, info = build_row(
                            medians, geo,
                            street=street_input,
                            locality=matched_locality,
                            property_type=property_type, legal_status=legal_status, direction=direction,
                            area_m2=area_m2, width_m=width_m, length_m=length_m,
                            num_floors=num_floors, num_bedrooms=num_bedrooms, road_width_m=road_width_m,
                            bin_flags=bin_flags, text_flags=text_flags,
                        )

                    if row is None:
                        st.error("Không geocode được địa chỉ")
                    else:
                        row = apply_locality_encoding(row, meta, matched_locality)
                        price = predict_price(models, meta, row, budget_range)
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
        else:
            st.warning("Không tìm thấy phường trong địa chỉ. Thử dùng Form chi tiết!")

else:
    # Form chi tiết
    col_loc, col_house = st.columns(2)

    with col_loc:
        st.subheader("📍 Vị trí")
        localities = geo.localities()
        default_idx = localities.index("phường bình thạnh") if "phường bình thạnh" in localities else 0
        locality = st.selectbox("Phường / Xã", localities, index=default_idx)

        streets = geo.streets_of(locality)
        street_choice = st.selectbox("Đường", streets + ["✏️ Khác (nhập tay)"])
        street = (st.text_input("Tên đường", placeholder="ví dụ: đường lê quang định")
                  if street_choice == "✏️ Khác (nhập tay)" else street_choice)

        st.subheader("🏷️ Phân loại")
        property_type = st.radio("Loại nhà", list(PROPERTY_TYPES),
                                  format_func=PROPERTY_TYPES.get, horizontal=True)
        budget_range  = st.selectbox("Phân khúc giá (dùng để chọn model bucket)",
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
        with st.spinner("Đang tra cứu vị trí và tính feature địa lý..."):
            row, info = build_row(
                medians, geo,
                street=street, locality=locality,
                property_type=property_type, legal_status=legal_status, direction=direction,
                area_m2=area_m2, width_m=width_m, length_m=length_m,
                num_floors=num_floors, num_bedrooms=num_bedrooms, road_width_m=road_width_m,
                bin_flags=bin_flags, text_flags=text_flags,
            )

        if row is None:
            st.error("Không xác định được vị trí — kiểm tra lại tên đường / phường.")
        else:
            row = apply_locality_encoding(row, meta, locality)
            price = predict_price(models, meta, row, budget_range)
            mape_err = price * 0.1325

            r1, r2, r3 = st.columns(3)
            r1.metric("Giá dự đoán", f"{price:,.2f} tỷ VND")
            r2.metric("Khoảng tham khảo (±MAPE)", f"{max(price - mape_err, 0):,.1f} – {price + mape_err:,.1f} tỷ")
            r3.metric("Giá / m²", f"{price * 1000 / area_m2:,.0f} triệu/m²")

            m_col, d_col = st.columns([1, 1])
            with m_col:
                st.map(pd.DataFrame({"lat": [info["lat"]], "lon": [info["lon"]]}), zoom=14)
            with d_col:
                bucket_label = f"{BUDGET_RANGES[budget_range]} / {PROPERTY_TYPES[property_type]}"
                st.write(f"**Bucket đã dùng:** {bucket_label}")
                st.write(f"**Nguồn tọa độ:** {info['source']}")
                loc_price = row.get("locality_price_median", 0.0)
                if loc_price > 0:
                    st.write(f"**Giá nền phường (train set):** {loc_price / 1e9:.2f} tỷ")
                if info["poi_source"] == "overpass":
                    st.info("Vị trí nằm ngoài vùng đã crawl — feature địa lý tính qua Overpass API và lưu cache.")
                elif info["cache_dist_km"] > 2:
                    st.warning(f"Feature địa lý lấy từ điểm crawl cách {info['cache_dist_km']:.1f} km — độ chính xác có thể giảm.")
                with st.expander("Feature địa lý (từ pipeline ETL)"):
                    poi_df = pd.DataFrame(
                        [(k, f"{round(v, 3):g}" if v is not None else "thiếu")
                         for k, v in info["pois"].items()],
                        columns=["feature", "giá trị"],
                    )
                    st.dataframe(poi_df, hide_index=True, use_container_width=True)

st.divider()
st.info("📊 Xem BI Dashboard: `streamlit run app/dashboard.py`")
