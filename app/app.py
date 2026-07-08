"""Web demo định giá nhà TP.HCM — chạy: streamlit run app/app.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import pydeck as pdk
except Exception:
    pdk = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import GeoLookup
from inference import load_model, build_row, apply_locality_encoding, predict_price

ROOT = Path(__file__).resolve().parent.parent
BI_DATA_FILE = ROOT / "data" / "processed" / "alonhadat_features.csv"

st.set_page_config(page_title="Định giá nhà TP.HCM", page_icon="🏠", layout="wide")

PROPERTY_TYPES = {'nha_mat_tien': 'Nhà mặt tiền', 'nha_trong_hem': 'Nhà trong hẻm'}
LEGAL_STATUS = {'so_hong_so_do': 'Sổ hồng / Sổ đỏ', 'giay_to_hop_le': 'Giấy tờ hợp lệ', 'unknown': 'Không rõ'}
DIRECTIONS = {
    'unknown': 'Không rõ', 'dong': 'Đông', 'tay': 'Tây', 'nam': 'Nam', 'bac': 'Bắc',
    'dong_nam': 'Đông Nam', 'dong_bac': 'Đông Bắc', 'tay_nam': 'Tây Nam', 'tay_bac': 'Tây Bắc',
}


@st.cache_data(show_spinner=False)
def load_bi_data():
    if not BI_DATA_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(BI_DATA_FILE)
    required = ["lat", "lon", "price_vnd", "area_m2", "post_day", "locality", "property_type", "region"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df.dropna(subset=["lat", "lon", "price_vnd", "area_m2", "post_day"]).copy()
    df["post_day"] = pd.to_datetime(df["post_day"], errors="coerce")
    df = df.dropna(subset=["post_day"]).copy()
    df["price_billion_vnd"] = df["price_vnd"] / 1e9
    df["price_per_m2_million"] = (df["price_vnd"] / df["area_m2"]) / 1e6
    df["month"] = df["post_day"].dt.to_period("M").dt.to_timestamp()
    return df


def render_dashboard():
    st.subheader("📊 Business Intelligence Dashboard")
    st.caption("Heatmap và xu hướng giá được dựng từ dữ liệu crawl đã enrich trong pipeline.")

    df = load_bi_data()
    if df.empty:
        st.warning("Không tìm thấy dữ liệu BI hợp lệ trong data/processed/alonhadat_features.csv")
        return

    filter_cols = st.columns(3)
    property_options = ["Tất cả"] + sorted(df["property_type"].dropna().astype(str).unique().tolist())
    locality_options = ["Tất cả"] + sorted(df["locality"].dropna().astype(str).unique().tolist())
    date_min = df["post_day"].min().date()
    date_max = df["post_day"].max().date()

    selected_property = filter_cols[0].selectbox("Loại nhà", property_options)
    selected_locality = filter_cols[1].selectbox("Phường / Xã", locality_options)
    selected_dates = filter_cols[2].date_input(
        "Khoảng thời gian",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    filtered = df.copy()
    if selected_property != "Tất cả":
        filtered = filtered[filtered["property_type"].astype(str) == selected_property]
    if selected_locality != "Tất cả":
        filtered = filtered[filtered["locality"].astype(str) == selected_locality]

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
        filtered = filtered[
            (filtered["post_day"].dt.date >= start_date) &
            (filtered["post_day"].dt.date <= end_date)
        ]

    if filtered.empty:
        st.warning("Bộ lọc hiện tại không có dữ liệu.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Listings", f"{len(filtered):,}")
    k2.metric("Giá trung vị", f"{filtered['price_billion_vnd'].median():,.2f} tỷ")
    k3.metric("Giá/m² trung vị", f"{filtered['price_per_m2_million'].median():,.0f} triệu/m²")
    k4.metric("Phường/Xã", f"{filtered['locality'].nunique():,}")

    map_col, trend_col = st.columns([1.2, 1])
    with map_col:
        st.markdown("#### Heatmap giá/m²")
        heat_df = filtered[["lat", "lon", "price_per_m2_million"]].dropna().copy()
        if heat_df.empty:
            st.info("Không đủ tọa độ để dựng heatmap.")
        elif pdk is None:
            st.map(heat_df.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]])
        else:
            deck = pdk.Deck(
                layers=[
                    pdk.Layer(
                        "HeatmapLayer",
                        data=heat_df,
                        get_position="[lon, lat]",
                        get_weight="price_per_m2_million",
                        radius_pixels=55,
                    )
                ],
                initial_view_state=pdk.ViewState(
                    latitude=float(heat_df["lat"].mean()),
                    longitude=float(heat_df["lon"].mean()),
                    zoom=10.2,
                    pitch=0,
                ),
                tooltip={"text": "Giá/m²: {price_per_m2_million} triệu"},
            )
            st.pydeck_chart(deck, use_container_width=True)

    with trend_col:
        st.markdown("#### Xu hướng thị trường")
        trend = (
            filtered.groupby("month", as_index=False)
            .agg(
                median_price_billion_vnd=("price_billion_vnd", "median"),
                listing_count=("price_billion_vnd", "size"),
                median_price_per_m2_million=("price_per_m2_million", "median"),
            )
            .sort_values("month")
        )
        if len(trend):
            st.line_chart(trend.set_index("month")["median_price_billion_vnd"])
            st.bar_chart(trend.set_index("month")["listing_count"])
        else:
            st.info("Không đủ dữ liệu thời gian để vẽ xu hướng.")

    st.markdown("#### Top phường/xã theo giá/m² trung vị")
    locality_rank = (
        filtered.groupby("locality", as_index=False)
        .agg(
            median_price_per_m2_million=("price_per_m2_million", "median"),
            listings=("price_per_m2_million", "size"),
        )
        .sort_values(["median_price_per_m2_million", "listings"], ascending=[False, False])
        .head(10)
    )
    st.dataframe(locality_rank, hide_index=True, width="stretch")
    if len(locality_rank):
        st.bar_chart(locality_rank.set_index("locality")["median_price_per_m2_million"])


@st.cache_resource
def load_assets():
    model, meta, medians = load_model()
    return model, meta, medians, GeoLookup()


model, meta, medians, geo = load_assets()

st.title("🏠 Định giá nhà TP.HCM")
metrics = meta.get('metrics', {})
caption_parts = [
    f"Model: **{meta['model_type']}**",
    f"R² {metrics.get('r2_score', float('nan')):.3f}" if 'r2_score' in metrics else "R² n/a",
    f"MAE {metrics.get('mae', float('nan')):.2f} tỷ VND (test set)" if 'mae' in metrics else "MAE n/a",
    f"MAPE {metrics.get('mape', float('nan')):.2f}% (test set)" if 'mape' in metrics else "MAPE n/a",
    f"train trên {meta['train_size']:,} listings",
]
st.caption(
    " · ".join(caption_parts)
)

col_loc, col_house = st.columns(2)

with col_loc:
    st.subheader("📍 Vị trí")
    localities = geo.localities()
    default_idx = localities.index('phường bình thạnh') if 'phường bình thạnh' in localities else 0
    locality = st.selectbox("Phường / Xã", localities, index=default_idx)

    streets = geo.streets_of(locality)
    street_choice = st.selectbox("Đường", streets + ['✏️ Khác (nhập tay)'])
    if street_choice == '✏️ Khác (nhập tay)':
        street = st.text_input("Tên đường", placeholder="ví dụ: đường lê quang định")
    else:
        street = street_choice

    property_type = st.radio("Loại nhà", list(PROPERTY_TYPES), format_func=PROPERTY_TYPES.get, horizontal=True)
    legal_status = st.selectbox("Pháp lý", list(LEGAL_STATUS), format_func=LEGAL_STATUS.get)
    direction = st.selectbox("Hướng nhà", list(DIRECTIONS), format_func=DIRECTIONS.get)

with col_house:
    st.subheader("📐 Thông số")
    c1, c2 = st.columns(2)
    area_m2 = c1.number_input("Diện tích (m²)", 10.0, 1000.0, 80.0, step=5.0)
    road_width_m = c2.number_input("Đường trước nhà (m)", 1.0, 60.0, 6.0, step=0.5)
    width_unknown = c1.checkbox("Không rõ chiều ngang")
    width_m = c1.number_input("Chiều ngang (m)", 1.0, 50.0, 4.0, step=0.5, disabled=width_unknown)
    length_unknown = c2.checkbox("Không rõ chiều dài")
    length_m = c2.number_input("Chiều dài (m)", 1.0, 100.0, 20.0, step=0.5, disabled=length_unknown)
    if width_unknown:
        width_m = None
    if length_unknown:
        length_m = None
    num_floors = c1.number_input("Số tầng", 1, 15, 3)
    num_bedrooms = c2.number_input("Số phòng ngủ", 1, 20, 3)

    st.subheader("✨ Tiện ích & đặc điểm")
    b1, b2, b3 = st.columns(3)
    bin_flags = {
        'kitchen_bin': b1.checkbox("Nhà bếp", True),
        'dining_room_bin': b2.checkbox("Phòng ăn", True),
        'terrace_bin': b3.checkbox("Sân thượng"),
        'car_parking_bin': b1.checkbox("Chỗ để xe hơi"),
    }
    title_flags = {
        'title_hem_xe_hoi': b2.checkbox("Hẻm xe hơi"),
        'title_thang_may': b3.checkbox("Thang máy"),
        'title_ham': b1.checkbox("Hầm"),
        'title_can_goc': b2.checkbox("Căn góc / 2 mặt tiền"),
        'title_no_hau': b3.checkbox("Nở hậu"),
        'title_kinh_doanh': b1.checkbox("Tiện kinh doanh"),
        'title_biet_thu': b2.checkbox("Biệt thự / villa"),
        # nhà mặt tiền thì tin rao gần như chắc chắn có chữ "mặt tiền"
        'title_mat_tien': property_type == 'nha_mat_tien',
    }

st.divider()

if st.button("💰 Định giá", type="primary", width="stretch"):
    with st.spinner("Đang tra cứu vị trí và tính feature địa lý..."):
        row, info = build_row(
            meta['features'], medians, geo,
            street=street, locality=locality,
            property_type=property_type, legal_status=legal_status, direction=direction,
            area_m2=area_m2, width_m=width_m, length_m=length_m,
            num_floors=num_floors, num_bedrooms=num_bedrooms, road_width_m=road_width_m,
            bin_flags=bin_flags, title_flags=title_flags,
        )

    if row is None:
        st.error("Không xác định được vị trí — kiểm tra lại tên đường / phường.")
    else:
        row = apply_locality_encoding(row, meta, locality)
        price = predict_price(model, meta, row)
        mae = meta['metrics']['mae']

        r1, r2, r3 = st.columns(3)
        r1.metric("Giá dự đoán", f"{price:,.2f} tỷ VND")
        r2.metric("Khoảng tham khảo (± MAE)", f"{max(price - mae, 0):,.1f} – {price + mae:,.1f} tỷ")
        r3.metric("Giá / m²", f"{price * 1000 / area_m2:,.0f} triệu/m²")

        m_col, d_col = st.columns([1, 1])
        with m_col:
            st.map(pd.DataFrame({'lat': [info['lat']], 'lon': [info['lon']]}), zoom=14)
        with d_col:
            st.write(f"**Nguồn tọa độ:** {info['source']}")
            st.write(f"**Giá nền của phường (train set):** {row['locality_price_median']:.2f} tỷ")
            if info['poi_source'] == 'overpass':
                st.info("Vị trí nằm ngoài vùng đã crawl — feature địa lý được tính "
                        "trực tiếp qua Overpass API (client của pipeline ETL) và đã lưu cache.")
            elif info['cache_dist_km'] > 2:
                st.warning(
                    f"Feature địa lý lấy từ điểm crawl cách đây {info['cache_dist_km']:.1f} km — "
                    "độ chính xác có thể giảm."
                )
            with st.expander("Feature địa lý đã tính (từ pipeline ETL)"):
                poi_df = pd.DataFrame(
                    [(k, f"{round(v, 3):g}" if v is not None else "thiếu (dùng cờ missing)")
                     for k, v in info['pois'].items()],
                    columns=['feature', 'giá trị'],
                )
                st.dataframe(poi_df, hide_index=True, width="stretch")

st.divider()
render_dashboard()
