"""Market Trend Analysis Dashboard
Chạy: streamlit run app/dashboard.py
Để định giá: streamlit run app/app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import pydeck as pdk
except Exception:
    pdk = None

ROOT = Path(__file__).resolve().parent.parent
BI_DATA_FILE = ROOT / "data" / "processed" / "alonhadat_features.csv"

st.set_page_config(
    page_title="📊 Phân tích thị trường BĐS TP.HCM",
    page_icon="📈",
    layout="wide",
)

st.title("📊 Phân tích Thị trường Bất Động Sản TP.HCM")
st.caption("Heatmap giá, xu hướng thị trường từ dữ liệu crawl đã enrich.")

# ---------------------------------------------------------------------------
# Load BI data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_bi_data():
    if not BI_DATA_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(BI_DATA_FILE)
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


df = load_bi_data()
if df.empty:
    st.error("Không tìm thấy dữ liệu BI hợp lệ trong data/processed/alonhadat_features.csv")
    st.stop()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
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
    st.stop()

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("📍 Listings", f"{len(filt):,}")
k2.metric("💰 Giá trung vị", f"{filt['price_billion_vnd'].median():,.2f} tỷ")
k3.metric("📐 Giá/m² trung vị", f"{filt['price_per_m2_million'].median():,.0f} triệu/m²")
k4.metric("🏘️ Phường/Xã", f"{filt['locality'].nunique():,}")

# ---------------------------------------------------------------------------
# Heatmap + Trends
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Top localities
# ---------------------------------------------------------------------------
st.divider()
st.markdown("#### 🏆 Top 10 Phường/Xã theo Giá/m² Trung vị")
rank = (filt.groupby("locality", as_index=False)
        .agg(median_price_per_m2_million=("price_per_m2_million", "median"),
             listings=("price_per_m2_million", "size"))
        .sort_values("median_price_per_m2_million", ascending=False).head(10))
st.dataframe(rank, hide_index=True, use_container_width=True)
if len(rank):
    st.bar_chart(rank.set_index("locality")["median_price_per_m2_million"], use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.info("💡 Để định giá nhà: `streamlit run app/app.py`")
