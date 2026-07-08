"""
Interactive BI Dashboard - Plotly-based market analysis.
Shows price heatmaps, market trends, locality rankings, and prediction accuracy.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="BI Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_predictions_data():
    """Load prediction results."""
    predictions_file = Path(__file__).parent.parent / 'models' / 'data' / 'predictions_hybrid_latest.csv'
    if not predictions_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(predictions_file)
    required = ['price_billion_vnd', 'predicted_price_billion_vnd', 'area_m2', 'locality', 'lat', 'lon']
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    # Clean data
    df = df.dropna(subset=['price_billion_vnd', 'predicted_price_billion_vnd', 'area_m2', 'lat', 'lon'])
    df = df[(df['price_billion_vnd'] > 0.1) & (df['price_billion_vnd'] <= 100)]
    df['error_billion_vnd'] = df['predicted_price_billion_vnd'] - df['price_billion_vnd']
    df['ape_pct'] = np.abs(df['error_billion_vnd']) / df['price_billion_vnd'] * 100
    df['price_per_m2_million'] = (df['price_billion_vnd'] * 1000) / (df['area_m2'] + 1)
    df['pred_price_per_m2_million'] = (df['predicted_price_billion_vnd'] * 1000) / (df['area_m2'] + 1)

    return df

st.title("📊 House Price Prediction BI Dashboard")
st.markdown("Market heatmaps, trends, model accuracy, and locality rankings based on hybrid ensemble predictions.")

df = load_predictions_data()

if df.empty:
    st.warning("No prediction data available. Run inference first.")
    st.stop()

# ============================================================================
# KPI CARDS
# ============================================================================
st.divider()
st.subheader("📈 Key Metrics")

kpi_cols = st.columns(5)
with kpi_cols[0]:
    st.metric("Total Listings", f"{len(df):,}")
with kpi_cols[1]:
    st.metric("Median Price", f"{df['price_billion_vnd'].median():.2f}B")
with kpi_cols[2]:
    st.metric("MAPE", f"{df['ape_pct'].median():.1f}%")
with kpi_cols[3]:
    st.metric("MAE", f"{df['error_billion_vnd'].abs().mean():.2f}B")
with kpi_cols[4]:
    st.metric("Localities", f"{df['locality'].nunique()}")

# ============================================================================
# FILTERS
# ============================================================================
st.divider()
st.subheader("🔍 Filters")

filter_cols = st.columns(4)

with filter_cols[0]:
    price_range = st.slider(
        "Price Range (billion VND)",
        min_value=0.1,
        max_value=100.0,
        value=(0.1, 50.0),
        step=0.5
    )

with filter_cols[1]:
    area_range = st.slider(
        "Area Range (m²)",
        min_value=10,
        max_value=1000,
        value=(10, 500),
        step=10
    )

with filter_cols[2]:
    selected_localities = st.multiselect(
        "Localities",
        options=sorted(df['locality'].unique()),
        default=[]
    )

with filter_cols[3]:
    error_threshold = st.slider(
        "Max APE % (for accuracy charts)",
        min_value=5.0,
        max_value=50.0,
        value=20.0,
        step=1.0
    )

# Apply filters
df_filtered = df[
    (df['price_billion_vnd'] >= price_range[0]) &
    (df['price_billion_vnd'] <= price_range[1]) &
    (df['area_m2'] >= area_range[0]) &
    (df['area_m2'] <= area_range[1])
].copy()

if selected_localities:
    df_filtered = df_filtered[df_filtered['locality'].isin(selected_localities)]

st.info(f"Showing {len(df_filtered):,} of {len(df):,} listings after filters")

# ============================================================================
# HEATMAP & TRENDS
# ============================================================================
st.divider()
st.subheader("🗺️ Geographic Heatmap")

heatmap_col, info_col = st.columns([2, 1])

with heatmap_col:
    # Price per m² heatmap
    fig_heat = go.Figure(data=go.Scattermapbox(
        lat=df_filtered['lat'],
        lon=df_filtered['lon'],
        mode='markers',
        marker=dict(
            size=6,
            color=df_filtered['price_per_m2_million'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Price/m²<br>(Million VND)"),
            opacity=0.7
        ),
        text=df_filtered.apply(lambda r: f"Price: {r['price_billion_vnd']:.1f}B<br>Area: {r['area_m2']:.0f}m²<br>Price/m²: {r['price_per_m2_million']:.0f}M", axis=1),
        hoverinfo='text'
    ))

    fig_heat.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=10.8, lon=106.7),
            zoom=11
        ),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with info_col:
    st.markdown("**Heatmap Info**")
    st.metric("Min Price/m²", f"{df_filtered['price_per_m2_million'].min():.0f}M")
    st.metric("Max Price/m²", f"{df_filtered['price_per_m2_million'].max():.0f}M")
    st.metric("Median Price/m²", f"{df_filtered['price_per_m2_million'].median():.0f}M")

# ============================================================================
# ACTUAL VS PREDICTED
# ============================================================================
st.divider()
st.subheader("🎯 Model Accuracy")

acc_col1, acc_col2 = st.columns(2)

with acc_col1:
    st.markdown("**Actual vs Predicted Prices**")
    fig_scatter = px.scatter(
        df_filtered[df_filtered['ape_pct'] <= error_threshold],
        x='price_billion_vnd',
        y='predicted_price_billion_vnd',
        hover_data=['area_m2', 'locality', 'ape_pct'],
        labels={'price_billion_vnd': 'Actual (B)', 'predicted_price_billion_vnd': 'Predicted (B)'},
        color='ape_pct',
        color_continuous_scale='RdYlGn_r',
    )

    # Add diagonal line
    min_price = df_filtered['price_billion_vnd'].min()
    max_price = df_filtered['price_billion_vnd'].max()
    fig_scatter.add_trace(go.Scatter(
        x=[min_price, max_price],
        y=[min_price, max_price],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='gray')
    ))

    fig_scatter.update_layout(height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)

with acc_col2:
    st.markdown("**Error Distribution**")
    fig_error = go.Figure(data=[
        go.Histogram(
            x=df_filtered['ape_pct'],
            nbinsx=50,
            name='APE %',
            marker_color='indianred'
        )
    ])
    fig_error.add_vline(x=df_filtered['ape_pct'].median(), line_dash="dash", line_color="blue", annotation_text="Median")
    fig_error.update_layout(
        title="Absolute Percentage Error Distribution",
        xaxis_title="APE %",
        yaxis_title="Count",
        height=450
    )
    st.plotly_chart(fig_error, use_container_width=True)

# ============================================================================
# PRICE SEGMENTS
# ============================================================================
st.divider()
st.subheader("💰 Performance by Price Segment")

segments = [(0, 5), (5, 20), (20, 100)]
segment_stats = []

for seg_lo, seg_hi in segments:
    mask = (df_filtered['price_billion_vnd'] > seg_lo) & (df_filtered['price_billion_vnd'] <= seg_hi)
    seg_data = df_filtered[mask]

    if len(seg_data) > 0:
        segment_stats.append({
            'Segment': f'{seg_lo}–{seg_hi}B',
            'Count': len(seg_data),
            'Median Price': f"{seg_data['price_billion_vnd'].median():.2f}B",
            'MAPE': f"{seg_data['ape_pct'].median():.1f}%",
            'MAE': f"{seg_data['error_billion_vnd'].abs().mean():.2f}B",
            'R²': f"{(1 - (seg_data['error_billion_vnd']**2).sum() / ((seg_data['price_billion_vnd'] - seg_data['price_billion_vnd'].mean())**2).sum()):.3f}" if len(seg_data) > 1 else "N/A"
        })

segment_df = pd.DataFrame(segment_stats)
st.dataframe(segment_df, use_container_width=True, hide_index=True)

# ============================================================================
# LOCALITY RANKINGS
# ============================================================================
st.divider()
st.subheader("🏘️ Locality Rankings")

ranking_cols = st.columns(3)

with ranking_cols[0]:
    st.markdown("**By Median Price/m²**")
    locality_rank = (
        df_filtered.groupby('locality')
        .agg(median_price_m2=('price_per_m2_million', 'median'), count=('locality', 'size'))
        .sort_values('median_price_m2', ascending=False)
        .head(10)
    )
    fig_rank = px.bar(
        locality_rank.reset_index(),
        x='median_price_m2',
        y='locality',
        color='median_price_m2',
        color_continuous_scale='Turbo',
        orientation='h',
        labels={'median_price_m2': 'Price/m² (Million VND)', 'locality': 'Locality'},
    )
    fig_rank.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_rank, use_container_width=True)

with ranking_cols[1]:
    st.markdown("**By Listing Count**")
    locality_count = (
        df_filtered.groupby('locality')
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    fig_count = px.bar(
        locality_count.reset_index(name='count'),
        x='count',
        y='locality',
        color='count',
        color_continuous_scale='Blues',
        orientation='h',
        labels={'count': 'Count', 'locality': 'Locality'},
    )
    fig_count.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_count, use_container_width=True)

with ranking_cols[2]:
    st.markdown("**By Median MAPE %**")
    locality_accuracy = (
        df_filtered.groupby('locality')
        .agg(median_mape=('ape_pct', 'median'), count=('locality', 'size'))
        .query('count >= 5')  # Only localities with 5+ listings
        .sort_values('median_mape', ascending=True)
        .head(10)
    )
    fig_acc = px.bar(
        locality_accuracy.reset_index(),
        x='median_mape',
        y='locality',
        color='median_mape',
        color_continuous_scale='RdYlGn_r',
        orientation='h',
        labels={'median_mape': 'Median APE %', 'locality': 'Locality'},
    )
    fig_acc.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_acc, use_container_width=True)

# ============================================================================
# PRICE TREND
# ============================================================================
st.divider()
st.subheader("📉 Price Trends by Area Size")

if 'post_day' in df_filtered.columns:
    df_filtered['post_day'] = pd.to_datetime(df_filtered['post_day'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['post_day'])

    if not df_filtered.empty:
        df_filtered['month'] = df_filtered['post_day'].dt.to_period('M').astype(str)

        trend_data = (
            df_filtered.groupby('month')
            .agg(
                median_price=('price_billion_vnd', 'median'),
                count=('price_billion_vnd', 'size')
            )
            .reset_index()
        )

        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])

        fig_trend.add_trace(
            go.Scatter(x=trend_data['month'], y=trend_data['median_price'], name='Median Price', line=dict(color='blue')),
            secondary_y=False,
        )

        fig_trend.add_trace(
            go.Bar(x=trend_data['month'], y=trend_data['count'], name='Listing Count', marker=dict(color='lightblue'), opacity=0.3),
            secondary_y=True,
        )

        fig_trend.update_layout(height=400, hovermode='x unified')
        fig_trend.update_xaxes(title_text="Month")
        fig_trend.update_yaxes(title_text="Median Price (B)", secondary_y=False)
        fig_trend.update_yaxes(title_text="Count", secondary_y=True)

        st.plotly_chart(fig_trend, use_container_width=True)

st.divider()
st.caption("Dashboard updated with latest predictions. Refresh to see new data.")
