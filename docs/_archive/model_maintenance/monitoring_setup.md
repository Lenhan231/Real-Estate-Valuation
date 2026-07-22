# 📊 Performance Monitoring Setup

**Track model predictions & actual values in production**

## 🎯 Cần Monitor Cái Gì?

1. **Prediction Accuracy** - MAPE, MAE, R²
2. **Prediction Latency** - Tốc độ inference
3. **Data Distribution** - Input data changes
4. **Error Patterns** - Errors theo properties/locations
5. **Model Drift** - Performance degradation over time

## 📋 Monitoring Metrics

```python
# Core metrics to log
{
    "timestamp": "2026-07-20T10:30:00Z",
    "prediction_id": "pred_12345",
    "property_id": "prop_789",
    "predicted_price": 2500000000,
    "actual_price": 2650000000,
    "error_pct": 5.66,
    "absolute_error": 150000000,
    "input_features": {
        "area": 85.5,
        "bedrooms": 3,
        "district": "District 1",
        "location_lat": 10.7769,
        "location_lng": 106.6963
    },
    "model_version": "v1.0",
    "inference_time_ms": 45
}
```

## 📝 Implementation Steps

### 1. Create Monitoring Logger

```python
# monitoring/logger.py
import json
from datetime import datetime
from pathlib import Path

class PredictionLogger:
    def __init__(self, log_file="logs/predictions.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_prediction(self, property_id, predicted_price, actual_price=None, 
                      model_version="v1.0", features=None, inference_time=None):
        """Log a single prediction"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "property_id": property_id,
            "predicted_price": float(predicted_price),
            "actual_price": float(actual_price) if actual_price else None,
            "error_pct": self._calc_error(predicted_price, actual_price),
            "model_version": model_version,
            "features": features,
            "inference_time_ms": inference_time
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def _calc_error(self, pred, actual):
        if actual is None:
            return None
        return abs((pred - actual) / actual) * 100
```

### 2. Calculate Metrics Periodically

```python
# monitoring/metrics.py
import pandas as pd

def calculate_metrics(log_file, days=7):
    """Calculate metrics from last N days"""
    df = pd.read_json(log_file, lines=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter last N days
    recent = df[df['timestamp'] >= pd.Timestamp.now() - pd.Timedelta(days=days)]
    
    # Only use predictions with actual values
    complete = recent[recent['actual_price'].notna()]
    
    if len(complete) == 0:
        return None
    
    metrics = {
        "n_predictions": len(complete),
        "mape": (complete['error_pct'].mean()),
        "mae": complete['absolute_error'].mean(),
        "median_error": complete['error_pct'].median(),
        "std_error": complete['error_pct'].std(),
        "avg_latency_ms": complete['inference_time_ms'].mean(),
        "min_latency_ms": complete['inference_time_ms'].min(),
        "max_latency_ms": complete['inference_time_ms'].max(),
    }
    
    return metrics
```

### 3. Alert System

```python
# monitoring/alerts.py
def check_alerts(metrics, thresholds=None):
    """Check if metrics exceed thresholds"""
    if thresholds is None:
        thresholds = {
            "mape": 25.0,        # Alert if MAPE > 25%
            "latency_ms": 500,   # Alert if latency > 500ms
            "predictions": 10    # Need at least 10 predictions
        }
    
    alerts = []
    
    if metrics['n_predictions'] < thresholds['predictions']:
        alerts.append(f"⚠️ Low prediction volume: {metrics['n_predictions']}")
    
    if metrics['mape'] > thresholds['mape']:
        alerts.append(f"🚨 MAPE degraded: {metrics['mape']:.2f}% > {thresholds['mape']}%")
    
    if metrics['avg_latency_ms'] > thresholds['latency_ms']:
        alerts.append(f"🚨 High latency: {metrics['avg_latency_ms']:.0f}ms")
    
    return alerts
```

## 📊 Dashboard Setup

Create `monitoring/dashboard.py`:

```python
# Simple monitoring dashboard
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Model Monitoring", layout="wide")
st.title("🤖 Model Performance Monitoring")

# Load recent predictions
df = pd.read_json("logs/predictions.jsonl", lines=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAPE", "18.01%", "+0.5%")
col2.metric("Predictions (7d)", "245", "+12")
col3.metric("Avg Latency", "45ms", "-5ms")
col4.metric("Model Version", "v1.0", "Stable")

# Charts
st.subheader("Error Over Time")
fig = px.scatter(df, x='timestamp', y='error_pct', hover_data=['property_id'])
st.plotly_chart(fig, use_container_width=True)

# Alerts
st.subheader("Active Alerts")
alerts = check_alerts(calculate_metrics("logs/predictions.jsonl"))
if alerts:
    for alert in alerts:
        st.warning(alert)
else:
    st.success("✅ All metrics normal")
```

Run dashboard:
```bash
streamlit run monitoring/dashboard.py
```

## 🚀 Integration with App

```python
# app/app_simple.py - Add monitoring to predictions
from monitoring.logger import PredictionLogger
import time

logger = PredictionLogger()

@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    
    # Make prediction
    pred_price = model.predict(features)
    
    # Log prediction
    inference_time = (time.time() - start) * 1000
    logger.log_prediction(
        property_id=request.json.get("property_id"),
        predicted_price=pred_price,
        model_version="v1.0",
        features=features,
        inference_time=inference_time
    )
    
    return {"predicted_price": pred_price}
```

## 📈 Frequency

- **Real-time**: Log every prediction
- **Hourly**: Calculate rolling metrics
- **Daily**: Generate summary report
- **Weekly**: Review trends, check for drift
- **Monthly**: Full audit, update documentation
