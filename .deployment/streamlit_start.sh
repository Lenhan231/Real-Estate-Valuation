#!/bin/bash
set -e

# Use PORT environment variable if set, otherwise default to 8501
PORT=${PORT:-8501}

echo "Starting Streamlit on port $PORT..."
exec python -m streamlit run app/ui/streamlit_app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --logger.level=debug
