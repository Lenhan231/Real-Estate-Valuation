#!/bin/bash

# Start both API and Streamlit for Real Estate Valuation App
# Usage: bash run.sh

echo "🚀 Starting Real Estate Valuation App..."
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi

# Start API in background
echo "📡 Starting API on port 8000..."
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait for API to start
sleep 3

# Start Streamlit in background
echo "🎨 Starting Streamlit on port 8500..."
streamlit run app/ui/streamlit_app.py --server.port 8500 &
STREAMLIT_PID=$!

echo ""
echo "✅ Both services started!"
echo ""
echo "📡 API:       http://localhost:8000/docs"
echo "🎨 Streamlit: http://localhost:8500"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for both processes
wait $API_PID $STREAMLIT_PID
