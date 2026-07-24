#!/bin/bash

# Production startup script for FastAPI application

set -e  # Exit on error

echo "=========================================="
echo "Real Estate Valuation API - Startup"
echo "=========================================="

# Check environment variables
echo "[1/4] Checking environment variables..."
required_vars=("SUPABASE_URL" "SUPABASE_SERVICE_KEY")
for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "❌ Error: $var is not set"
    exit 1
  fi
  echo "✅ $var is set"
done

# Load models
echo "[2/4] Pre-loading ML models..."
python -c "
from app.core.models import get_models
try:
  models, meta, _, geo = get_models()
  print('✅ Models loaded successfully')
  print(f'  - LightGBM: {type(models[0]).__name__}')
  print(f'  - XGBoost: {type(models[1]).__name__}')
  print(f'  - CatBoost: {type(models[2]).__name__}')
except Exception as e:
  print(f'⚠️  Warning: Model loading failed: {e}')
  print('  This is normal if data files are not available')
"

# Check Supabase connectivity
echo "[3/4] Checking Supabase connectivity..."
python -c "
from supabase import create_client
import os

try:
  url = os.getenv('SUPABASE_URL')
  key = os.getenv('SUPABASE_SERVICE_KEY')
  client = create_client(url, key)
  # Try a simple query
  response = client.table('feedback').select('count', count='exact').execute()
  print('✅ Supabase connection OK')
  print(f'  - Feedback records: {response.count}')
except Exception as e:
  print(f'❌ Error: Supabase connection failed: {e}')
  exit(1)
"

# Start API server
echo "[4/4] Starting FastAPI server..."
echo "=========================================="
echo "🚀 API Server is running!"
echo "📍 Health check: GET /health"
echo "📖 API docs: GET /docs"
echo "=========================================="

# Use environment variable for port (Railway sets $PORT)
PORT=${PORT:-8000}
exec uvicorn app.main:app --host 0.0.0.0 --port $PORT
