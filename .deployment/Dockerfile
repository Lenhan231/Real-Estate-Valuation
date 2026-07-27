# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies (production only)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by LightGBM and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies only
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Run FastAPI server with dynamic port support for cloud deployment
CMD exec sh -c 'exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}'
