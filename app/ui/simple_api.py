"""Minimal FastAPI to test if services can run on Render."""
from fastapi import FastAPI
import os
import sys

print(f"[STARTUP] Importing FastAPI...", flush=True)
app = FastAPI()
print(f"[STARTUP] FastAPI imported", flush=True)

@app.get("/")
def read_root():
    return {
        "message": "Streamlit UI API is running",
        "port": os.getenv("PORT", "8501"),
        "api_url": os.getenv("API_URL", "http://localhost:8000")
    }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    print(f"[STARTUP] Main block entered", flush=True)
    import uvicorn
    port = int(os.getenv("PORT", "8501"))
    print(f"[STARTUP] Starting uvicorn on 0.0.0.0:{port}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")
