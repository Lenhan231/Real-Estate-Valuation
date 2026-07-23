#!/usr/bin/env python
"""Minimal FastAPI with bulletproof error handling."""
import sys
import os

# Write to stderr immediately (unbuffered)
sys.stderr.write("[PRE-IMPORT] Python started\n")
sys.stderr.flush()

try:
    sys.stderr.write(f"[PRE-IMPORT] PORT env var: {os.getenv('PORT', 'NOT SET')}\n")
    sys.stderr.flush()

    sys.stderr.write("[IMPORT] Starting FastAPI import...\n")
    sys.stderr.flush()
    from fastapi import FastAPI
    sys.stderr.write("[IMPORT] FastAPI imported OK\n")
    sys.stderr.flush()

    sys.stderr.write("[IMPORT] Starting uvicorn import...\n")
    sys.stderr.flush()
    import uvicorn
    sys.stderr.write("[IMPORT] uvicorn imported OK\n")
    sys.stderr.flush()

    sys.stderr.write("[STARTUP] Creating FastAPI app...\n")
    sys.stderr.flush()
    app = FastAPI()

    @app.get("/")
    def read_root():
        return {"message": "API running", "port": os.getenv("PORT", "8501")}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    if __name__ == "__main__":
        port = int(os.getenv("PORT", "8501"))
        sys.stderr.write(f"[STARTUP] Running uvicorn on port {port}...\n")
        sys.stderr.flush()
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")

except Exception as e:
    sys.stderr.write(f"[ERROR] {type(e).__name__}: {e}\n")
    sys.stderr.flush()
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
