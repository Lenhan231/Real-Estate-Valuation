#!/usr/bin/env python
"""Streamlit launcher with environment variable support."""
import os
import sys
import subprocess

port = os.getenv("PORT", "8501")
app = "app/ui/test_app.py"

print(f"[INFO] Starting Streamlit on port {port}...")
print(f"[INFO] Running: streamlit run {app} --server.port {port} --server.address 0.0.0.0 --server.headless true")

cmd = [
    "streamlit",
    "run",
    app,
    "--server.port", port,
    "--server.address", "0.0.0.0",
    "--server.headless=true",
]

print(f"[INFO] Command: {' '.join(cmd)}")
sys.stdout.flush()

os.execvp("streamlit", cmd)
