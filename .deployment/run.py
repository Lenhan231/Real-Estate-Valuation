#!/usr/bin/env python
"""Wrapper to catch and display any startup errors."""
import sys
import os

print(f"[DEBUG] Python version: {sys.version}")
print(f"[DEBUG] Working directory: {os.getcwd()}")
print(f"[DEBUG] PORT: {os.getenv('PORT', 'NOT SET')}")
print(f"[DEBUG] API_URL: {os.getenv('API_URL', 'NOT SET')}")

try:
    print("[DEBUG] Attempting to import streamlit...")
    import streamlit as st
    print(f"[DEBUG] ✓ Streamlit imported successfully (v{st.__version__})")

    print("[DEBUG] Starting streamlit run...")
    sys.argv = [
        "streamlit",
        "run",
        "app/ui/streamlit_app.py",
        f"--server.port={os.getenv('PORT', '8501')}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--logger.level=debug",
    ]

    from streamlit.cli import main
    main()

except Exception as e:
    print(f"[ERROR] Exception occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
