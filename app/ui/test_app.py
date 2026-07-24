"""Minimal test app to verify Streamlit works"""
import streamlit as st
import os

st.write("✅ Streamlit is running!")
st.write(f"PORT: {os.getenv('PORT', '8501')}")
st.write(f"API_URL: {os.getenv('API_URL', 'http://localhost:8000')}")
