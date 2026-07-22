"""Configuration and settings."""
import os
from pathlib import Path


class Settings:
    """Application settings."""

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

    # Models
    MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"
    TRAINING_DATA = PROJECT_ROOT / "data" / "processed" / "model_training_data.csv"

    # API
    API_TITLE = "Real Estate Valuation API"
    API_VERSION = "2.0.0"
    API_DESCRIPTION = "Complete backend for real estate valuation system"


settings = Settings()
