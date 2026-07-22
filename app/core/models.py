"""ML models loader (singleton pattern)."""
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

from .config import settings


class ModelsLoader:
    """Singleton for loading and caching ML models."""

    _instance = None
    _models = None
    _meta = None
    _medians = None
    _geo = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> Tuple[Dict, Dict, Any, Any]:
        """Load models if not already loaded."""
        if self._models is None:
            self._load_models()
        return self._models, self._meta, self._medians, self._geo

    def _load_models(self):
        """Load all models and dependencies."""
        try:
            # Add scripts path for preprocessing import
            scripts_path = settings.PROJECT_ROOT / "models" / "scripts"
            if str(scripts_path) not in sys.path:
                sys.path.insert(0, str(scripts_path))

            from app.core.inference import load_models
            from app.core.geo import GeoLookup

            print("[MODELS] Loading ML models...")
            self._models, self._meta, self._medians = load_models()
            self._geo = GeoLookup()
            print("[MODELS] ✅ Models loaded successfully")

        except Exception as e:
            print(f"[MODELS] ❌ Failed to load models: {e}")
            raise


# Global loader instance
models_loader = ModelsLoader()


def get_models() -> Tuple[Dict, Dict, Any, Any]:
    """Get loaded models (lazy loading)."""
    return models_loader.load()


def get_geo():
    """Get GeoLookup instance."""
    _, _, _, geo = models_loader.load()
    return geo
