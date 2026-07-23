"""Pytest configuration and shared fixtures."""
import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Mock Supabase before any imports
sys.modules['supabase'] = MagicMock()


@pytest.fixture
def client():
    """FastAPI test client."""
    with patch('supabase.create_client'):
        from app.main import app
        return TestClient(app)


@pytest.fixture
def mock_supabase():
    """Mock Supabase client."""
    mock = MagicMock()
    mock.table.return_value.select.return_value.execute.return_value = MagicMock(
        data=[], error=None
    )
    return mock


@pytest.fixture
def feedback_sample():
    """Sample feedback data."""
    return {
        "predicted_price_vnd": 5e9,
        "actual_price_vnd": 5.5e9,
        "rating": "👍 Accurate",
        "bucket": "mid",
        "confidence": 85.5,
        "feature_version": 1,
        "features_json": {
            "street": "Đường Test",
            "locality": "Phường Bình Thạnh",
            "area_m2": 100.0,
            "num_floors": 3,
        },
        "timestamp": "2026-07-23T10:00:00",
    }


@pytest.fixture
def prediction_sample():
    """Sample prediction request."""
    return {
        "street": "Đường Lê Quang Định",
        "locality": "phường bình thạnh",
        "property_type": "nha_mat_tien",
        "legal_status": "so_hong_so_do",
        "direction": "dong",
        "area_m2": 80.0,
        "width_m": 4.0,
        "length_m": 20.0,
        "num_floors": 3,
        "num_bedrooms": 3,
        "road_width_m": 6.0,
        "bin_flags": {
            "kitchen_bin": True,
            "dining_room_bin": True,
            "terrace_bin": False,
            "car_parking_bin": False,
        },
        "text_flags": {
            "is_hem_xe_hoi": False,
            "is_no_hau": False,
            "has_noi_that": False,
            "is_gap": False,
            "is_kinh_doanh": False,
        },
    }


@pytest.fixture
def retraining_feedback_df():
    """Sample feedback DataFrame for retraining."""
    data = {
        "area_m2": [80.0, 100.0, 120.0],
        "num_floors": [3, 3, 4],
        "num_bedrooms": [3, 3, 4],
        "road_width_m": [6.0, 8.0, 6.0],
        "width_m": [4.0, 5.0, 4.5],
        "length_m": [20.0, 20.0, 25.0],
        "price_vnd": [5e9, 6e9, 7e9],
    }

    # Add more features to match model requirements
    df = pd.DataFrame(data)

    # Add dummy features (these would be properly engineered in real scenarios)
    for i in range(70):
        df[f"feature_{i}"] = np.random.randn(len(df))

    return df


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-key-123")
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")


@pytest.fixture(autouse=True)
def auto_mock_env(mock_env_vars):
    """Automatically mock env vars for all tests."""
    pass


@pytest.fixture
def mock_supabase_response():
    """Mock successful Supabase response."""
    response = MagicMock()
    response.data = [
        {
            "id": 1,
            "predicted_price_vnd": 5e9,
            "actual_price_vnd": 5.5e9,
            "rating": 5,
            "bucket": "mid",
            "confidence": 85.5,
            "timestamp": "2026-07-23T10:00:00",
            "created_at": "2026-07-23T10:00:00",
            "features_json": {"street": "Test St", "locality": "Test Ward"},
        }
    ]
    response.error = None
    return response
