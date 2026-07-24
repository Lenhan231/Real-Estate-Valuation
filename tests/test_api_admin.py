"""Tests for admin API endpoints."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


class TestAdminEndpoints:
    """Test admin endpoints."""

    def test_drift_status_no_data(self, client):
        """Test drift status endpoint returns no_data when no feedback."""
        with patch("models.retraining.get_retraining_data", return_value=None):
            response = client.get("/api/admin/drift-status")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "no_data"

    def test_model_comparison_no_data(self, client):
        """Test model comparison endpoint returns no_data when no feedback."""
        with patch("models.retraining.get_retraining_data", return_value=None):
            response = client.get("/api/admin/model-comparison")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "no_data"

    def test_retrain_insufficient_data(self, client):
        """Test retraining with insufficient feedback data."""
        with patch("models.retraining.get_retraining_data", return_value=None):
            response = client.post("/api/admin/retrain")

            assert response.status_code == 400
            assert "Insufficient feedback" in response.json()["detail"]


class TestAdminApiContracts:
    """Test that admin endpoints follow correct API contracts."""

    def test_drift_status_response_schema(self, client):
        """Test drift status response has required fields."""
        with patch("models.retraining.get_retraining_data", return_value=None):
            response = client.get("/api/admin/drift-status")

            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "message" in data

    def test_model_comparison_response_schema(self, client):
        """Test model comparison response has required fields."""
        with patch("models.retraining.get_retraining_data", return_value=None):
            response = client.get("/api/admin/model-comparison")

            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "message" in data

    def test_retrain_insufficient_data_response_schema(self, client):
        """Test retrain response format on insufficient data."""
        with patch("models.retraining.get_retraining_data", return_value=None):
            response = client.post("/api/admin/retrain")

            assert response.status_code == 400
            data = response.json()
            assert "detail" in data
