"""Tests for prediction API endpoints."""
import pytest
from unittest.mock import patch, MagicMock


class TestPredictEndpoint:
    """Test prediction endpoint."""

    def test_predict_missing_required_fields(self, client, prediction_sample):
        """Test prediction with missing required fields."""
        incomplete = {"street": "Test Street"}  # Missing most fields

        response = client.post("/api/predict", json=incomplete)

        assert response.status_code == 422  # Validation error

    def test_predict_price_type_validation(self, client, prediction_sample):
        """Test that prices must be numbers."""
        prediction_sample["area_m2"] = "not_a_number"

        response = client.post("/api/predict", json=prediction_sample)

        assert response.status_code == 422


class TestLocalitiesEndpoint:
    """Test localities endpoint."""

    def test_localities_endpoint_exists(self, client):
        """Test that localities endpoint is accessible."""
        mock_geo = MagicMock()
        mock_geo.localities.return_value = ["Phường Bình Thạnh", "Phường 1"]

        with patch("app.services.inference.get_geo", return_value=mock_geo):
            response = client.get("/api/localities")

            assert response.status_code == 200
            data = response.json()
            assert "localities" in data
            assert len(data["localities"]) == 2


class TestPredictApiContracts:
    """Test that prediction endpoints follow correct API contracts."""

    def test_predict_requires_json(self, client, prediction_sample):
        """Test that predict endpoint requires JSON."""
        response = client.post("/api/predict", data="not json")

        # Should return error for invalid content type or JSON parse error
        assert response.status_code >= 400
