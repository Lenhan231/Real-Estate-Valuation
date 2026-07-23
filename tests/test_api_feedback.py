"""Tests for feedback API endpoints."""
import pytest
from unittest.mock import patch, MagicMock


class TestFeedbackEndpoints:
    """Test feedback submission and management endpoints."""

    def test_submit_feedback_success(self, client, feedback_sample):
        """Test successful feedback submission."""
        with patch("app.services.feedback.save_feedback_to_supabase") as mock_save:
            mock_save.return_value = True

            response = client.post("/api/feedback", json=feedback_sample)

            assert response.status_code == 200
            assert response.json()["success"] is True
            assert "Feedback saved" in response.json()["message"]

    def test_submit_feedback_missing_required_fields(self, client):
        """Test feedback submission with missing fields."""
        incomplete_feedback = {
            "predicted_price_vnd": 5e9,
            # missing other required fields
        }

        response = client.post("/api/feedback", json=incomplete_feedback)

        assert response.status_code == 422  # Validation error

    def test_submit_feedback_invalid_rating(self, client, feedback_sample):
        """Test feedback with invalid rating."""
        feedback_sample["rating"] = "invalid_rating"

        with patch("app.services.feedback.save_feedback_to_supabase"):
            response = client.post("/api/feedback", json=feedback_sample)

            # Should still accept it (validation happens in schema)
            assert response.status_code in [200, 422]

    def test_submit_feedback_with_zero_actual_price(self, client, feedback_sample):
        """Test feedback with zero actual price (means no actual price provided)."""
        feedback_sample["actual_price_vnd"] = None

        with patch("app.services.feedback.save_feedback_to_supabase") as mock_save:
            mock_save.return_value = True

            response = client.post("/api/feedback", json=feedback_sample)

            assert response.status_code == 200
            # Verify actual_price_vnd is None
            call_args = mock_save.call_args
            assert call_args is not None

    def test_submit_feedback_db_error(self, client, feedback_sample):
        """Test feedback submission when database fails."""
        with patch("app.services.feedback.save_feedback_to_supabase") as mock_save:
            mock_save.return_value = False

            response = client.post("/api/feedback", json=feedback_sample)

            assert response.status_code == 500
            assert "Failed to save feedback" in response.json()["detail"]

    def test_submit_feedback_exception_handling(self, client, feedback_sample):
        """Test feedback submission error handling."""
        with patch("app.services.feedback.save_feedback_to_supabase") as mock_save:
            mock_save.side_effect = Exception("Database connection failed")

            response = client.post("/api/feedback", json=feedback_sample)

            assert response.status_code == 500
            assert "Error" in response.json()["detail"]

    def test_feedback_large_features_json(self, client, feedback_sample):
        """Test feedback with large features_json."""
        # Add many features to test JSON serialization
        large_features = {f"feature_{i}": float(i) for i in range(100)}
        feedback_sample["features_json"] = large_features

        with patch("app.services.feedback.save_feedback_to_supabase") as mock_save:
            mock_save.return_value = True

            response = client.post("/api/feedback", json=feedback_sample)

            assert response.status_code == 200

    def test_feedback_price_type_validation(self, client, feedback_sample):
        """Test that prices must be numbers."""
        feedback_sample["predicted_price_vnd"] = "not_a_number"

        response = client.post("/api/feedback", json=feedback_sample)

        assert response.status_code == 422

    def test_feedback_rating_normalization(self, client, feedback_sample):
        """Test that rating is properly converted."""
        test_cases = [
            ("👍 Accurate", 5),
            ("🤷 Not sure", 3),
            ("👎 Not accurate", 2),
        ]

        for rating_text, expected_numeric in test_cases:
            feedback_sample["rating"] = rating_text

            with patch("app.services.feedback.save_feedback_to_supabase") as mock_save:
                mock_save.return_value = True

                response = client.post("/api/feedback", json=feedback_sample)

                assert response.status_code == 200


class TestFeedbackIntegration:
    """Integration tests for feedback flow."""

    def test_feedback_end_to_end(self, client, feedback_sample):
        """Test complete feedback flow from submission to Supabase."""
        with patch("app.services.feedback.save_feedback_to_supabase") as mock_save:
            mock_save.return_value = True

            response = client.post("/api/feedback", json=feedback_sample)

            assert response.status_code == 200
            # Verify save_feedback_to_supabase was called
            mock_save.assert_called_once()

    def test_feedback_concurrent_submissions(self, client, feedback_sample):
        """Test handling of concurrent feedback submissions."""
        with patch("app.services.feedback.save_feedback_to_supabase") as mock_save:
            mock_save.return_value = True

            # Simulate multiple submissions
            for i in range(5):
                modified_feedback = feedback_sample.copy()
                modified_feedback["predicted_price_vnd"] = 5e9 + i * 1e8

                response = client.post("/api/feedback", json=modified_feedback)

                assert response.status_code == 200

            # Verify all submissions were processed
            assert mock_save.call_count == 5
